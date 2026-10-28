"""
Charge-aware grouping of MS1 signals into isotopologue envelopes and singletons.

Struct-of-arrays by design: peaks stay in one `SpectrumArray`, and a parallel
`group_labels` array (one entry per peak, -1 = noise/unassigned) keys each peak to
a signal group. Small per-group arrays (one entry per group) hold the group-level
labels the MDN needs and fills. Nothing is instantiated per signal, so this stays
cheap on high-throughput data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .envelopes import (
    SpectrumArray,
    ISOTOPE_SPACING,
    find_peaks_by_spacing,
)
from .halogen import envelope_is_halogen, HALOGEN_M2_OFFSET
from .ions import deconv_mass


H2_LOSS = 2.01565  # Da, in-source -H2

# A genuine M0 two below the (tallest) seed means the seed is a Cl/Br M+2.
# Br1 is the only pattern with seed/M0 near 1 and a small M+4.
# seed/M0, M+4/seed (IsoSpecPy, C20 scaffold):
#   Br: 1.01, 0.035 | ClBr: 1.33, 0.27 | Cl4: 1.32, 0.50 | Br2: 1.98, 0.51 | Br3: 2.95, 1.0
# so anything taller than Br1 needs an M+4 above the seed too.
_MAX_BR1_RATIO = 1.1
_MIN_UPPER_RATIO = 0.2


def _halogen_ladder(n: NDArray) -> NDArray:
    """Offset of the peak n steps below a Cl/Br seed: n/2 halogen steps, odd n via 13C."""
    return np.ceil(n / 2) * HALOGEN_M2_OFFSET - (n % 2) * ISOTOPE_SPACING


def _h2_loss_members(
    spec_arr: SpectrumArray,
    seed_idx: int,
    member_idxs: NDArray[np.intp],
    z: int,
    tol: float,
) -> NDArray[np.bool_]:
    """
    Mask over `member_idxs`: True for members below the seed that belong to an
    in-source -H2 species (a copy of the seed's envelope shifted down 2.0157 Da),
    not to the seed's own envelope.

    The seed is the tallest peak, so a genuine member below it is the low end of a
    Cl/Br envelope (or 13C, for very large ions).
    - Mass: each lower member is matched to the nearest of three offset ladders:
        Cl/Br, 13C (one step only), or H2 loss + Cl/Br. Members nearest the H2
        ladder are dropped.
        The H2 ladder is ~9 mDa above 13C and ~18 mDa above Cl/Br. Nearest-ladder
        rather than a tolerance means low-accuracy data degrades to occasional
        misses instead of being gated out.
    - Intensity: if seed/M0 is too tall for Br1, yet there is no matching M+4
        above the seed (see `_MAX_BR1_RATIO`), every lower member is dropped.
        Independent of mass accuracy.
    """
    mz = spec_arr['mz'][member_idxs]
    intsy = spec_arr['intsy'][member_idxs]
    seed_mz = float(spec_arr['mz'][seed_idx])
    seed_intsy = float(spec_arr['intsy'][seed_idx])
    d = (seed_mz - mz) * z   # offset below the seed, in Da
    n = np.rint(d / ISOTOPE_SPACING)
    below = n >= 1

    to_halogen = np.abs(d - _halogen_ladder(n))
    # 13C below the seed means carbon alone makes M+n tallest: ~1.8 kDa for n=1,
    # beyond small molecules for n>=2 -- where H2 + Cl/Br lands on 13C (4.0137 vs 4.0134).
    to_13c = np.where(n == 1, np.abs(d - ISOTOPE_SPACING), np.inf)
    to_h2 = np.abs(d - (H2_LOSS + _halogen_ladder(n - 2)))
    drop = below & (to_h2 < np.minimum(to_halogen, to_13c))

    m0 = below & ~drop & (n == 2)
    if m0.any():
        # Whole spectrum, not members: a real M+4 can sit under the noise floor.
        upper = np.abs(spec_arr['mz'] - (seed_mz + 2 * ISOTOPE_SPACING / z)) <= tol
        upper_intsy = float(spec_arr['intsy'][upper].sum())
        if (seed_intsy / float(intsy[m0].sum()) > _MAX_BR1_RATIO
                and upper_intsy / seed_intsy < _MIN_UPPER_RATIO):
            drop |= below
    return drop


def _count_sub_integer(
    spec_arr: SpectrumArray,
    member_idxs: NDArray[np.intp],
    seed_mz: float,
    tol: float,
) -> int:
    """
    Number of members at *fractional* isotope positions -- offsets from the seed
    that are not near an integer multiple of ISOTOPE_SPACING.

    A z=1 envelope only produces integer-multiple offsets, so these fractional
    peaks are exactly the evidence that a higher charge (z>1) is real rather than a
    stray peak. Requiring several of them stops a single spurious half-Da peak from
    being read as a 2+ envelope.
    """
    if len(member_idxs) == 0:
        return 0
    offs = np.abs(spec_arr['mz'][member_idxs] - seed_mz)
    nearest_integer = np.round(offs / ISOTOPE_SPACING) * ISOTOPE_SPACING
    return int(np.sum(np.abs(offs - nearest_integer) > tol))


def _fractional_m1_leads(
    spec_arr: SpectrumArray,
    member_idxs: NDArray[np.intp],
    seed_mz: float,
    z: int,
    tol: float,
) -> bool:
    """
    True if the charge-z M+1 (seed + spacing/z) is present and taller than the
    peak at the z=1 M+1 slot (seed + spacing).

    Single-peak evidence for z>1, for envelopes whose other fractional
    isotopologues (e.g. a 2+ M+3) are too weak to see. A stray half-Da peak next
    to a real z=1 envelope is shorter than that envelope's M+1, while a real 2+
    M+1 is taller than its M+2 (unless Cl/Br, which shows M+3 anyway).
    """
    mz = spec_arr['mz'][member_idxs]
    intsy = spec_arr['intsy'][member_idxs]
    m1 = np.abs(mz - (seed_mz + ISOTOPE_SPACING / z)) <= tol
    if not m1.any():
        return False
    z1_m1 = intsy[np.abs(mz - (seed_mz + ISOTOPE_SPACING)) <= tol].sum()
    return float(intsy[m1].max()) > float(z1_m1)


@dataclass(slots=True)
class NoiseThreshold:
    """
    Which peaks survive to be grouped. Unmet criteria are ignored (None); those
    that are set are combined with AND. `min_rel` is a fraction of the base peak.
    """
    min_abs: float | None = None
    min_rel: float | None = None
    top_n: int | None = None

    def mask(self, spec_arr: SpectrumArray) -> NDArray[np.bool_]:
        intsy = spec_arr['intsy']
        keep = np.ones(len(spec_arr), dtype=bool)
        if self.min_abs is not None:
            keep &= intsy >= self.min_abs
        if self.min_rel is not None and len(intsy) > 0:
            keep &= intsy >= self.min_rel * intsy.max()
        if self.top_n is not None and self.top_n < len(intsy):
            # Keep the top_n most intense peaks (among those still kept).
            order = np.argsort(intsy)[::-1]
            top = order[: self.top_n]
            top_mask = np.zeros(len(intsy), dtype=bool)
            top_mask[top] = True
            keep &= top_mask
        return keep


@dataclass(slots=True)
class GroupedSpectrum:
    """
    A spectrum partitioned into signal groups (isotope envelopes + singletons).

    `group_labels` is peak-aligned (len == len(spec_arr)); every other array is
    group-aligned (len == n_groups). Group ids are dense: 0 .. n_groups-1.
    `adduct_label`, `M`, and `component_id` are populated later by the MDN.
    """
    spec_arr: SpectrumArray
    group_labels: NDArray[np.intp]          # (n_peaks,), -1 = noise/unassigned
    charge: NDArray[np.intp]                 # (n_groups,)
    mono_idx: NDArray[np.intp]               # (n_groups,), index into spec_arr
    deconv_mass: NDArray[np.float64]         # (n_groups,)
    is_halogen: NDArray[np.bool_]            # (n_groups,)
    apex_intensity: NDArray[np.float64]      # (n_groups,), max peak intensity of the group
    adduct_label: NDArray[np.object_] = field(default=None, repr=False)   # (n_groups,)
    M: NDArray[np.float64] = field(default=None, repr=False)              # (n_groups,)
    component_id: NDArray[np.intp] = field(default=None, repr=False)      # (n_groups,)

    def __post_init__(self):
        n = self.n_groups
        if self.adduct_label is None:
            self.adduct_label = np.full(n, None, dtype=object)
        if self.M is None:
            self.M = np.full(n, np.nan, dtype=np.float64)
        if self.component_id is None:
            self.component_id = np.full(n, -1, dtype=np.intp)

    @property
    def n_groups(self) -> int:
        return len(self.charge)

    def peaks_of(self, gid: int) -> SpectrumArray:
        """The peaks belonging to group `gid`, as a SpectrumArray view."""
        return self.spec_arr[self.group_labels == gid]

    def mono_mz(self, gid: int) -> float:
        """m/z of the monoisotopic (lowest-m/z) peak of group `gid`."""
        return float(self.spec_arr['mz'][self.mono_idx[gid]])

    def group_containing(self, mz: float, tol: float = 0.01) -> int | None:
        """
        Group id of the assigned peak nearest to `mz` within `tol`, else None.

        Used to select the base (precursor) envelope from a precursor m/z.
        """
        assigned = self.group_labels >= 0
        if not np.any(assigned):
            return None
        idxs = np.where(assigned)[0]
        d = np.abs(self.spec_arr['mz'][idxs] - mz)
        j = int(np.argmin(d))
        if d[j] > tol:
            return None
        return int(self.group_labels[idxs[j]])


def group_signals(
    spec_arr: SpectrumArray,
    *,
    max_charge: int = 3,
    isotope_tol: float = 0.02,
    max_isotopes: int = 8,
    noise: NoiseThreshold | None = None,
    detect_halogens: bool = False,
    min_charge_evidence: int = 2,
) -> GroupedSpectrum:
    """
    Group peaks into charge-resolved isotope envelopes (and singletons).

    Greedy tallest-first, like `find_isotope_envelopes`. Per seed it prefers the
    lowest charge (z=1), and only accepts a higher charge z when there are at least
    `min_charge_evidence` peaks at *fractional* isotope positions -- offsets a z=1
    envelope cannot produce (e.g. the +0.5, +1.5 Da ladder of a 2+). This keeps a
    single stray half-Da peak from being misread as a 2+ envelope (which otherwise
    surfaces downstream as a spurious [2M+2H]2+ adduct).
    A single fractional peak also suffices when it is the z-charge M+1 and outranks
     the z=1 M+1 slot (see `_fractional_m1_leads`).
    z is tried from `max_charge` down. Keep it <= 3: the z=4 ladder contains
     every z=2 position, so a 2+ envelope would read as 4+.
    A group's charge sets the spacing used both to collect its isotopologues,
     and to locate M+1/M+2 for Cl/Br detection.
    Peaks two below a seed that look like an in-source -H2 loss rather than a Cl/Br
     M0 are left out (see `_h2_loss_members`), so they don't become M0.

    Args:
        spec_arr: peaks (any intensity scale; `noise.min_rel` is base-relative).
        max_charge: highest charge state to try (z in 1..max_charge).
        isotope_tol: m/z tolerance (Da) for isotope spacing matches.
        max_isotopes: max isotopologues to collect per envelope.
        noise: which peaks survive to be grouped (default: keep all).
        detect_halogens: if True, will flag groups whose M+2 is too tall to be
            halogen-free (Cl/Br).
        min_charge_evidence: fractional-position peaks required to accept z>1.

    Returns:
        GroupedSpectrum with per-peak group labels and per-group charge, mono
        index, deconvoluted mass, and halogen flag.
    """
    n_peaks = len(spec_arr)
    group_labels = np.full(n_peaks, -1, dtype=np.intp)

    mask = (
        NoiseThreshold().mask(spec_arr) if noise is None else noise.mask(spec_arr)
    )

    intensity_order = np.argsort(spec_arr['intsy'])[::-1]

    charges: list[int] = []
    gid = 0
    for seed_idx in intensity_order:
        if not mask[seed_idx] or group_labels[seed_idx] != -1:
            continue

        seed_mz = float(spec_arr['mz'][seed_idx])

        def _collect(z: int) -> NDArray[np.intp]:
            spacing = ISOTOPE_SPACING / z
            nb = find_peaks_by_spacing(
                spec_arr, seed_idx, spacing, max_isotopes, isotope_tol
            )
            return nb[mask[nb] & (group_labels[nb] == -1)]

        # Default to singly charged; step up only with a real fractional ladder.
        chosen_z = 1
        members = _collect(1)
        for z in range(max_charge, 1, -1):
            cand = _collect(z)
            if (
                _count_sub_integer(spec_arr, cand, seed_mz, isotope_tol) >= min_charge_evidence
                or _fractional_m1_leads(spec_arr, cand, seed_mz, z, isotope_tol)
            ):
                chosen_z = z
                members = cand
                break

        # Leave the -H2 species unassigned so it forms its own group.
        members = members[~_h2_loss_members(spec_arr, seed_idx, members, chosen_z, isotope_tol)]

        group_labels[seed_idx] = gid
        group_labels[members] = gid
        charges.append(chosen_z)
        gid += 1

    n_groups = gid
    charge_arr = np.asarray(charges, dtype=np.intp)
    mono_idx = np.zeros(n_groups, dtype=np.intp)
    deconv = np.zeros(n_groups, dtype=np.float64)
    is_halogen = np.zeros(n_groups, dtype=bool)
    apex_intensity = np.zeros(n_groups, dtype=np.float64)

    for g in range(n_groups):
        peak_idxs = np.where(group_labels == g)[0]
        # Monoisotopic = lowest-m/z peak of the group.
        mono = peak_idxs[np.argmin(spec_arr['mz'][peak_idxs])]
        mono_idx[g] = mono
        apex_intensity[g] = float(spec_arr['intsy'][peak_idxs].max())
        z = int(charge_arr[g])
        deconv[g] = deconv_mass(float(spec_arr['mz'][mono]), z)
        # >= 2, not >=3 because perbrominated envelope can lose weak M+1
        if detect_halogens and len(peak_idxs) >= 2:
            is_halogen[g] = envelope_is_halogen(
                spec_arr[peak_idxs], charge=z, tol=isotope_tol
            )

    return GroupedSpectrum(
        spec_arr=spec_arr,
        group_labels=group_labels,
        charge=charge_arr,
        mono_idx=mono_idx,
        deconv_mass=deconv,
        is_halogen=is_halogen,
        apex_intensity=apex_intensity,
    )
