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
from .halogen import envelope_is_halogen
from .ions import deconv_mass


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
    max_charge: int = 2,
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
    A group's charge sets the spacing used both to collect its isotopologues,
     and to locate M+1/M+2 for Cl/Br detection.

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
            if _count_sub_integer(spec_arr, cand, seed_mz, isotope_tol) >= min_charge_evidence:
                chosen_z = z
                members = cand
                break

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
