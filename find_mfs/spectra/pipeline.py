"""Spectrum annotation pipeline: cleaning → envelopes → adducts → halogen → formula query."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .envelopes import (
    SpectrumArray,
    ISOTOPE_SPACING,
    normalize,
    make_intensity_mask,
    make_shoulder_mask,
    find_isotope_envelopes,
    _envelope_stats,
)
from .adducts import (
    POSITIVE_ADDUCT_SPECS,
    POSITIVE_PAIR_DELTAS,
    POSITIVE_LOSS_LABELS,
    find_adduct_groups,
    identify_adduct,
)
from .halogen import detect_halogen_envelopes
from ..core.finder import FormulaFinder

if TYPE_CHECKING:
    from ..core.results import FormulaSearchResults

_HALOGENS = ['Br', 'Cl']

@dataclass(slots=True)
class AnnotatedSpectrum:
    """
    Flat intermediate representation of a parsed MS1 spectrum.

    All per-envelope arrays have length n_envelopes
        (= envelope_labels.max() + 1).
    envelope_labels is per-peak (same length as spec_arr).

    After query_envelopes(), the results field holds per-envelope
    FormulaSearchResults.
    """
    spec_arr: SpectrumArray
    envelope_labels: NDArray[np.intp]       # shape (n_peaks,), -1 = unassigned
    adduct_labels: NDArray[np.intp]         # shape (n_envelopes,)
    halogen_flags: NDArray[np.bool_]        # shape (n_envelopes,)
    mono_mz: NDArray[np.float64]            # shape (n_envelopes,)
    results: list[FormulaSearchResults] | None = field(
        default=None, repr=False
    )  # populated by query_envelopes

    @property
    def n_envelopes(self) -> int:
        return len(self.mono_mz)

    @property
    def base_envelope_idx(self) -> int | None:
        """
        Index of the envelope with the highest max intensity, or None if empty
        """
        if self.n_envelopes == 0:
            return None
        _, max_intsy = _envelope_stats(
            self.spec_arr, self.envelope_labels, self.n_envelopes
        )
        return int(np.argmax(max_intsy))

    @property
    def base_result(self) -> FormulaSearchResults | None:
        """
        FormulaSearchResults for the base (tallest) envelope, or None.
        """
        if self.results is None:
            return None

        if self.base_envelope_idx is None:
            return None

        return self.results[self.base_envelope_idx].sort_by_error()

    def get_envelope(self, idx) -> SpectrumArray:
        peak_mask = self.envelope_labels == idx
        return self.spec_arr[peak_mask]


# TODO: Way too many arguments imo
def parse_spectrum(
    spec_arr: SpectrumArray,
    *,
    # spectrum cleaning
    min_intensity: float | None = None,
    shoulder_tol: float | None = None,
    # isotope envelope detection
    spacing: float = ISOTOPE_SPACING,
    isotope_tol: float = 0.05,
    max_isotopes: int = 10,
    # adduct grouping
    adduct_pair_deltas: dict[tuple[str, str], float] = POSITIVE_PAIR_DELTAS,
    adduct_tol_da: float = 0.01,
    adduct_tol_ppm: float = 5.0,
    # halogen detection
    detect_halogens: bool = False,
    m2_m0_threshold: float = 0.33,
) -> AnnotatedSpectrum:
    """
    Build an AnnotatedSpectrum:
    clean -> detect envelopes -> group adducts -> detect halogens
    """
    # Build spectrum mask (for cleaning)
    mask = None
    if min_intensity is not None:
        mask = make_intensity_mask(spec_arr, min_intensity)
    if shoulder_tol is not None:
        shoulder = make_shoulder_mask(spec_arr, shoulder_tol)
        mask = shoulder if mask is None else (mask & shoulder)

    # Detect isotope envelopes
    envelope_labels = find_isotope_envelopes(
        spec_arr, spacing, isotope_tol, max_isotopes, mask,
    )

    n_envelopes = int(envelope_labels.max()) + 1 if len(envelope_labels) > 0 else 0

    # Halogen detection (must run before trimming — trimming destroys the zig-zag)
    if detect_halogens and n_envelopes > 0:
        halogen_flags = detect_halogen_envelopes(
            spec_arr, envelope_labels, spacing, isotope_tol, m2_m0_threshold,
        )
    else:
        halogen_flags = np.zeros(n_envelopes, dtype=bool)

    # Trim in-source fragment peaks below the base peak (non-halogen only).
    # Unassigns peaks with m/z below the tallest peak in each envelope.
    # TODO: Assumes monoisotopic = base peak; fails for high-MW (>~1000 Da).
    for eid in range(n_envelopes):
        if halogen_flags[eid]:
            continue
        peak_idxs = np.where(envelope_labels == eid)[0]
        if len(peak_idxs) == 0:
            continue
        mzs = spec_arr['mz'][peak_idxs]
        intsys = spec_arr['intsy'][peak_idxs]
        base_mz = mzs[np.argmax(intsys)]
        trim_mask = mzs < base_mz
        if np.any(trim_mask):
            envelope_labels[peak_idxs[trim_mask]] = -1

    # Per-envelope stats (uses cleaned labels)
    if n_envelopes > 0:
        mono_mz, _ = _envelope_stats(
            spec_arr, envelope_labels, n_envelopes
        )
    else:
        mono_mz = np.empty(0, dtype=np.float64)

    # Adduct grouping (uses cleaned labels)
    if n_envelopes > 0:
        adduct_labels = find_adduct_groups(
            spec_arr, envelope_labels, adduct_pair_deltas, adduct_tol_da, adduct_tol_ppm,
        )
    else:
        adduct_labels = np.empty(0, dtype=np.intp)

    return AnnotatedSpectrum(
        spec_arr=spec_arr,
        envelope_labels=envelope_labels,
        adduct_labels=adduct_labels,
        halogen_flags=halogen_flags,
        mono_mz=mono_mz,
    )

def _resolve_adduct_for_envelope(
    annotated: AnnotatedSpectrum,
    eid: int,
    adduct_pair_deltas: dict[tuple[str, str], float],
    tol_da: float,
    tol_ppm: float,
    loss_labels: frozenset[str] = frozenset(),
    adduct_specs: list[tuple[str, str, int]] | None = None,
) -> str | None:
    """
    Identify the adduct string for a given envelope from its
    adduct-group partners.

    Loss products get a net adduct: e.g. [M+H-H2O]+ → adduct="-OH".
    """
    group = annotated.adduct_labels[eid]
    group_eids = np.where(annotated.adduct_labels == group)[0]
    partner_eids = group_eids[group_eids != eid]

    if len(partner_eids) == 0:
        return None

    return identify_adduct(
        annotated.mono_mz[eid],
        annotated.mono_mz[partner_eids],
        adduct_pair_deltas,
        tol_da,
        tol_ppm,
        loss_labels=loss_labels,
        adduct_specs=adduct_specs,
    )

def _append_halogen_counts(
    finder_elements: Iterable,
    counts: dict | str | None,
) -> dict | str:
    """
    Append Br*/Cl* to a min_counts or max_counts argument
    """
    if counts is None:
        # No element constraints defined; use full wildcard set
        return "*".join([x for x in finder_elements] + _HALOGENS) + '*'

    if isinstance(counts, str):
        return counts + "*".join(_HALOGENS) + "*"

    # dict — add wildcard entries for Br and Cl if not already present
    counts = dict(counts)
    counts.setdefault('Br', None)
    counts.setdefault('Cl', None)

    return counts


def query_envelopes(
    annotated: AnnotatedSpectrum,
    finder: FormulaFinder,
    *,
    detect_halogens: bool = False,
    adduct_specs: list[tuple[str, str, int]] = POSITIVE_ADDUCT_SPECS,
    adduct_pair_deltas: dict[tuple[str, str], float] = POSITIVE_PAIR_DELTAS,
    loss_labels: frozenset[str] = POSITIVE_LOSS_LABELS,
    adduct_tol_da: float = 0.01,
    adduct_tol_ppm: float = 5.0,
    isotope_mz_tol_ppm: float = 5.0,
    **finder_kwargs: Any,
) -> AnnotatedSpectrum:
    """
    Query formulae for every envelope in an AnnotatedSpectrum.

    Populates annotated.results (list of length n_envelopes, each a
    FormulaSearchResults or None) and returns the same AnnotatedSpectrum.

    When detect_halogens=True, envelopes flagged as halogenated are queried
    with an auto-created halogen finder (finder's elements + Br + Cl) and
    min_counts='Br*Cl*'.
    """
    n_envelopes = annotated.n_envelopes

    # Only build halogen finder if there are actually halogenated envelopes
    halogen_finder: FormulaFinder | None = None
    if detect_halogens and np.any(annotated.halogen_flags):
        user_elements = finder.element_set.copy()
        user_elements.update(
            {'Br', 'Cl'}
        )
        halogen_finder = FormulaFinder(user_elements)

    # Iterate over envelopes and accumulate mf results
    results: list[FormulaSearchResults | None] = []
    for eid in range(n_envelopes):
        peaks: SpectrumArray = annotated.get_envelope(eid)

        if len(peaks) == 0:
            results.append(None)
            continue

        # Pick finder and extra kwargs based on halogen flag
        if annotated.halogen_flags[eid] and halogen_finder is not None:
            active_finder = halogen_finder
            kw = dict(finder_kwargs)
            kw['min_counts'] = _append_halogen_counts(
                finder_elements=finder.element_set,
                counts=kw.get('min_counts'),
            )
            kw['max_counts'] = _append_halogen_counts(
                finder_elements=finder.element_set,
                counts=kw.get('max_counts')
            )
        else:
            active_finder = finder
            kw = dict(finder_kwargs)

        # Resolve adduct
        adduct = _resolve_adduct_for_envelope(
            annotated, eid, adduct_pair_deltas, adduct_tol_da, adduct_tol_ppm,
            loss_labels=loss_labels,
            adduct_specs=adduct_specs,
        )

        result = active_finder.find_formulae(
            mass=annotated.mono_mz[eid],
            charge=1,  # TODO: Don't hardcode
            adduct=adduct or "H",  # Set adduct to H if non foune # TODO
            **kw,
        )
        results.append(result)

    annotated.results = results
    return annotated


def query_spectrum(
    spec_arr: SpectrumArray,
    finder: FormulaFinder,
    *,
    detect_halogens: bool = False,
    # cleaning
    min_intensity: float | None = None,
    shoulder_tol: float | None = None,
    # envelope detection
    spacing: float = ISOTOPE_SPACING,
    isotope_tol: float = 0.05,
    max_isotopes: int = 8,
    # adduct grouping
    adduct_specs: list[tuple[str, str, int]] = POSITIVE_ADDUCT_SPECS,
    adduct_pair_deltas: dict[tuple[str, str], float] = POSITIVE_PAIR_DELTAS,
    loss_labels: frozenset[str] = POSITIVE_LOSS_LABELS,
    adduct_tol_da: float = 0.01,
    adduct_tol_ppm: float = 5.0,
    # isotope match
    isotope_mz_tol_ppm: float = 5.0,
    # halogen
    m2_m0_threshold: float = 0.33,
    # finder
    **finder_kwargs: Any,
) -> AnnotatedSpectrum:
    """
    Full pipeline: clean → annotate → query formulae for all envelopes.

    Convenience wrapper around parse_spectrum() + query_envelopes().
    Returns the AnnotatedSpectrum with results populated.

    Note:
        The spectrum is base-peak normalized (to 1.0) before cleaning, so
        `min_intensity` is a **fraction of the base peak** -- e.g. 0.01 keeps
        peaks at or above 1% of the tallest. This scale changed from 0-100 to
        0-1 when the library standardised on the base-peak-1.0 convention;
        divide any previously-tuned `min_intensity` by 100.
    """
    annotated = parse_spectrum(
        normalize(spec_arr),
        min_intensity=min_intensity,
        shoulder_tol=shoulder_tol,
        spacing=spacing,
        isotope_tol=isotope_tol,
        max_isotopes=max_isotopes,
        adduct_pair_deltas=adduct_pair_deltas,
        adduct_tol_da=adduct_tol_da,
        adduct_tol_ppm=adduct_tol_ppm,
        detect_halogens=detect_halogens,
        m2_m0_threshold=m2_m0_threshold,
    )

    return query_envelopes(
        annotated,
        finder,
        detect_halogens=detect_halogens,
        adduct_specs=adduct_specs,
        adduct_pair_deltas=adduct_pair_deltas,
        loss_labels=loss_labels,
        adduct_tol_da=adduct_tol_da,
        adduct_tol_ppm=adduct_tol_ppm,
        isotope_mz_tol_ppm=isotope_mz_tol_ppm,
        **finder_kwargs,
    )
