"""
Entry for MS2 spectrum + optional MS1 => ranked formulae.

If MS1 is given, extracts isotope envelope and tries to find the adduct.
Otherwise, just uses the given precursor m/z and checks all adducts:

    decompose (per adduct) => concat
        => score (chem + mass + iso + MS2) => rank

# TODO
Scope: this is **per-precursor**. Driving it across a whole scan -- detecting
envelopes, picking precursors, grouping adducts -- is the caller's job for now.
(`find_mfs.spectra.query_spectrum` does some of that today but is slated for
deprecation, so don't build on it.)
"""

from __future__ import annotations

import numpy as np

from .core.finder import get_finder
from .core.results import FormulaSearchResults
from .ms2.tables import ION_TO_ADDUCT, normalize_adducts

# Adducts considered when the caller does not say. These are the positive-mode
# ions the MS2 reranker knows about, so the MS2 term applies to all of them.
DEFAULT_ADDUCTS: tuple[tuple[str | None, int], ...] = tuple(ION_TO_ADDUCT.values())


def annotate_precursor(
    precursor_mz: float,
    *,
    adducts=DEFAULT_ADDUCTS,
    elements: str = 'CHNOPS',
    error_ppm: float = 5.0,
    scorer=None,
    ms1_peaks: np.ndarray | None = None,
    ms2_peaks: np.ndarray | None = None,
    instrument: str = 'unknown',
    ms2_weight: float = 1.0,
    ms2_temperature: float = 1.0,
    sort: bool = True,
    finder_kwargs: dict | None = None,
    **score_kwargs,
) -> FormulaSearchResults:
    """
    Rank molecular formulae for one precursor.

    Args:
        precursor_mz: Observed precursor m/z.
        adducts: What to consider. A single adduct (`"H"`), a single ion string
            (`"[M+H]+"`), an explicit `(adduct, charge)` pair, or a list of any
            of those. Defaults to every ion the MS2 reranker supports.
        elements: Element set to decompose over ('CHNOPS', 'CHNOPSFClBrI', ...).
        error_ppm: Precursor mass tolerance for decomposition.
        scorer: A `FormulaScorer`. Defaults to `FormulaScorer.default()`. Attach
            an MS2 reranker with `.with_ms2(...)` to enable the MS2 term.
        ms1_peaks: Optional `(n, 2)` MS1 peak list for the isotope term.
        ms2_peaks: Optional `(n, 2)` MS2 peak list for the MS2 term. Requires a
            scorer with a reranker attached.
        instrument: Instrument name, for the reranker's one-hot.
        ms2_weight: Weight on the MS2 term when ranking. 0 disables it.
        ms2_temperature: Softmax temperature for the MS2 term. Note only
            `ms2_weight / ms2_temperature` affects ordering -- see
            `FormulaSearchResults.log_posterior`.
        sort: Return candidates ranked best-first. Set False to keep
            decomposition order (scores are attached either way).
        finder_kwargs: Extra arguments for `FormulaFinder.find_formulae`, e.g.
            `{"min_counts": {"C": 1}}` to require carbon, or `filter_rdbe`.
        **score_kwargs: Passed through to `FormulaScorer.score` (e.g.
            `ms2_top_n`, `mass_sigma_ppm`, `iso_weight`, `chem_weight`).

    Returns:
        FormulaSearchResults over every (formula, adduct) considered, scored
        and -- unless `sort=False` -- ranked by the full log-posterior.

    Example:
        >>> from find_mfs import FormulaScorer, annotate_precursor
        >>> scorer = FormulaScorer.default().with_ms2("mistnet.npz")
        >>> hits = annotate_precursor(
        ...     515.3228, ms2_peaks=peaks, scorer=scorer, error_ppm=5.0
        ... )
        >>> hits[0].formula.formula, hits[0].adduct
    """
    if scorer is None:
        from .scoring import FormulaScorer
        scorer = FormulaScorer.default()

    finder = get_finder(elements)
    searches = [
        finder.find_formulae(
            mass=precursor_mz, charge=charge, adduct=adduct, error_ppm=error_ppm,
            **(finder_kwargs or {}),
        )
        for adduct, charge in normalize_adducts(adducts)
    ]
    results = FormulaSearchResults.concat(searches)

    scorer.score(
        results,
        ms1_peaks=ms1_peaks,
        ms2_peaks=ms2_peaks,
        precursor_mz=precursor_mz,
        instrument=instrument,
        **score_kwargs,
    )

    if not sort:
        return results
    return results.sort_by_posterior(
        ms2_weight=ms2_weight, ms2_temperature=ms2_temperature
    )
