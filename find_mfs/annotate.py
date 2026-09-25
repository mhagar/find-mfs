"""
Entries for ranking molecular formulae from mass spectra.

`annotate_precursor` is the per-precursor primitive: given a precursor m/z (and
optionally MS1/MS2 peaks) it decomposes over every requested adduct and scores:

    decompose (per adduct) => concat
        => score (chem + mass + iso + MS2) => rank

`annotate_analyte_dia` is the analyte-centric orchestrator for DIA: from an MS1
scan + precursor + one MS2 it detects the isotope envelopes, runs the mass-
difference network to resolve the base envelope's adduct/charge and the analyte's
neutral mass, flags halogenation, and then calls `annotate_precursor` on the base
envelope with the resolved adduct narrowed down. It returns the grouped spectrum
and adduct labels alongside the candidates, so callers can visualise the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.finder import get_finder
from .core.results import FormulaSearchResults
from .ms2.tables import ION_TO_ADDUCT, normalize_adducts
from .scoring import FormulaScorer
from .spectra.halogen import envelope_is_halogen
from .spectra.envelopes import SpectrumArray, normalize
from .spectra.grouping import GroupedSpectrum, NoiseThreshold, group_signals
from .spectra.ions import IonType, ION_VOCAB, _H
from .spectra.network import solve_for_base
from .utils.formulae import parse_counts

# Adducts considered when the caller does not say. These are the positive-mode
# ions the MS2 reranker knows about, so the MS2 term applies to all of them.
DEFAULT_ADDUCTS: tuple[tuple[str | None, int], ...] = tuple(ION_TO_ADDUCT.values())

# Default MS1 noise floor for annotate_analyte_dia: drop peaks below 5% of the base
# peak before grouping. Near-noise peaks otherwise form spurious 0.5-Da ladders (a
# fake 2+) and coincidental low-abundance adduct networks. Pass an explicit
# NoiseThreshold (or None) to override.
_DEFAULT_NOISE = NoiseThreshold(min_rel=0.05)

# Default search space when unconstrained by caller
DEFAULT_MAX_COUNTS = 'C*H*N*O*P*S*'

# The elements whose isotope pattern `envelope_is_halogen` can see.
# Only these may appear in `halogen_cap`
_DETECTABLE_HALOGENS = ('Cl', 'Br')


def _as_counts(
        counts: str | dict
) -> dict[str, float]:
    """
    A constraint string or dict -> {symbol: count}, in given order.
    """
    if isinstance(counts, str):
        return parse_counts(counts)
    return dict(counts)


def resolve_search_bounds(
    max_counts: str | dict,
    min_counts: str | dict | None,
    halogen_cap: str | dict | None,
    halogenated: bool,
) -> tuple[str, dict[str, float], dict[str, float]]:
    """
    Converts max_counts/min_counts/halogen_cap/halogenated into
        (elements, max bounds, min bounds) for `finder`

    The element set is derived from `max_counts` (anything with count > 0).
    An element capped at 0 (i.e. "P0") is left out.

    When `halogenated`, the `halogen_cap` entries override matching `max_counts` entries,

    Raises:
        ValueError: If `halogen_cap` names anything but Cl/Br, if nothing is
            allowed, or if `min_counts` requires an element `max_counts` forbids.
    """
    max_d = _as_counts(max_counts)

    if halogen_cap is not None:
        cap = _as_counts(halogen_cap)
        bad = [x for x in cap if x not in _DETECTABLE_HALOGENS]
        if bad or not cap:
            raise ValueError(
                f"halogen_cap may only bound {'/'.join(_DETECTABLE_HALOGENS)} "
                f"(got {halogen_cap!r})"
            )
        if halogenated:
            max_d.update(cap)

    elements = [x for x, n in max_d.items() if n > 0]
    if not elements:
        raise ValueError(f"max_counts allows no elements (got {max_counts!r})")

    min_d = _as_counts(min_counts) if min_counts else {}
    forbidden = [x for x, n in min_d.items() if n > 0 and x not in elements]
    if forbidden:
        raise ValueError(
            f"min_counts requires {forbidden}, which max_counts does not allow"
        )

    return (
        ''.join(elements),
        {x: max_d[x] for x in elements},
        {x: min_d.get(x, 0) for x in elements},
    )


def annotate_precursor(
    precursor_mz: float,
    *,
    adducts=DEFAULT_ADDUCTS,
    max_counts: str | dict = DEFAULT_MAX_COUNTS,
    min_counts: str | dict | None = None,
    halogen_cap: str | dict | None = None,
    error_ppm: float = 5.0,
    scorer=None,
    ms1_peaks: SpectrumArray | None = None,
    ms2_peaks: SpectrumArray | None = None,
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
        max_counts: Upper element bounds, as a string ("C*H*N*O*P0S2") or dict.
            This also defines the element set (anything with count > 0)
            Unbounded CHNOPS by default.
        min_counts: Optional lower element bounds, same format.
            May only name elements in `max_counts`.
        halogen_cap: Enables Cl/Br detection and sets its cap, e.g. "Cl4Br3".
            If `ms1_peaks` triggers the halogen detector,
            `max_counts` inherits the bounds given by `halogen_cap`,
            widening element set. May only name Cl and Br.
            By default, is None (disables halogen detection).
        error_ppm: Precursor mass tolerance for decomposition.
        scorer: A `FormulaScorer`. Defaults to `FormulaScorer.default()`. Attach
            an MS2 reranker with `.with_ms2(...)` to enable the MS2 term.
        ms1_peaks: Optional SpectrumArray MS1 peak list for the isotope term.
        ms2_peaks: Optional SpectrumArray MS2 peak list for the MS2 term. Requires a
            scorer with MistNet attached.
        instrument: Instrument name, for the reranker's one-hot.
        ms2_weight: Weight on the MS2 term when ranking. 0 disables it.
        ms2_temperature: Softmax temperature for the MS2 term. Note only
            `ms2_weight / ms2_temperature` affects ordering -- see
            `FormulaSearchResults.log_posterior`.
        sort: Return candidates ranked best-first. Set False to keep
            decomposition order (scores are attached either way).
        finder_kwargs: Extra arguments for `FormulaFinder.find_formulae`, e.g.
            `filter_rdbe` or `check_octet`.
        **score_kwargs: Passed through to `FormulaScorer.score` (e.g.
            `ms2_top_n`, `mass_sigma_ppm`, `iso_weight`, `chem_weight`).

    Returns:
        FormulaSearchResults over every (formula, adduct) considered, scored
        and ranked by the full log-posterior (unless `sort` is False`).
        The query parameters are stored in `query_params`

    Example:
        >>> from find_mfs import FormulaScorer, annotate_precursor
        >>> scorer = FormulaScorer().with_ms2("mistnet.npz")
        >>> hits = annotate_precursor(
        ...     515.3228, ms2_peaks=peaks, scorer=scorer, error_ppm=5.0,
        ...     max_counts="C*H*N*O*P0S2", halogen_cap="Cl4Br3",
        ... )
        >>> hits[0].formula.formula, hits[0].adduct
    """
    # Is the precursor envelope's M+2 too tall to be halogen-free?
    halogen_detected = None
    if halogen_cap is not None and ms1_peaks is not None:
        halogen_detected = envelope_is_halogen(ms1_peaks, ppm=error_ppm)

    return _rank_formulae(
        precursor_mz,
        adducts=adducts,
        max_counts=max_counts,
        min_counts=min_counts,
        halogen_cap=halogen_cap,
        halogen_detected=halogen_detected,
        error_ppm=error_ppm,
        scorer=scorer,
        ms1_peaks=ms1_peaks,
        ms2_peaks=ms2_peaks,
        instrument=instrument,
        ms2_weight=ms2_weight,
        ms2_temperature=ms2_temperature,
        sort=sort,
        finder_kwargs=finder_kwargs,
        **score_kwargs,
    )


def _rank_formulae(
    precursor_mz: float,
    *,
    adducts,
    max_counts: str | dict,
    min_counts: str | dict | None,
    halogen_cap: str | dict | None,
    halogen_detected: bool | None,
    error_ppm: float,
    scorer,
    ms1_peaks: SpectrumArray | None,
    ms2_peaks: SpectrumArray | None,
    instrument: str,
    ms2_weight: float,
    ms2_temperature: float,
    sort: bool,
    finder_kwargs: dict | None,
    **score_kwargs,
) -> FormulaSearchResults:
    """
    `annotate_precursor` minus halogen detection:
    the caller has already decided `halogen_detected`
    (`annotate_analyte_dia` reads it off its grouped envelope, rather than
     re-running detection on the base peaks).
    """
    if scorer is None:
        scorer = FormulaScorer()

    finder_kwargs = dict(finder_kwargs or {})
    clashing = {'min_counts', 'max_counts'} & finder_kwargs.keys()
    if clashing:
        raise TypeError(
            f"pass {sorted(clashing)} directly, not via finder_kwargs"
        )

    elements, max_bounds, min_bounds = resolve_search_bounds(
        max_counts, min_counts, halogen_cap, bool(halogen_detected),
    )

    # Generate mf queries for each of the adducts requested
    finder = get_finder(list(max_bounds))
    searches = [
        finder.find_formulae(
            mass=precursor_mz, charge=charge, adduct=adduct, error_ppm=error_ppm,
            max_counts=max_bounds, min_counts=min_bounds,
            **finder_kwargs,
        )
        for adduct, charge in normalize_adducts(adducts)
    ]
    results = FormulaSearchResults.concat(searches)
    results.query_params['elements'] = elements
    results.query_params['halogen_detected'] = halogen_detected

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


@dataclass(slots=True)
class AnalyteAnnotation:
    """
    Result of `annotate_analyte_dia` for one analyte.

    Attributes:
        grouped: The MS1 scan partitioned into signal groups (for viz/tooling);
            its `adduct_label` / `M` / `component_id` are filled for the resolved
            component.
        base_group_id: Index of the base (precursor) group in `grouped`.
        precursor_mz: The precursor m/z actually used -- the passed `precursor_mz`,
            or the auto-selected base envelope's monoisotopic m/z when none was given.
        adduct: Resolved adduct string for the base group (FormulaFinder form),
            or None for a radical/no-adduct ion.
        charge: Resolved charge of the base group.
        is_halogen: Whether the base envelope has the excess M+2 of Cl/Br.
        M: Resolved analyte neutral mass.
        candidates: Ranked `FormulaSearchResults` for the analyte.
        near_tie_adducts: Runner-up base-adduct interpretations the MDN could not
            confidently rule out (winner first). By default these are NOT decomposed
            into `candidates` -- see `include_near_ties`.
    """
    grouped: GroupedSpectrum
    base_group_id: int
    precursor_mz: float
    adduct: str | None
    charge: int
    is_halogen: bool
    M: float
    candidates: FormulaSearchResults
    near_tie_adducts: list[tuple[str | None, int]] = None  # MDN near-ties (metadata)


def annotate_analyte_dia(
    ms1_peaks: SpectrumArray,
    precursor_mz: float | None = None,
    ms2_peaks: SpectrumArray | None = None,
    *,
    scorer=None,
    max_counts: str | dict = DEFAULT_MAX_COUNTS,
    min_counts: str | dict | None = None,
    halogen_cap: str | dict | None = None,
    error_ppm: float = 5.0,
    max_charge: int = 2,
    isotope_tol: float = 0.02,
    noise: NoiseThreshold | None = _DEFAULT_NOISE,
    precursor_tol: float = 0.02,
    vocab: list[IonType] = ION_VOCAB,
    intensity_weight: float = 1.0,
    include_near_ties: bool = False,
    instrument: str = 'unknown',
    ms2_weight: float = 1.0,
    ms2_temperature: float = 1.0,
    sort: bool = True,
    finder_kwargs: dict | None = None,
    **score_kwargs,
) -> AnalyteAnnotation:
    """
    Rank molecular formulae for one analyte from a DIA MS1 scan + precursor + MS2.

    Steps: group the MS1 scan into charge-resolved isotope envelopes -> select the
    base envelope -> resolve its adduct/charge and the analyte mass M via the
    mass-difference network -> flag halogenation -> `annotate_precursor` on the base
    envelope, with the adduct narrowed to the MDN's answer (plus any near-ties, for
    the MS2 term to break).

    In DIA there is no per-precursor selection: the MS2 spectrum is dominated by
    fragments of the most intense MS1 species, so by default the base envelope is
    the tallest one in the scan (this approximates DDA, which is what MistNet was
    trained on). Pass `precursor_mz` only to override that -- e.g. a DDA scan with a
    known isolation target -- and the base envelope becomes the one matching it.

    Args:
        ms1_peaks: MS1 peak list (base-peak normalized internally).
        precursor_mz: m/z of the fragmented precursor. If None (default), the base
            envelope is the tallest signal in the scan and the precursor is taken as
            that envelope's monoisotopic m/z.
        ms2_peaks: Optional MS2 peaks for the reranker (needs a scorer with MistNet).
        max_counts / min_counts: Element bounds; `max_counts` defines the element
            set. See `annotate_precursor`.
        halogen_cap: Enables Cl/Br widening and sets its cap (see
            `annotate_precursor`). Applied when the *base envelope* is flagged
            halogenated. Detection itself always runs, so `is_halogen` is
            reported either way.
        error_ppm: Precursor tolerance (also the MDN mass tolerance).
        max_charge: Highest charge state to infer during grouping.
        noise: Peak survival threshold at grouping time. Defaults to a 5%-of-base
            floor (drops near-noise peaks that otherwise form fake 2+ ladders and
            trace adduct networks). Pass `NoiseThreshold()` or `None` to keep all.
        precursor_tol: m/z tolerance for matching `precursor_mz` to an envelope.
        vocab: Ion-type vocabulary for the MDN.
        intensity_weight: Exponent on each group's relative intensity when scoring
            adduct networks (1.0 = linear, biases toward high-abundance networks;
            0.0 = intensity-blind).
        include_near_ties: If True, also decompose the base under the MDN's runner-up
            adduct interpretations. These imply a *different* analyte mass, so without
            a decisive MS2 term a loss/fragment formula can outrank the true one --
            hence the default is False (decompose only the winning adduct, matching a
            manual single-adduct annotation). Near-ties are always reported in
            `AnalyteAnnotation.near_tie_adducts` regardless.
        finder_kwargs / **score_kwargs: forwarded to `annotate_precursor`.

    Returns:
        AnalyteAnnotation with the grouped spectrum, resolved base-envelope
        adduct/charge/halogen/M, and ranked candidates.

    Note:
        `find_formulae` does not convert m/z to neutral mass by charge, so a base
        envelope resolved as multiply-charged or a multimer is decomposed from the
        resolved M reconstructed as a z=1 [M+H]+-equivalent; the isotope term is
        then approximate for that (uncommon) case.
    """
    ms1_peaks = normalize(ms1_peaks)
    grouped = group_signals(
        ms1_peaks,
        max_charge=max_charge,
        isotope_tol=isotope_tol,
        noise=noise,
        # Always flag: cheap, and callers want is_halogen even when not widening.
        detect_halogens=True,
    )

    if precursor_mz is None:
        # DIA: base envelope = the tallest (assigned) signal in the scan.
        assigned = np.where(grouped.group_labels >= 0)[0]
        if len(assigned) == 0:
            raise ValueError(
                "No signal groups in MS1 (empty spectrum or everything below the "
                "noise threshold)."
            )
        tallest = assigned[np.argmax(grouped.spec_arr['intsy'][assigned])]
        base = int(grouped.group_labels[tallest])
        used_precursor_mz = grouped.mono_mz(base)
    else:
        base = grouped.group_containing(precursor_mz, tol=precursor_tol)
        if base is None:
            raise ValueError(
                f"No isotope envelope found within {precursor_tol} of precursor m/z "
                f"{precursor_mz}. Check precursor_tol or the noise threshold."
            )
        used_precursor_mz = precursor_mz

    sol = solve_for_base(
        grouped, base, vocab, tol_ppm=error_ppm, intensity_weight=intensity_weight
    )
    is_halogen = bool(grouped.is_halogen[base])

    base_peaks = grouped.peaks_of(base)

    # By default decompose only the MDN's winning adduct (a single analyte mass),
    # matching a manual single-adduct annotation. Near-tie adducts imply a different
    # M, so merging them lets a loss/fragment formula compete and -- without a
    # decisive MS2 term -- outrank the true formula (see include_near_ties).
    chosen_ions = sol.near_tie_ions if include_near_ties else [sol.base_ion]

    # Adducts the finder can take against the raw precursor m/z are the z=1
    # monomer ion types (find_formulae assumes z=1 m/z arithmetic).
    usable = [
        (ion.adduct_str, ion.charge)
        for ion in chosen_ions
        if ion.n == 1 and ion.charge == 1
    ]

    if usable and sol.base_ion.n == 1 and sol.base_ion.charge == 1:
        query_mz = used_precursor_mz
        adducts = usable
        ms1_for_iso = base_peaks
    else:
        # Multiply-charged / multimer base: decompose the resolved M as a z=1
        # [M+H]+-equivalent (see Note in the docstring).
        query_mz = sol.M + _H
        adducts = [("H", 1)]
        ms1_for_iso = base_peaks

    candidates = _rank_formulae(
        query_mz,
        adducts=adducts,
        max_counts=max_counts,
        min_counts=min_counts,
        halogen_cap=halogen_cap,
        halogen_detected=is_halogen if halogen_cap is not None else None,
        error_ppm=error_ppm,
        scorer=scorer,
        ms1_peaks=ms1_for_iso,
        ms2_peaks=ms2_peaks,
        instrument=instrument,
        ms2_weight=ms2_weight,
        ms2_temperature=ms2_temperature,
        sort=sort,
        finder_kwargs=finder_kwargs,
        **score_kwargs,
    )

    return AnalyteAnnotation(
        grouped=grouped,
        base_group_id=base,
        precursor_mz=used_precursor_mz,
        adduct=sol.base_ion.adduct_str,
        charge=sol.base_ion.charge,
        is_halogen=is_halogen,
        M=sol.M,
        candidates=candidates,
        near_tie_adducts=sol.candidate_adducts(),
    )
