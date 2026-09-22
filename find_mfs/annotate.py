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

# Adducts considered when the caller does not say. These are the positive-mode
# ions the MS2 reranker knows about, so the MS2 term applies to all of them.
DEFAULT_ADDUCTS: tuple[tuple[str | None, int], ...] = tuple(ION_TO_ADDUCT.values())

# Default MS1 noise floor for annotate_analyte_dia: drop peaks below 5% of the base
# peak before grouping. Near-noise peaks otherwise form spurious 0.5-Da ladders (a
# fake 2+) and coincidental low-abundance adduct networks. Pass an explicit
# NoiseThreshold (or None) to override.
_DEFAULT_NOISE = NoiseThreshold(min_rel=0.05)


def annotate_precursor(
    precursor_mz: float,
    *,
    adducts=DEFAULT_ADDUCTS,
    elements: str = 'CHNOPS',
    error_ppm: float = 5.0,
    scorer=None,
    autodetect_cl_br: bool = False,
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
        elements: Element set to decompose over ('CHNOPS', 'CHNOPSFClBrI', ...).
        autodetect_cl_br: If true, and ms1_peaks is given, tries to determine
            whether chlorine/bromine is present using the precursor isotope
             envelope - if present, appends Br and Cl to whatever is set
             in `elements`
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
            `{"min_counts": {"C": 1}}` to require carbon, or `filter_rdbe`.
        **score_kwargs: Passed through to `FormulaScorer.score` (e.g.
            `ms2_top_n`, `mass_sigma_ppm`, `iso_weight`, `chem_weight`).

    Returns:
        FormulaSearchResults over every (formula, adduct) considered, scored
        and -- unless `sort=False` -- ranked by the full log-posterior.

    Example:
        >>> from find_mfs import FormulaScorer, annotate_precursor
        >>> scorer = FormulaScorer().with_ms2("mistnet.npz")
        >>> hits = annotate_precursor(
        ...     515.3228, ms2_peaks=peaks, scorer=scorer, error_ppm=5.0
        ... )
        >>> hits[0].formula.formula, hits[0].adduct
    """
    if scorer is None:
        scorer = FormulaScorer()

    # Set element set depending on whether precursor envelope has
    # telltale signs of containing Br/Cl
    if autodetect_cl_br and ms1_peaks is not None:
        if envelope_is_halogen(
                envelope=ms1_peaks,
        ):
            for x in ('Cl', 'Br'):
                if x not in elements:
                    elements += x

    # Generate mf queries for each of the adducts requested
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
        is_halogen: Whether the base envelope shows the Cl/Br M+2 zig-zag.
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
    elements: str = 'CHNOPS',
    error_ppm: float = 5.0,
    max_charge: int = 2,
    isotope_tol: float = 0.02,
    noise: NoiseThreshold | None = _DEFAULT_NOISE,
    detect_halogens: bool = True,
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
        elements: Element set; widened with Cl/Br if the base envelope is halogenated.
        error_ppm: Precursor tolerance (also the MDN mass tolerance).
        max_charge: Highest charge state to infer during grouping.
        noise: Peak survival threshold at grouping time. Defaults to a 5%-of-base
            floor (drops near-noise peaks that otherwise form fake 2+ ladders and
            trace adduct networks). Pass `NoiseThreshold()` or `None` to keep all.
        detect_halogens: Flag halogenated envelopes and widen the element set.
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
        detect_halogens=detect_halogens,
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

    elems = elements
    if is_halogen:
        for x in ('Cl', 'Br'):
            if x not in elems:
                elems += x

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

    candidates = annotate_precursor(
        query_mz,
        adducts=adducts,
        elements=elems,
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
