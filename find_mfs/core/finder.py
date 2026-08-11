"""
Main API entry point for find_mfs

This module contains FormulaFinder, which orchestrates
- mass decomposition
- formula validation
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union, Iterable, TYPE_CHECKING

import numpy as np
from molmass import Formula

from molmass.elements import ELECTRON

from .decomposer import MassDecomposer
from .light_formula import LightFormula
from .validator import FormulaValidator
from ..utils.filtering import BOND_ELECTRONS
from ..utils.formulae import to_bounds_dict
from ..isotopes.ratios import get_m1_ratio, get_m2_direct

if TYPE_CHECKING:
    from .results import FormulaSearchResults


@dataclass(slots=True)
class FormulaCandidate:
    """
    Structured result from formula finding.

    Comparisons with other FormulaCandidates uses absolute error_da
    (i.e. for expressions such as `form_cand_a > form_cand_b`

    The score fields are None by default, to be all populated by
    `FormulaScorer.score()`.
    They are additive log-terms of a stacked posterior:

        log_posterior = chem_logprior + iso_loglik + mass_loglik

    Attributes:
        formula: The core molecular formula (without adduct) as a
            molmass.Formula or LightFormula instance
        error_ppm: Mass error in parts per million
        error_da: Mass error in Daltons
        rdbe: Ring and Double Bond Equivalents of the core molecule
            (may be None for some elements)
        adduct: Adduct string as specified by the user (e.g. "Na", "-H"),
            or None if no adduct was specified
        ion_formula: The charged ion formula (core + signed adduct offsets),
            used for isotope-envelope simulation. Equals ``formula`` when no
            adduct was specified. None until materialized.
        chem_logprior: GMM chemical-plausibility log-prior, log P(formula).
        iso_loglik: isotope-pattern log-likelihood, log P(envelope|formula).
            None when no observed envelope was scored.
        mass_loglik: Gaussian precursor-mass log-likelihood, log P(precursor|formula).
        ms2_logit: Raw MistNet reranker output for this candidate. Per-candidate
            and meaningful on its own, so it is safe to cache and to carry through
            sorting and filtering. None when no MS2 spectrum was scored.

            The *normalized* MS2 term, log P(formula|MS2), is deliberately not
            stored here. It is a softmax over the candidate set, so its value
            depends on which other candidates are present. Calculate using
            `FormulaSearchResults.ms2_loglik()` instead.
        log_posterior: Sum of the *per-candidate* terms only
            (chem_logprior + iso_loglik + mass_loglik). The set-dependent MS2
            term is excluded. For the full score including MS2, use
            `FormulaSearchResults.log_posterior()`
    """
    formula: Union[Formula, LightFormula]
    error_ppm: float
    error_da: float
    rdbe: Optional[float]
    adduct: Optional[str] = None
    ion_formula: Optional[Union[Formula, LightFormula]] = None
    chem_logprior: Optional[float] = None
    iso_loglik: Optional[float] = None
    mass_loglik: Optional[float] = None
    ms2_logit: Optional[float] = None
    log_posterior: Optional[float] = None

    def __lt__(self, other: 'FormulaCandidate'):
        return abs(self.error_da) < abs(other.error_da)

    def __le__(self, other: 'FormulaCandidate'):
        return abs(self.error_da) <= abs(other.error_da)

    def __gt__(self, other: 'FormulaCandidate'):
        return abs(self.error_da) > abs(other.error_da)

    def __ge__(self, other: 'FormulaCandidate'):
        return abs(self.error_da) >= abs(other.error_da)




class FormulaFinder:
    """
    API for finding molecular formulae from masses

    This class should be initialized once with a set of elements, and
    then be used to find formulae for multiple query masses.

    Example:
        >>> # Create finder for CHNOPS elements
        >>> finder = FormulaFinder('CHNOPS')
        >>>
        >>> # Find formulae for a mass
        >>> results = finder.find_formulae(
        >>>     mass=180.063,
        >>>     ppm_error=5.0,
        >>>     filter_rdbe=(0, 20),
        >>>     check_octet=True
        >>> )
    """

    def __init__(
        self,
        elements: Iterable[str] = 'CHNOPS',
        use_precalculated: bool = True,
    ):
        """
        Initialize FormulaFinder with a set of elements.

        Args:
            elements: Elements to consider for mass decomposition.
                Can be a string like 'CHNOPS' or list like ['C', 'H', 'N'].
                Default is 'CHNOPS'.

            use_precalculated: Whether to use pre-calculated Extended Residue
                Tables for faster initialization when available.
                Note: currently, they are only available for CHNOPS and
                CHNOPS + Halogens
                Default is True.
        """
        self.decomposer = MassDecomposer(
            elements=elements,
            use_precalculated=use_precalculated,
        )
        self.validator = FormulaValidator()

        # Cache per-element-set constants reused on every query.
        self._symbols: list[str] = list(self.decomposer.element_symbols)
        self._has_known_bond_e: bool = all(s in BOND_ELECTRONS for s in self._symbols)
        self._unknown_bond_e_indices: np.ndarray = np.array(
            [i for i, sym in enumerate(self._symbols) if sym not in BOND_ELECTRONS],
            dtype=np.intp,
        )
        self._rdbe_coeffs_fallback = np.array(
            [0.5 * (BOND_ELECTRONS.get(s, 2) - 2) for s in self._symbols],
            dtype=np.float64,
        )

        if self._has_known_bond_e:
            self._rdbe_coeffs = self._rdbe_coeffs_fallback
        else:
            self._rdbe_coeffs = None

        # Isotope pre-filter coefficients (M+1/M+2 approximation)
        self._iso_m1_coeffs = np.array(
            [get_m1_ratio(s) for s in self._symbols],
            dtype=np.float64,
        )
        self._iso_m2_direct_coeffs = np.array(
            [get_m2_direct(s) for s in self._symbols],
            dtype=np.float64,
        )

    @staticmethod
    def _parse_adduct(
        adduct_str: str
    ) -> tuple[Formula, float]:
        """
        Parse adduct string and return Formula object and mass adjustment

        Args:
            adduct_str: Adduct formula string (must be neutral, no '+' allowed)
                Examples: 'Na', 'H', '-H', 'C2H3N'

        Returns:
            Tuple of (Formula object, mass to subtract from query mass)

        Raises:
            ValueError: If adduct string contains '+'
        """
        if '+' in adduct_str:
            raise ValueError(
                "Adduct string must not contain '+'. "
                "Specify charge separately using the 'charge' parameter."
            )

        # Handle negative adducts like '-H'
        if adduct_str.startswith('-'):
            adduct_formula_str = adduct_str[1:]  # Remove leading '-'
            adduct_formula = Formula(adduct_formula_str)
            adduct_mass = -adduct_formula.monoisotopic_mass
        else:
            adduct_formula = Formula(adduct_str)
            adduct_mass = adduct_formula.monoisotopic_mass

        return adduct_formula, adduct_mass

    def find_formulae(
        self,
        mass: float,
        charge: int = 0,
        error_ppm: Optional[float] = 0.0,
        error_da: Optional[float] = 0.0,
        adduct: Optional[str] = None,
        min_counts: Optional[dict[str, int] | str] = None,
        max_counts: Optional[dict[str, int] | str] = None,
        max_results: int = 500000,
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
        isotope_prefilter: Optional[np.ndarray] = None,
        iso_prefilter_tol_rel: float = 0.5,
        iso_prefilter_tol_abs: float = 0.3,
    ) -> 'FormulaSearchResults':
        """
        Find molecular formula candidates for a given mass.

        Decompose the query mass into candidate molecular formulae.
        Applies validation filters, and returns a sorted
        list of candidates with error metrics.

        Args:
            mass: Target mass to decompose (i.e. exact mass)

            charge: Charge state of the ion.
                Default: 0 (neutral)

            error_ppm: Mass tolerance in parts per million.
                Either ppm_error or mz_error must be specified.
                Default: 0.0

            error_da: Mass tolerance in Daltons.
                Either ppm_error or mz_error must be specified.
                Default: 0.0

            adduct: Neutral adduct formula to add/remove from the molecule.
                The adduct mass is subtracted before decomposition, then the
                adduct is added back to each candidate formula.
                Must be neutral (no '+' allowed); specify charge separately.

                Examples: "Na" for [M+Na]+, "H" for [M+H]+, "-H" for [M-H]-

                Note: For isotope matching, the ion composition is computed as
                (core + signed adduct offsets). Candidates that would yield
                negative ion element counts are discarded.

                Default: None (no adduct)

            min_counts: Minimum count for each element.
                Can be a dict like {"C": 5} or a string like "C5H10".
                String format: Elements not mentioned default to 0.

                Example: "C5" with elements "CHNOPS" means C≥5, H=N=O=P=S=0

                Default: None (no minimum)

            max_counts: Maximum count for each element.
                Can be a dict like {"C": 20, "H": 40} or a string like "C20H40".

                String format: Elements not mentioned default to 0, allowing
                intuitive parent ion constraints. Element counts default to 1
                if no number specified (e.g., "S" means "S1").

                Examples:
                - "C20H40" with elements "CHNOPS" means C≤20, H≤40, N=O=P=S=0
                - "C12H22O11" constrains to subsets of this parent ion
                - "C20H40P0" explicitly forbids phosphorus

                Default: None (no maximum)

            max_results: Maximum number of candidates to generate before
                filtering. There is a trade-off here; if this number is too
                low, then the real formula might sometimes not be found. Raising
                the value increases computational cost, however. I have not benchmarked
                a good number for this, but 10k has worked OK for me so far.

                Default: 10000

            filter_rdbe: Tuple of (min_rdbe, max_rdbe) to filter by
                Ring and Double Bond Equivalents.

                Ensure charge is specified if using this filter.

                Default: None (no RDBE filtering)

            check_octet: If True, only return formulae that obey the octet rule.

                Assumes typical biological oxidation states. Ensure charge is
                specified if using this filter.

                Default: False

            isotope_prefilter: Optional observed isotope envelope as a 2D
                ``[m/z, intensity]`` array used as a *fast decomposition-time
                gate only*. When provided, candidates whose approximate M+1/M+2
                abundance ratios are far from the observed envelope are dropped
                during decomposition (a perf optimization for large searches).

                This is NOT isotope scoring -- it omits candidates. For
                score-not-omit isotope matching, leave this None and use
                ``FormulaScorer.score(results, ms1_peaks=..., precursor_mz=...)``
                on the returned results instead.

                Default: None (no prefiltering; every candidate is returned)

            iso_prefilter_tol_rel: Relative tolerance for the M+1/M+2 prefilter.
                Only used when ``isotope_prefilter`` is given. Default: 0.5

            iso_prefilter_tol_abs: Absolute tolerance for the M+1/M+2 prefilter.
                Only used when ``isotope_prefilter`` is given. Default: 0.3

        Returns:
            FormulaSearchResults object containing candidates sorted by mass
            error (smallest first). These support iteration, indexing, filtering,
            and formatted printing.

        Raises:
            ValueError: If neither ppm_error nor mz_error is specified

        Example:
            >>> from find_mfs import FormulaFinder, FormulaScorer
            >>>
            >>> finder = FormulaFinder('CHNOPS')
            >>>
            >>> # Simple search with 5 ppm tolerance
            >>> finder.find_formulae(
            >>>     mass=180.063,
            >>>     error_ppm=5.0
            >>> )
            >>>
            >>> # Search for [M+Na]+ adduct
            >>> finder.find_formulae(
            >>>     mass=203.053,
            >>>     charge=1,
            >>>     adduct="Na",
            >>>     error_ppm=5.0
            >>> )

            >>> # Search for [M-H]- adduct (negative mode)
            >>> finder.find_formulae(
            >>>     mass=179.056,
            >>>     charge=-1,
            >>>     adduct="-H",
            >>>     error_ppm=5.0
            >>> )

            >>> # Advanced search with multiple filters
            >>> finder.find_formulae(
            >>>     mass=180.063,
            >>>     charge=1,
            >>>     error_ppm=5.0,
            >>>     min_counts={"C": 6},
            >>>     max_counts={"C": 12, "H": 24},
            >>>     filter_rdbe=(0, 15),
            >>>     check_octet=True
            >>> )

            >>> # Post-hoc filtering
            >>> filtered = results.filter_by_rdbe(5, 10)

            >>> # Isotope + mass scoring (score, don't omit)
            >>> import numpy as np
            >>> observed = np.array(
            >>>     [
            >>>        [180.063, 1.00],
            >>>        [181.067, 0.11],
            >>>     ]
            >>> )
            >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
            >>> FormulaScorer.default().score(
            >>>     results,
            >>>     ms1_peaks=observed,
            >>>     precursor_mz=180.063,
            >>> )
            >>> ranked = results.sort_by_posterior()
        """

        # Parse adduct if provided
        adduct_formula = None
        adduct_mass = 0.0
        if adduct:
            adduct_formula, adduct_mass = self._parse_adduct(adduct)
        adduct_sign = -1 if adduct is not None and adduct.startswith('-') else 1

        # Convert min_counts and max_counts into dicts, depending on user input
        # i.e. they might be strings
        min_counts_dict: dict[str, int] | None = None
        if isinstance(min_counts, dict):
            min_counts_dict = min_counts

        elif min_counts is not None:
            min_counts_dict = to_bounds_dict(
                min_counts,
                elements=[x.symbol for x in self.decomposer.elements]
            )

        max_counts_dict: dict[str, int] | None = None
        if isinstance(max_counts, dict):
            max_counts_dict = max_counts
        elif max_counts is not None:
            max_counts_dict = to_bounds_dict(
                max_counts,
                elements=[x.symbol for x in self.decomposer.elements]
            )

        # Adjust mass for adduct: decompose the neutral molecule mass
        adjusted_mass = mass - adduct_mass

        # Pre-filter on counts before constructing expensive Formula objects.
        # The decomposition always operates on the core molecule (adduct mass
        # subtracted), so RDBE/octet filtering applies to the core molecule's
        # element counts regardless of whether an adduct is specified.
        symbols = self._symbols
        can_prefilter: bool = (
            (filter_rdbe is not None or check_octet)
            and self._has_known_bond_e
        )

        # Prepare RDBE coefficients:
        # - full known set: safe for in-kernel pre-filtering and candidate RDBE
        # - partial known set: used only for residual post-filtering path
        rdbe_coeffs = None
        unknown_symbol_indices = None
        if self._has_known_bond_e:
            rdbe_coeffs = self._rdbe_coeffs
        elif filter_rdbe is not None or check_octet:
            rdbe_coeffs = self._rdbe_coeffs_fallback
            unknown_symbol_indices = self._unknown_bond_e_indices

        # Always pass RDBE coefficients so RDBE is computed for every candidate.
        # When pre-filtering is possible, also push RDBE range + octet checks
        # into the Cython decomposition kernel.
        # The octet check needs the *core molecule's* charge parity:
        #   - With adduct: adduct carries the charge; core is neutral (even parity)
        #   - Without adduct: charge is on the molecule itself
        decompose_kwargs = {}
        if rdbe_coeffs is not None:
            decompose_kwargs['rdbe_coeffs'] = rdbe_coeffs
        if can_prefilter:
            rdbe_min = filter_rdbe[0] if filter_rdbe is not None else -np.inf
            rdbe_max = filter_rdbe[1] if filter_rdbe is not None else np.inf
            core_charge_parity_even = True if adduct is not None else None
            decompose_kwargs.update({
                'rdbe_min': float(rdbe_min),
                'rdbe_max': float(rdbe_max),
                'check_octet': check_octet,
                'charge_parity_even': core_charge_parity_even,
            })

        # Approximate M+1/M+2 isotope pre-filter (perf gate, opt-in). Extracts
        # M+1/M+2 ratios from the observed envelope and pushes them into the
        # decomposition kernel to drop grossly mismatched candidates early.
        # This OMITS candidates and is off by default; it is not scoring.
        if (
            isotope_prefilter is not None
            and self._iso_m1_coeffs is not None
        ):
            obs_env = isotope_prefilter
            # Use lowest-mass peak as the monoisotopic reference,
            # since the base peak (tallest) may not be M+0 for
            # large molecules.
            mono_idx = np.argmin(obs_env[:, 0])
            mono_mz = obs_env[mono_idx, 0]
            mono_intsy = obs_env[mono_idx, 1]

            # Find M+1 and M+2 peaks relative to monoisotopic
            obs_m1_ratio = 0.0
            obs_m2_ratio = 0.0
            for i in range(obs_env.shape[0]):
                delta = obs_env[i, 0] - mono_mz
                if 0.9 <= delta <= 1.1:
                    obs_m1_ratio = obs_env[i, 1] / mono_intsy
                elif 1.9 <= delta <= 2.1:
                    obs_m2_ratio = obs_env[i, 1] / mono_intsy

            if obs_m1_ratio > 0.0:
                decompose_kwargs['iso_m1_coeffs'] = self._iso_m1_coeffs
                decompose_kwargs['iso_m2_direct_coeffs'] = self._iso_m2_direct_coeffs
                decompose_kwargs['obs_m1_ratio'] = obs_m1_ratio
                decompose_kwargs['obs_m2_ratio'] = obs_m2_ratio
                decompose_kwargs['iso_tol_rel'] = iso_prefilter_tol_rel
                decompose_kwargs['iso_tol_abs'] = iso_prefilter_tol_abs

        # Fused decomposition + scoring in one Cython pipeline call.
        # NOTE: adduct_mass from _parse_adduct() is signed:
        #   - "Na" -> +Na mass
        #   - "-H" -> -H mass
        # This signed mass must be used consistently for exact-mass scoring.
        adduct_mass_signed = adduct_mass

        if can_prefilter:
            remaining_filter_rdbe = None
            remaining_check_octet = False
        else:
            remaining_filter_rdbe = filter_rdbe
            remaining_check_octet = check_octet

        raw, symbols = self.decomposer.decompose_and_score(
            query_mass=adjusted_mass,
            charge=charge,
            ppm_error=error_ppm,
            mz_error=error_da,
            min_counts=min_counts_dict,
            max_counts=max_counts_dict,
            max_results=max_results,
            ion_query_mass=mass,
            adduct_mass=adduct_mass_signed,
            **decompose_kwargs,
        )

        # Signed adduct element offsets are needed for adduct-aware logic:
        # - ion composition (core + adduct) for isotope-envelope simulation
        # - residual octet parity (core is neutral when adduct is present)
        adduct_elements: dict[str, int] = {}
        if adduct_formula is not None:
            for sym, item in adduct_formula.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                if item.count > 0:
                    adduct_elements[sym] = adduct_sign * item.count

        # Compiled post-processing pipeline: residual rdbe/octet validation.
        from ._pipeline import run_query_pipeline
        raw = run_query_pipeline(
            raw=raw,
            core_symbols=symbols,
            charge=charge,
            query_mass=mass,
            remaining_filter_rdbe=remaining_filter_rdbe,
            remaining_check_octet=remaining_check_octet,
            adduct_elements=adduct_elements if adduct_elements else None,
            adduct_present=adduct_formula is not None,
            unknown_symbol_indices=unknown_symbol_indices,
        )

        # Store query parameters for reference
        query_params = {
            'mass': mass,
            'charge': charge,
            'error_ppm': error_ppm,
            'error_da': error_da,
            'adduct': adduct,
            'min_counts': min_counts,
            'max_counts': max_counts,
            'max_results': max_results,
            'filter_rdbe': filter_rdbe,
            'check_octet': check_octet,
        }

        from .results import FormulaSearchResults, _LazyBackend

        charge_mass_offset = ELECTRON.mass * charge if adduct_formula is not None else 0.0
        backend = _LazyBackend(
            raw=raw,
            symbols=symbols,
            charge=charge if adduct_formula is None else 0,
            ion_charge=charge,
            adduct=adduct,
            adduct_elements=adduct_elements if adduct_elements else None,
            charge_mass_offset=charge_mass_offset,
            adduct_mass=adduct_mass_signed,
        )

        return FormulaSearchResults(
            candidates=[],
            query_mass=mass,
            query_params=query_params,
            _backend=backend,
        )

    @property
    def element_set(self) -> set[str]:
        """
        Returns: a set of elements used by this Finder,
        i.e. {'C', 'H', 'N'..}
        """
        return set(self.decomposer.element_symbols)


# --- shared finder cache ----------------------------------------------------- #
# Building a FormulaFinder sets up a MassDecomposer and loads an Extended
# Residue Table. That is cheap once warm (~0.2 ms) but not free, and callers
# that decompose per-spectrum or per-peak would otherwise rebuild it constantly.
#
# The cache key is canonicalised, so 'CHNOPS' and ['C','H','N','O','P','S']
# resolve to the same instance -- element order is irrelevant because
# MassDecomposer sorts symbols by mass internally.


@lru_cache(maxsize=None)
def _parse_element_string(elements: str) -> frozenset:
    """Split an element string ('CHNOPS') into symbols. Cached: parsing a
    Formula is ~50x the cost of the cache lookup it guards."""
    return frozenset(str(e) for e in Formula(elements).composition().keys())


def _canonical_elements(
    elements: Union[str, Iterable[str]],
) -> frozenset:
    """Normalise an element spec to a hashable, order-independent key."""
    if isinstance(elements, str):
        return _parse_element_string(elements)
    return frozenset(str(e) for e in elements)


@lru_cache(maxsize=None)
def _build_finder(
    key: frozenset,
    use_precalculated: bool,
) -> 'FormulaFinder':
    return FormulaFinder(sorted(key), use_precalculated=use_precalculated)


def get_finder(
    elements: Union[str, Iterable[str]] = 'CHNOPS',
    *,
    use_precalculated: bool = True,
) -> 'FormulaFinder':
    """
    Return a shared, cached FormulaFinder for an element set.

    Prefer this over constructing FormulaFinder directly when the same element
    set is used repeatedly (per spectrum, per peak, across subsystems) -- it
    keeps one instance per set rather than one per call site.

    Args:
        elements: Elements to consider, as a string ('CHNOPS') or an iterable
            (['C', 'H', 'N', 'O', 'P', 'S']). Spelling and order do not matter;
            both forms of the same set return the same object.
        use_precalculated: Use pre-calculated Extended Residue Tables when
            available (CHNOPS and CHNOPS+halogens).

    Returns:
        A cached FormulaFinder. **Shared** -- treat it as read-only; do not
        mutate it in place.

    Example:
        >>> from find_mfs import get_finder
        >>> finder = get_finder('CHNOPS')
        >>> finder is get_finder(['C', 'H', 'N', 'O', 'P', 'S'])
        True
    """
    return _build_finder(_canonical_elements(elements), use_precalculated)


# Expose cache control on the public function so callers (and tests) don't have
# to reach into the private builder.
get_finder.cache_clear = _build_finder.cache_clear
get_finder.cache_info = _build_finder.cache_info
