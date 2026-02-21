"""
Main API entry point for find_mfs

This module contains FormulaFinder, which orchestrates
- mass decomposition
- formula validation
"""
from dataclasses import dataclass
from typing import Optional, Union, Iterable, TYPE_CHECKING

import numpy as np
from molmass import Formula

from molmass.elements import ELECTRON

from .decomposer import MassDecomposer
from .light_formula import LightFormula
from .validator import FormulaValidator
from ..utils.filtering import BOND_ELECTRONS
from ..utils.formulae import to_bounds_dict

if TYPE_CHECKING:
    from ..isotopes.config import IsotopeMatchConfig
    from ..isotopes.results import IsotopeMatchResult
    from .results import FormulaSearchResults


@dataclass(slots=True)
class FormulaCandidate:
    """
    Structured result from formula finding.

    Comparisons with other FormulaCandidates uses absolute error_da
    (i.e. for expressions such as `form_cand_a > form_cand_b`

    Attributes:
        formula: The core molecular formula (without adduct) as a
            molmass.Formula or LightFormula instance
        error_ppm: Mass error in parts per million
        error_da: Mass error in Daltons
        rdbe: Ring and Double Bond Equivalents of the core molecule
            (may be None for some elements)
        adduct: Adduct string as specified by the user (e.g. "Na", "-H"),
            or None if no adduct was specified
        isotope_match_result: Results from isotope pattern matching if performed.
            Contains both aggregate score (for filtering) and detailed per-peak
            information (for inspection).
    """
    formula: Union[Formula, LightFormula]
    error_ppm: float
    error_da: float
    rdbe: Optional[float]
    adduct: Optional[str] = None
    isotope_match_result: Optional['IsotopeMatchResult'] = None

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
        from ..isotopes._isospec_bridge import M1_RATIOS, M2_DIRECT
        self._iso_m1_coeffs = np.array(
            [M1_RATIOS.get(s, 0.0) for s in self._symbols],
            dtype=np.float64,
        )
        self._iso_m2_direct_coeffs = np.array(
            [M2_DIRECT.get(s, 0.0) for s in self._symbols],
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
        max_results: int = 10000,
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
        isotope_match: Optional['IsotopeMatchConfig'] = None,
    ) -> 'FormulaSearchResults':
        """
        Find molecular formula candidates for a given mass.

        This method decomposes the query mass into possible elemental
        compositions, applies validation filters, and returns a sorted
        list of candidates with error metrics.

        Args:
            mass: Target mass to decompose (m/z value)

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
                adduct is added back to each candidate formula. Must be neutral
                (no '+' allowed); specify charge separately.
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
                filtering. This limits computational cost for broad searches.
                Default: 10000

            filter_rdbe: Tuple of (min_rdbe, max_rdbe) to filter by
                Ring and Double Bond Equivalents. Ensure charge is specified
                if using this filter.
                Default: None (no RDBE filtering)

            check_octet: If True, only return formulae that obey the octet rule.
                Assumes typical biological oxidation states. Ensure charge is
                specified if using this filter.
                Default: False

            isotope_match: SingleEnvelopeMatch config for isotope pattern
                validation. If provided, only returns
                formulae whose predicted isotope pattern matches the observed
                pattern. Requires IsoSpecPy to be installed.
                Default: None (no isotope matching)

        Returns:
            FormulaSearchResults object containing candidates sorted by mass
            error (smallest first). Supports iteration, indexing, filtering,
            and pretty printing.

        Raises:
            ValueError: If neither ppm_error nor mz_error is specified
            ImportError: If isotope matching is requested but IsoSpecPy not installed

        Example:
            >>> from find_mfs import FormulaFinder
            >>> from find_mfs.isotopes import SingleEnvelopeMatch
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

            >>> # With isotope matching
            >>> import numpy as np
            >>> envelope = np.array(
            >>>     [
            >>>        [180.063, 1.00],
            >>>        [181.067, 0.11],
            >>>     ]
            >>> )
            >>> iso_config = SingleEnvelopeMatch(envelope, mz_tolerance=0.01)
            >>> results = finder.find_formulae(
            >>>     mass=180.063,
            >>>     error_ppm=5.0,
            >>>     isotope_match=iso_config
            >>> )
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

        # Isotope pre-filter: extract M+1/M+2 ratios from observed envelope
        if (
            isotope_match is not None
            and isotope_match.enable_approx_prefilter
            and self._iso_m1_coeffs is not None
        ):
            obs_env = isotope_match.envelope
            base_idx = np.argmax(obs_env[:, 1])
            base_mz = obs_env[base_idx, 0]

            # Find M+1 peak (delta 0.9–1.1 Da from base)
            obs_m1_ratio = 0.0
            obs_m2_ratio = 0.0
            for i in range(obs_env.shape[0]):
                delta = obs_env[i, 0] - base_mz
                if 0.9 <= delta <= 1.1:
                    obs_m1_ratio = obs_env[i, 1] / obs_env[base_idx, 1]
                elif 1.9 <= delta <= 2.1:
                    obs_m2_ratio = obs_env[i, 1] / obs_env[base_idx, 1]

            if obs_m1_ratio > 0.0:
                decompose_kwargs['iso_m1_coeffs'] = self._iso_m1_coeffs
                decompose_kwargs['iso_m2_direct_coeffs'] = self._iso_m2_direct_coeffs
                decompose_kwargs['obs_m1_ratio'] = obs_m1_ratio
                decompose_kwargs['obs_m2_ratio'] = obs_m2_ratio
                decompose_kwargs['iso_tol_rel'] = isotope_match.approx_tolerance_rel
                decompose_kwargs['iso_tol_abs'] = isotope_match.approx_tolerance_abs

        # Fused decomposition + scoring + optional isotope matching:
        # all in one Cython pipeline call.
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
        # - isotope matching on ion composition (core + adduct)
        # - residual octet parity (core is neutral when adduct is present)
        adduct_elements: dict[str, int] = {}
        if adduct_formula is not None:
            for sym, item in adduct_formula.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                if item.count > 0:
                    adduct_elements[sym] = adduct_sign * item.count

        # Compiled post-processing pipeline:
        # - residual rdbe/octet validation
        # - isotope matching + rmse cutoff
        from ._pipeline import run_query_pipeline
        raw = run_query_pipeline(
            raw=raw,
            core_symbols=symbols,
            charge=charge,
            query_mass=mass,
            remaining_filter_rdbe=remaining_filter_rdbe,
            remaining_check_octet=remaining_check_octet,
            isotope_match=isotope_match,
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
            'isotope_match': isotope_match,
        }

        from .results import FormulaSearchResults, _LazyBackend

        charge_mass_offset = ELECTRON.mass * charge if adduct_formula is not None else 0.0
        n_obs = isotope_match.envelope.shape[0] if isotope_match is not None else 0
        backend = _LazyBackend(
            raw=raw,
            symbols=symbols,
            charge=charge if adduct_formula is None else 0,
            ion_charge=charge,
            adduct=adduct,
            adduct_elements=adduct_elements if adduct_elements else None,
            n_obs=n_obs,
            charge_mass_offset=charge_mass_offset,
            adduct_mass=adduct_mass_signed,
            simulated_mz_tolerance=(
                isotope_match.simulated_mz_tolerance
                if isotope_match is not None else None
            ),
            simulated_intensity_threshold=(
                isotope_match.simulated_intensity_threshold
                if isotope_match is not None else None
            ),
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
