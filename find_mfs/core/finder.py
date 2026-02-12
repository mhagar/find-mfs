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


@dataclass
class FormulaCandidate:
    """
    Structured result from formula finding.

    Comparisons with other FormulaCandidates uses absolute error_da
    (i.e. for expressions such as `form_cand_a > form_cand_b`

    Attributes:
        formula: The molecular formula as a molmass.Formula instance
        error_ppm: Mass error in parts per million
        error_da: Mass error in Daltons
        rdbe: Ring and Double Bond Equivalents (may be None for some elements)
        isotope_match_result: Results from isotope pattern matching if performed.
            Contains both aggregate score (for filtering) and detailed per-peak
            information (for inspection).
    """
    formula: Union[Formula, LightFormula]
    error_ppm: float
    error_da: float
    rdbe: Optional[float]
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

        # Get raw element count vectors (avoids Formula construction)
        counts, symbols = self.decomposer.decompose_to_counts(
            query_mass=adjusted_mass,
            charge=charge,
            ppm_error=error_ppm,
            mz_error=error_da,
            min_counts=min_counts_dict,
            max_counts=max_counts_dict,
            max_results=max_results,
        )

        # Pre-filter on counts before constructing expensive Formula objects.
        # Only possible without adducts (adducts change composition/RDBE).
        has_known_bond_e = all(s in BOND_ELECTRONS for s in symbols)
        can_prefilter = (
            adduct is None
            and (filter_rdbe is not None or check_octet)
            and len(counts) > 0
            and has_known_bond_e
        )

        # Prepare RDBE coefficients (reused for pre-filtering and candidate RDBE)
        if has_known_bond_e:
            rdbe_coeffs = np.array(
                [0.5 * (BOND_ELECTRONS[s] - 2) for s in symbols],
                dtype=np.float64,
            )

        if can_prefilter:
            # Vectorized RDBE computation: RDBE = counts @ coeffs + 1
            counts_f = counts.astype(np.float64)
            all_rdbes = counts_f @ rdbe_coeffs + 1.0

            mask = np.ones(len(counts), dtype=bool)

            if filter_rdbe is not None:
                min_r, max_r = filter_rdbe
                mask &= (min_r <= all_rdbes) & (all_rdbes <= max_r)

            if check_octet:
                doubled = 2.0 * all_rdbes
                is_half_int = doubled % 2.0 == 1.0
                if abs(charge) % 2 == 0:
                    # Even charge: RDBE must NOT be half-integer
                    mask &= ~is_half_int
                else:
                    # Odd charge: RDBE must be half-integer
                    mask &= is_half_int

            counts = counts[mask]
            # RDBE/octet already done; only isotope matching remains
            remaining_filter_rdbe = None
            remaining_check_octet = False
        else:
            remaining_filter_rdbe = filter_rdbe
            remaining_check_octet = check_octet

        # Vectorized computation of error and RDBE from count vectors.
        # A single numpy matmul replaces N per-formula Python calls to
        # formula.monoisotopic_mass and get_rdbe(formula).
        # Verified to match Formula.monoisotopic_mass to <1e-13 Da.
        charge_mass_offset = ELECTRON.mass * charge
        real_masses_arr = self.decomposer.real_masses

        if len(counts) > 0:
            counts_f = counts.astype(np.float64)
            exact_masses = counts_f @ real_masses_arr - charge_mass_offset
            if adduct_formula is not None:
                exact_masses += adduct_formula.monoisotopic_mass
            all_err_ppm = (exact_masses - mass) / mass * 1e6
            all_err_da = exact_masses - mass
            if has_known_bond_e:
                all_rdbes = counts_f @ rdbe_coeffs + 1.0
            else:
                all_rdbes = None

            # Pre-sort by absolute error so candidates come out sorted
            sort_order = np.argsort(np.abs(all_err_ppm))
        else:
            sort_order = np.array([], dtype=np.intp)

        # Build candidates in sorted order: construct Formula, validate,
        # and create FormulaCandidate in one pass.
        needs_validation = (
            remaining_filter_rdbe is not None
            or remaining_check_octet
            or isotope_match is not None
        )

        candidates: list[FormulaCandidate] = []

        # Pre-extract adduct element counts (if any) to avoid repeated
        # composition() calls inside the loop.
        if adduct_formula is not None:
            adduct_elements = {}
            for sym, item in adduct_formula.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                if item.count > 0:
                    adduct_elements[sym] = item.count
            adduct_charge = adduct_formula.charge
        else:
            adduct_elements = None
            adduct_charge = 0

        for si in sort_order:
            row = counts[si]

            # Construct LightFormula directly from counts (avoids Formula parsing).
            # exact_masses[si] already includes adduct mass (added vectorized above),
            # so we merge adduct elements into the dict without re-adding its mass.
            elements_dict = {
                s: int(c) for s, c in zip(symbols, row) if c > 0
            }

            if adduct_elements is not None:
                for sym, cnt in adduct_elements.items():
                    elements_dict[sym] = elements_dict.get(sym, 0) + cnt

            formula = LightFormula(
                elements=elements_dict,
                charge=charge + adduct_charge,
                monoisotopic_mass=float(exact_masses[si]),
            )

            # Validate (remaining RDBE/octet for adduct case, + isotope matching)
            isotope_result = None
            if needs_validation:
                passes, isotope_result = self.validator.validate(
                    formula=formula,
                    filter_rdbe=remaining_filter_rdbe,
                    check_octet=remaining_check_octet,
                    isotope_match_config=isotope_match,
                )
                if not passes:
                    continue

            candidates.append(
                FormulaCandidate(
                    formula=formula,
                    error_ppm=float(all_err_ppm[si]),
                    error_da=float(all_err_da[si]),
                    rdbe=float(all_rdbes[si]) if all_rdbes is not None else None,
                    isotope_match_result=isotope_result,
                )
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

        # Import here to avoid circular dependency
        from .results import FormulaSearchResults
        return FormulaSearchResults(
            candidates=candidates,
            query_mass=mass,
            query_params=query_params
        )

    @property
    def element_set(self) -> set[str]:
        """
        Returns: a set of elements used by this Finder,
        i.e. {'C', 'H', 'N'..}
        """
        return set(self.decomposer.element_symbols)