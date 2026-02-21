"""
This module provides the FormulaValidator class for checking molecular
formulae against various chemical rules/constraints
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from molmass import Formula

from ..utils.filtering import (
    passes_octet_rule,
    get_rdbe,
)
from ..isotopes.envelope import match_isotope_envelope, match_isotope_envelope_fast

if TYPE_CHECKING:
    from ..isotopes.config import SingleEnvelopeMatch, IsotopeMatchConfig
    from ..isotopes.results import IsotopeMatchResult
    from .light_formula import LightFormula


class FormulaValidator:
    """
    Validates molecular formulae against chemical rules
    and constraints.

    This class provides methods to check formulae against:
    - RDBE (Ring and Double Bond Equivalent) constraints
    - Octet rule
    - Isotope pattern matching

    Example:
        >>> formula: Formula
        >>> validator = FormulaValidator()

        >>> # Check RDBE
        >>> if validator.validate_rdbe(formula, min_rdbe=0, max_rdbe=10):
        >>>     print("Valid RDBE")
        >>>
        >>> # Validate with multiple criteria
        >>> if validator.validate(
        >>>     formula,
        >>>     filter_rdbe=(0, 10),
        >>>     check_octet=True
        >>> ):
        >>>     print("Formula is valid")
    """
    @staticmethod
    def validate_rdbe(
        formula: Formula | LightFormula,
        min_rdbe: float,
        max_rdbe: float
    ) -> bool:
        """
        Check if formula's RDBE falls within specified range.

        Args:
            formula: Formula object to validate
            min_rdbe: Minimum acceptable RDBE value
            max_rdbe: Maximum acceptable RDBE value

        Returns:
            True if RDBE is within range, False otherwise
        """
        rdbe = get_rdbe(formula)

        if rdbe is None:
            # Formula contains elements we can't calculate RDBE for
            return False

        return min_rdbe <= rdbe <= max_rdbe

    def validate(
        self,
        formula: Formula | LightFormula,
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
        isotope_match_config: Optional['IsotopeMatchConfig'] = None,
    ) -> tuple[bool, Optional['IsotopeMatchResult']]:
        """
        Validate a formula and return both validation result, and isotope match details
        (if a config was given)

        Args:
            formula: Formula object to validate
            filter_rdbe: Tuple of (min_rdbe, max_rdbe) if RDBE filtering desired
            check_octet: If True, check octet rule
            isotope_match_config: SingleEnvelopeMatch config for isotope
                pattern validation

        Returns:
            Tuple of (passes_validation, isotope_match_result):
            - passes_validation: True if formula passes all checks
            - isotope_match_result: Isotope matching details if performed, None otherwise
        """
        # Check RDBE constraints
        if filter_rdbe is not None:
            min_rdbe, max_rdbe = filter_rdbe
            if not self.validate_rdbe(formula, min_rdbe, max_rdbe):
                return False, None

        # Check octet rule
        if check_octet:
            if not passes_octet_rule(formula):
                return False, None

        # Check isotope pattern
        isotope_result = None
        if isotope_match_config is not None:
            # Convert ppm to Da and use the largest one
            ppm_to_da = (
                1e-6
                * isotope_match_config.mz_tolerance_ppm
                * formula.monoisotopic_mass
            )
            if isotope_match_config.mz_tolerance_da > ppm_to_da:
                mz_tol = isotope_match_config.mz_tolerance_da
            else:
                mz_tol = ppm_to_da

            # Match isotope result
            isotope_result = match_isotope_envelope(
                formula=formula,
                observed_envelope=isotope_match_config.envelope,
                mz_match_tolerance=mz_tol,
                simulated_envelope_mz_tolerance=isotope_match_config.simulated_mz_tolerance,
                simulated_envelope_intsy_threshold=isotope_match_config.simulated_intensity_threshold,
            )

            # Check if match is good enough
            # (i.e more than one peak should match)
            if isotope_result.intensity_rmse > isotope_match_config.minimum_rmse:
                return False, isotope_result

        # All checks passed
        return True, isotope_result

    @staticmethod
    def validate_isotope_fast(
        symbols: list[str] | tuple[str, ...],
        counts: list[int] | tuple[int, ...],
        charge: int,
        monoisotopic_mass: float,
        isotope_match_config: 'IsotopeMatchConfig',
    ) -> tuple[bool, Optional['IsotopeMatchResult']]:
        """
        Fast isotope validation using Cython + C++ IsoSpecPy path.

        Bypasses string-based Formula construction for ~30x speedup.

        Args:
            symbols: Element symbols
            counts: Atom counts matching symbols
            charge: Ion charge state
            monoisotopic_mass: Monoisotopic mass for ppm→Da conversion
            isotope_match_config: Isotope matching configuration

        Returns:
            Tuple of (passes, isotope_result)
        """
        ppm_to_da = (
            1e-6
            * isotope_match_config.mz_tolerance_ppm
            * monoisotopic_mass
        )
        if isotope_match_config.mz_tolerance_da > ppm_to_da:
            mz_tol = isotope_match_config.mz_tolerance_da
        else:
            mz_tol = ppm_to_da

        isotope_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=charge,
            observed_envelope=isotope_match_config.envelope,
            mz_match_tolerance=mz_tol,
            simulated_mz_tolerance=isotope_match_config.simulated_mz_tolerance,
            simulated_intensity_threshold=isotope_match_config.simulated_intensity_threshold,
        )

        if isotope_result.intensity_rmse > isotope_match_config.minimum_rmse:
            return False, isotope_result

        return True, isotope_result
