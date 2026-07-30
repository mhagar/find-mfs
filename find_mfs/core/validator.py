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

if TYPE_CHECKING:
    from .light_formula import LightFormula


class FormulaValidator:
    """
    Validates molecular formulae against chemical rules
    and constraints.

    This class provides methods to check formulae against:
    - RDBE (Ring and Double Bond Equivalent) constraints
    - Octet rule

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
    ) -> bool:
        """
        Validate a formula against RDBE and octet constraints.

        Args:
            formula: Formula object to validate
            filter_rdbe: Tuple of (min_rdbe, max_rdbe) if RDBE filtering desired
            check_octet: If True, check octet rule

        Returns:
            True if the formula passes all requested checks, False otherwise.
        """
        # Check RDBE constraints
        if filter_rdbe is not None:
            min_rdbe, max_rdbe = filter_rdbe
            if not self.validate_rdbe(formula, min_rdbe, max_rdbe):
                return False

        # Check octet rule
        if check_octet:
            if not passes_octet_rule(formula):
                return False

        # All checks passed
        return True
