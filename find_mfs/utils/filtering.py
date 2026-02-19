"""
This module contains functions for checking molecular formulae against
Senior's theorem, and the octet rule
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from molmass import Formula, CompositionItem

if TYPE_CHECKING:
    from ..core.light_formula import LightFormula

BOND_ELECTRONS: dict[str, int] = {
    'H': 1,
    'Li': 1,
    'Na': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'Cl': 1,
    'Br': 1,
    'I': 1,
    'S': 2,
    'P': 5,
    'Si': 4,
    'B': 3,
}

def passes_octet_rule(
    formula: Formula | LightFormula,
) -> bool:
    """
    Check if a molecular formula satisfies the octet rule.

    Args:
        formula: Molecular formula object

    Returns:
        True if formula satisfies octet rule, False otherwise
    """
    # Calculate RDBE
    rdbe = get_rdbe(formula)
    if rdbe is None:
        return False

    # Return true/false depending on charge and whether rdbe is integer
    charge = formula.charge
    if abs(charge) % 2.0 == 0.0:
        # charge is even; rdbe should be integer
        return not _is_half_integer(rdbe)

    elif abs(charge) % 2.0 == 1.0:
        # charge is odd; rdbe should be half-integer
        return _is_half_integer(rdbe)

    else:
        raise ValueError(f"Invalid charge: {charge}")

def get_rdbe(
        formula: Formula | LightFormula,
) -> Optional[float]:
    """
    Calculate Ring and Double Bond Equivalents (RDBE) for a molecular formula.

    ***NOTE***: This assumes no funny business is going on! i.e.
    no sulfoxides/sulfones, phosphine stuff, radicals, etc.
    This calculation should not be used in those cases.

    Formula:
        RDBE = 0.5 × Σ[n_i × (b_i - 2)] + 1

    where n_i is the number of atoms with b_i bonding electrons.

    For typical organic elements:
        - C: b=4, contributes +1 per carbon
        - H: b=1, contributes -0.5 per hydrogen
        - N: b=3, contributes +0.5 per nitrogen
        - O, S: b=2, contributes 0

    See:
    A Novel Formalism To Characterize the Degree of Unsaturation of
    Organic Molecules. Badertscher, M. et al. (2001)
    doi: 10.1021/ci000135o

    Args:
        formula: molmass.Formula instance

    Returns:
        RDBE value as float, or None if formula contains unhandled element
    """
    n_b_sub_2: list[int] = []
    for element in formula.composition().values():
        element: CompositionItem

        if element.symbol == 'e-':
            continue

        count = element.count
        num_bond_eles = get_bond_electrons(element.symbol)

        if not num_bond_eles:
            return None

        n_b_sub_2.append(
            count * (num_bond_eles - 2)
        )

    return (0.5 * sum(n_b_sub_2)) + 1

def get_bond_electrons(
    symbol: str,
) -> Optional[int]:
    return BOND_ELECTRONS.get(symbol, None)

def _is_half_integer(
    x: float
) -> bool:
    return (2*x) % 2 == 1


