"""
This module compares functions for manipulating/comparing formulae
"""
from molmass import Formula

def formula_match(
    formula_a: Formula,
    formula_b: Formula,
) -> bool:
    """
    Returns true if two formulae are the same
    """
    if formula_a.formula == formula_b.formula:
        return True
    return False

def to_bounds_dict(
    formula: Formula | str,
    elements: list[str],
) -> dict[str, int]:
    """
    Given a formula, returns a dict which can be used as the
    'min_counts' or 'max_counts' arguments in MassDecomposer.decompose().

    `elements` should be a list of the elements in MassDecomposer's table;
    these will be set to '0' in the output dict
    """
    if isinstance(formula, str):
        formula = Formula(formula)

    if formula.charge != 0:
        raise ValueError(
            f"Formula must have a charge of 0 (given: {formula.charge})"
        )

    # Start with dict of all 0's
    output = {
        k: 0 for k in elements
    }

    # Update dict with elements from formula
    output.update(
        {
            x.symbol: x.count for x in formula.composition().values()
        }
    )

    return output