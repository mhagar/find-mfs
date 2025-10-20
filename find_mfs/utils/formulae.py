"""
This module compares functions for manipulating/comparing formulae
"""
import re
from typing import Iterable
from molmass import Formula, ELEMENTS

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
    formula: str,
    elements: Iterable[str],
) -> dict[str, int]:
    """
    Convert constraint strings like "C20H10O5P0" into a dict that can be used
    as min_counts or max_counts arguments in MassDecomposer.decompose()

    Behaviour:
    - Elements mentioned in the string get their specified counts (i.e. "C20" → C: 20)
    - Elements without counts default to 1 (i.e. "S" → S: 1)
    - Elements in the `elements` list but NOT mentioned default to 0
    - Zero counts are explicit and allowed (i.e. "P0" → P: 0)

    Users can use this format to intuitively use a parent formula as
    a constraint. For example, if the parent ion is "C12H22O11", you can use
    this directly as max_counts to find all formulae that are
    'subsets' of this composition

    Args:
        formula: Constraint string like "C20H10O5" or "C12H22O11S0"
        elements: List of element symbols in the decomposer's element set.
            Any element in this list but not mentioned in `formula` will be
            set to 0 in the output

    Returns:
        Dict mapping element symbols to their constraint counts

    Raises:
        ValueError: If the formula string contains invalid element symbols,
            or if an element in the formula is not in the `elements` list

    Examples:
        >>> to_bounds_dict("C20H10O5", ["C", "H", "N", "O", "P", "S"])
        {'C': 20, 'H': 10, 'O': 5, 'N': 0, 'P': 0, 'S': 0}

        >>> to_bounds_dict("C12H22O11S", ["C", "H", "N", "O", "P", "S"])
        {'C': 12, 'H': 22, 'O': 11, 'S': 1, 'N': 0, 'P': 0}

        >>> to_bounds_dict("C20H10O5P0", ["C", "H", "N", "O", "P", "S"])
        {'C': 20, 'H': 10, 'O': 5, 'P': 0, 'N': 0, 'S': 0}
    """
    # Element symbol (upper + optional lowercase) followed by optional number
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    parsed = {}
    for symbol, count in matches:
        if not symbol:  # Skip empty matches
            continue

        if symbol not in ELEMENTS:
            raise ValueError(
                f"Invalid element symbol: '{symbol}'"
            )

        # Validate that the symbol is in the allowed element set
        if symbol not in elements:
            raise ValueError(
                f"Element '{symbol}' is not in the "
                f"given element set: {elements}"
            )

        # Parse count: defaults to 1 if not specified (i.e. "S" means "S1")
        parsed[symbol] = int(count) if count else 1

    # Start with all elements in the element set at 0
    output = {k: 0 for k in elements}

    # Update with parsed constraint values
    output.update(parsed)

    return output