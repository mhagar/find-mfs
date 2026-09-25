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

def parse_counts(
    formula: str,
) -> dict[str, float]:
    """
    Parse a constraint string like "C*H*N*O*P0S2" into {symbol: count},
    keeping only the elements it mentions (in order of appearance)

    - Elements with a number get that count (i.e. "C20" -> C: 20)
    - Elements without a number default to 1 (i.e. "S" -> S: 1)
    - Zero counts are kept (i.e. "P0" -> P: 0)
    - Wildcard "*" denotes no bound (i.e. "O*" -> O: inf)

    Raises:
        ValueError: If the string contains an invalid element symbol.

    Examples:
        >>> parse_counts("C*H*N*O*P0S2")
        {'C': inf, 'H': inf, 'N': inf, 'O': inf, 'P': 0, 'S': 2}
    """
    # Element symbols
    pattern = r'([A-Z][a-z]?)(\d+|\*)?'

    parsed: dict[str, float] = {}
    for symbol, count in re.findall(pattern, formula):
        if symbol not in ELEMENTS:
            raise ValueError(
                f"Invalid element symbol: '{symbol}'"
            )

        if count == "*":
            parsed[symbol] = float('inf')
        elif count:
            parsed[symbol] = int(count)
        else:
            parsed[symbol] = 1

    return parsed


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
    - Wildcard "*" denotes no bounds (i.e. "O*" → O: inf)

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

        >>> to_bounds_dict("C6H7O*", ["C", "H", "N", "O", "P", "S"])
        {'C': 6, 'H': 7, 'O': inf, 'N': 0, 'P': 0, 'S': 0}
    """
    parsed = parse_counts(formula)

    # Validate that every symbol is in the allowed element set
    for symbol in parsed:
        if symbol not in elements:
            raise ValueError(
                f"Element '{symbol}' is not in the "
                f"given element set: {elements}"
            )

    # Start with all elements in the element set at 0
    output = {k: 0 for k in elements}

    # Update with parsed constraint values
    output.update(parsed)

    return output