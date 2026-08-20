"""
Contains functions for featurizing formulae for
the chemical prior GMM
"""
from __future__ import annotations

import numpy as np
from molmass import Formula

from find_mfs.core._light_formula import LightFormula

# Feature layout (order matters, must match featurize_formula() )
RATIO_FEATURES = ('H', 'O')                    # X/C ratio
COUNT_FEATURES = ('N', 'S', 'P', 'Cl', 'Br', 'I')  # ele counts
# + RDBE, RDBE/C, n_heteroatom_types  (3 extra)
N_FEATURES = len(RATIO_FEATURES) + len(COUNT_FEATURES) + 3

# Column indices of corresponding to halogen count
ROUTING_HALOGENS = ('Cl', 'Br', 'I')
HALOGEN_COLS = tuple(
    len(RATIO_FEATURES) + COUNT_FEATURES.index(h) for h in ROUTING_HALOGENS
)

_HETEROATOMS = {'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se', 'Si', 'B'}


def _rdbe(
        counts: dict[str, int],
) -> float:
    """
    # TODO: Unify this w utils.filtering.get_rdbe()
    # TODO: Should just unify formula encoding more broadly tbh

    Ring and double bond equivalence.
    RDBE = 1 + C - H/2 + N/2 + P/2 - (Cl + Br + I + F)/2
    """
    c = counts.get('C', 0)
    h = counts.get('H', 0)
    n = counts.get('N', 0)
    p = counts.get('P', 0)
    hal = (
        counts.get('F', 0) + counts.get('Cl', 0)
        + counts.get('Br', 0) + counts.get('I', 0)
    )
    return 1.0 + c - h / 2.0 + n / 2.0 + p / 2.0 - hal / 2.0


def featurize_formula(
        formula: Formula | LightFormula
) -> np.ndarray | None:
    """
    Convert molmass Formula object into GMM feature vector
    Returns None if the formula has no carbon (can't compute ratios).
    """
    elem_counts = _get_element_counts(formula)
    c_count = elem_counts.get('C', 0)
    if c_count == 0:
        return None

    feats = np.empty(N_FEATURES, dtype=np.float64)
    i = 0

    # Ratio features (X / C)
    for elem in RATIO_FEATURES:
        feats[i] = elem_counts.get(elem, 0) / c_count
        i += 1

    # Count features
    for elem in COUNT_FEATURES:
        feats[i] = elem_counts.get(elem, 0)
        i += 1

    # RDBE
    rdbe = _rdbe(elem_counts)
    feats[i] = rdbe
    i += 1

    # RDBE / C
    feats[i] = rdbe / c_count
    i += 1

    # Number of distinct heteroatom types
    feats[i] = sum(1 for e in _HETEROATOMS if elem_counts.get(e, 0) > 0)

    return feats


def contains_halogen(
        feature: np.ndarray,
) -> bool:
    return bool(np.any(feature[list(HALOGEN_COLS)] > 0))

def split_matrix_by_halogen(
        feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a matrix of featurized formulae, separates it
    into two matrices (halofree, halogen)
    """
    halo_mask = feature_matrix[:, list(HALOGEN_COLS)].sum(axis=1) > 0
    return (
        feature_matrix[~halo_mask],
        feature_matrix[halo_mask],
    )


def _get_element_counts(
        formula: Formula | LightFormula,
) -> dict[str, int]:
    """
    Extract element counts from a Formula or LightFormula
    """
    counts: dict[str, int] = {}
    comp = formula.composition()
    for symbol, item in comp.items():
        if symbol == '' or symbol == 'e-':
            continue
        if item.count > 0:
            counts[symbol] = item.count
    return counts


