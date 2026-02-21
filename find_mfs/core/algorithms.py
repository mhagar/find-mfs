"""
Mass decomposition using Bocker & Liptak algorithm,
(as implemented in SIRIUS) i.e. using an extended residue table

This algorithm was adapted from:
[Böcker & Lipták, 2007](https://link.springer.com/article/10.1007/s00453-007-0162-8)
[Böcker et. al., 2008](https://academic.oup.com/bioinformatics/article/25/2/218/218950)
"""
from math import gcd as _math_gcd


def _gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor of two integers
    """
    return _math_gcd(a, b)


def _is_decomposable(ERT, i: int, m: int, a1: int) -> bool:
    """
    Check if mass m is decomposable using first i+1 elements.
    Pure Python fallback for non-hot-path callers.
    """
    if m < 0:
        return False
    if ERT[int(m % a1), i] <= m:
        return True
    return False


# Import Cython implementation
try:
    from ._algorithms import _decompose_mass_range, decompose_and_score
except ImportError:
    raise ImportError(
        "Cython extension not built. Run: pip install -e \".[dev]\""
    )
