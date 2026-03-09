"""
Lightweight formula class that avoids expensive molmass.Formula parsing.

When element symbols, counts, charge, and monoisotopic mass are already known
(e.g. from vectorized decomposition), constructing a full molmass.Formula from
a string is wasteful — it re-parses what we already have.  LightFormula stores
these pre-computed values directly and exposes the same duck-type interface
used by the rest of the codebase.
"""
try:
    from ._light_formula import LightFormula
except ImportError:
    raise ImportError(
        "Cython LightFormula extension not built. "
        "Run: pip install -e \".[dev]\""
    )
