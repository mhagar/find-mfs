# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython extension type for LightFormula — C struct under the hood
for fast attribute access, from_counts(), and string building.
"""
from functools import reduce
from math import gcd

from molmass import Formula, Composition
from molmass.elements import ELEMENTS


cdef class LightFormula:
    """Drop-in replacement for molmass.Formula in the hot candidate path."""

    cdef readonly object _elements     # dict or None
    cdef readonly object _symbols      # list/tuple of str or None
    cdef readonly object _counts       # list/tuple of int or None
    cdef readonly int _charge
    cdef readonly double _monoisotopic_mass
    cdef object _formula_str           # cached str, built lazily
    cdef object _empirical_str         # cached str, built lazily

    def __init__(
        self,
        elements=None,
        int charge=0,
        double monoisotopic_mass=0.0,
        symbols=None,
        counts=None,
    ):
        if elements is not None and (symbols is not None or counts is not None):
            raise ValueError(
                "Provide either `elements` or (`symbols`, `counts`), not both."
            )

        if elements is None:
            if symbols is None and counts is None:
                elements = {}
            elif symbols is None or counts is None:
                raise ValueError(
                    "When `elements` is None, both `symbols` and `counts` are required."
                )
            elif len(symbols) != len(counts):
                raise ValueError(
                    f"`symbols` and `counts` must have same length; got "
                    f"{len(symbols)} and {len(counts)}"
                )

        self._elements = elements
        self._symbols = symbols
        self._counts = counts
        self._charge = charge
        self._monoisotopic_mass = monoisotopic_mass
        self._formula_str = None
        self._empirical_str = None

    @staticmethod
    def from_counts(
        symbols,
        counts,
        int charge=0,
        double monoisotopic_mass=0.0,
    ):
        """
        Construct from parallel symbols/counts with minimal overhead.

        This is intended for trusted hot paths (e.g. FormulaFinder) where
        symbols/counts shape is already validated upstream.
        """
        cdef LightFormula obj = LightFormula.__new__(LightFormula)
        obj._elements = None
        obj._symbols = symbols
        obj._counts = counts
        obj._charge = charge
        obj._monoisotopic_mass = monoisotopic_mass
        obj._formula_str = None
        obj._empirical_str = None
        return obj

    # ------------------------------------------------------------------
    # Properties matching molmass.Formula interface
    # ------------------------------------------------------------------

    @property
    def formula(self):
        """Hill-notation formula string (C first, H second, rest alphabetical),
        with charge notation matching molmass conventions."""
        if self._formula_str is None:
            self._formula_str = self._build_formula_str()
        return self._formula_str

    @property
    def monoisotopic_mass(self):
        return self._monoisotopic_mass

    @property
    def charge(self):
        return self._charge

    @property
    def empirical(self):
        """Empirical (simplest ratio) formula string."""
        if self._empirical_str is not None:
            return self._empirical_str
        self._empirical_str = self._build_empirical_str()
        return self._empirical_str

    cdef str _build_empirical_str(self):
        """Build empirical formula string (cached via property)."""
        cdef dict nonzero = {}
        cdef list items = self._nonzero_items()
        cdef int i, cnt, g_val
        for i in range(len(items)):
            sym, cnt = items[i]
            nonzero[sym] = cnt
        if not nonzero:
            return ''
        g_val = reduce(gcd, nonzero.values())
        reduced = {s: c // g_val for s, c in nonzero.items()}
        cdef list parts = []
        sorted_symbols = sorted(
            reduced,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )
        for sym in sorted_symbols:
            cnt = reduced[sym]
            parts.append(sym if cnt == 1 else f'{sym}{cnt}')
        return ''.join(parts)

    @property
    def atoms(self):
        """Total number of atoms."""
        cdef int total = 0
        cdef list items = self._nonzero_items()
        cdef int i
        for i in range(len(items)):
            total += <int>items[i][1]
        return total

    @property
    def nominal_mass(self):
        """Nominal (integer) mass — sum of most abundant isotope masses rounded."""
        cdef int total = 0
        cdef list items = self._nonzero_items()
        cdef int i, count
        for i in range(len(items)):
            symbol = items[i][0]
            count = items[i][1]
            elem = ELEMENTS[symbol]
            total += round(elem.isotopes[
                max(elem.isotopes, key=lambda k: elem.isotopes[k].abundance)
            ].mass) * count
        return total

    # ------------------------------------------------------------------
    # Composition — molmass-compatible
    # ------------------------------------------------------------------

    def composition(self):
        """Build a molmass.Composition matching Formula.composition() output."""
        cdef list nonzero_items = self._nonzero_items()
        cdef double total_mass = 0.0
        cdef int i, count
        cdef double mass_val, fraction

        for i in range(len(nonzero_items)):
            symbol = nonzero_items[i][0]
            count = nonzero_items[i][1]
            total_mass += ELEMENTS[symbol].mass * count

        cdef list comp_items = []
        for i in range(len(nonzero_items)):
            symbol = nonzero_items[i][0]
            count = nonzero_items[i][1]
            mass_val = ELEMENTS[symbol].mass * count
            fraction = mass_val / total_mass if total_mass > 0 else 0.0
            comp_items.append((symbol, count, mass_val, fraction))

        if self._charge != 0:
            comp_items.append(('e-', -self._charge, 0.0, 0.0))

        return Composition(comp_items)

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    def to_formula(self):
        """Convert to a real molmass.Formula."""
        return Formula(self.formula)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, LightFormula):
            merged = {}
            for item in self._nonzero_items():
                merged[item[0]] = item[1]
            for item in (<LightFormula>other)._nonzero_items():
                merged[item[0]] = merged.get(item[0], 0) + item[1]
            return LightFormula(
                elements=merged,
                charge=self._charge + (<LightFormula>other)._charge,
                monoisotopic_mass=self._monoisotopic_mass + (<LightFormula>other)._monoisotopic_mass,
            )

        if isinstance(other, Formula):
            merged = {}
            for item in self._nonzero_items():
                merged[item[0]] = item[1]
            for sym, comp_item in other.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                merged[sym] = merged.get(sym, 0) + comp_item.count
            return LightFormula(
                elements=merged,
                charge=self._charge + other.charge,
                monoisotopic_mass=self._monoisotopic_mass + other.monoisotopic_mass,
            )

        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Formula):
            return self.__add__(other)
        return NotImplemented

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self):
        return self.formula

    def __repr__(self):
        return f"LightFormula('{self.formula}')"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    cdef list _nonzero_items(self):
        """Return list of (symbol, count) tuples for non-zero elements."""
        cdef list result = []
        cdef int i, count

        if self._elements is not None:
            for symbol, count in (<dict>self._elements).items():
                if count > 0:
                    result.append((symbol, count))
            return result

        if self._symbols is not None and self._counts is not None:
            for i in range(len(self._symbols)):
                count = self._counts[i]
                if count > 0:
                    result.append((self._symbols[i], count))

        return result

    cdef str _build_formula_str(self):
        """Build Hill-notation string with molmass-style charge notation."""
        cdef list items = self._nonzero_items()
        cdef dict nonzero = {}
        cdef int i, cnt, abs_charge

        for i in range(len(items)):
            nonzero[items[i][0]] = items[i][1]

        sorted_symbols = sorted(
            nonzero,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )

        cdef list parts = []
        for sym in sorted_symbols:
            cnt = nonzero[sym]
            if cnt == 1:
                parts.append(sym)
            else:
                parts.append(f'{sym}{cnt}')

        cdef str base = ''.join(parts)

        if self._charge == 0:
            return base

        sign = '+' if self._charge > 0 else '-'
        abs_charge = abs(self._charge)
        if abs_charge == 1:
            return f'[{base}]{sign}'
        return f'[{base}]{abs_charge}{sign}'

    def _iter_nonzero_items(self):
        """Generator version for compatibility."""
        cdef list items = self._nonzero_items()
        cdef int i
        for i in range(len(items)):
            yield items[i]
