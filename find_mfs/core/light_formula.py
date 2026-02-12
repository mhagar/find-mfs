"""
Lightweight formula class that avoids expensive molmass.Formula parsing.

When element symbols, counts, charge, and monoisotopic mass are already known
(e.g. from vectorized decomposition), constructing a full molmass.Formula from
a string is wasteful — it re-parses what we already have.  LightFormula stores
these pre-computed values directly and exposes the same duck-type interface
used by the rest of the codebase.
"""
from __future__ import annotations

from functools import reduce
from math import gcd

from molmass import Formula, Composition, CompositionItem
from molmass.elements import ELEMENTS


class LightFormula:
    """Drop-in replacement for molmass.Formula in the hot candidate path."""

    __slots__ = ('_elements', '_charge', '_monoisotopic_mass', '_formula_str')

    def __init__(
        self,
        elements: dict[str, int],
        charge: int = 0,
        monoisotopic_mass: float = 0.0,
    ):
        self._elements = elements
        self._charge = charge
        self._monoisotopic_mass = monoisotopic_mass
        self._formula_str: str | None = None

    # ------------------------------------------------------------------
    # Properties matching molmass.Formula interface
    # ------------------------------------------------------------------

    @property
    def formula(self) -> str:
        """Hill-notation formula string (C first, H second, rest alphabetical),
        with charge notation matching molmass conventions."""
        if self._formula_str is None:
            self._formula_str = self._build_formula_str()
        return self._formula_str

    @property
    def monoisotopic_mass(self) -> float:
        return self._monoisotopic_mass

    @property
    def charge(self) -> int:
        return self._charge

    @property
    def empirical(self) -> str:
        """Empirical (simplest ratio) formula string."""
        nonzero = {s: c for s, c in self._elements.items() if c > 0}
        if not nonzero:
            return ''
        g = reduce(gcd, nonzero.values())
        reduced = {s: c // g for s, c in nonzero.items()}
        parts: list[str] = []
        sorted_symbols = sorted(
            reduced,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )
        for sym in sorted_symbols:
            cnt = reduced[sym]
            parts.append(sym if cnt == 1 else f'{sym}{cnt}')
        return ''.join(parts)

    @property
    def atoms(self) -> int:
        """Total number of atoms."""
        return sum(c for c in self._elements.values() if c > 0)

    @property
    def nominal_mass(self) -> int:
        """Nominal (integer) mass — sum of most abundant isotope masses rounded."""
        total = 0
        for symbol, count in self._elements.items():
            if count == 0:
                continue
            total += round(ELEMENTS[symbol].isotopes[
                max(ELEMENTS[symbol].isotopes, key=lambda k: ELEMENTS[symbol].isotopes[k].abundance)
            ].mass) * count
        return total

    # ------------------------------------------------------------------
    # Composition — molmass-compatible
    # ------------------------------------------------------------------

    def composition(self) -> Composition:
        """Build a molmass.Composition matching Formula.composition() output."""
        # First pass: compute total mass for fractions
        total_mass = 0.0
        for symbol, count in self._elements.items():
            if count == 0:
                continue
            total_mass += ELEMENTS[symbol].mass * count

        # Build tuple list: (symbol, count, mass, fraction)
        items: list[tuple[str, int, float, float]] = []
        for symbol, count in self._elements.items():
            if count == 0:
                continue
            mass = ELEMENTS[symbol].mass * count
            fraction = mass / total_mass if total_mass > 0 else 0.0
            items.append((symbol, count, mass, fraction))

        # Add electron entry if charged (matches molmass behavior)
        if self._charge != 0:
            items.append(('e-', -self._charge, 0.0, 0.0))

        return Composition(items)

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    def to_formula(self) -> Formula:
        """Convert to a real molmass.Formula."""
        return Formula(self.formula)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: LightFormula | Formula) -> LightFormula:
        if isinstance(other, LightFormula):
            merged = dict(self._elements)
            for sym, cnt in other._elements.items():
                merged[sym] = merged.get(sym, 0) + cnt
            return LightFormula(
                elements=merged,
                charge=self._charge + other._charge,
                monoisotopic_mass=self._monoisotopic_mass + other._monoisotopic_mass,
            )

        if isinstance(other, Formula):
            merged = dict(self._elements)
            for sym, item in other.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                merged[sym] = merged.get(sym, 0) + item.count
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

    def __str__(self) -> str:
        return self.formula

    def __repr__(self) -> str:
        return f"LightFormula('{self.formula}')"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_formula_str(self) -> str:
        """Build Hill-notation string with molmass-style charge notation."""
        parts: list[str] = []

        # Hill order: C first, H second, rest alphabetical
        sorted_symbols = sorted(
            (sym for sym, cnt in self._elements.items() if cnt > 0),
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )

        for sym in sorted_symbols:
            cnt = self._elements[sym]
            if cnt == 1:
                parts.append(sym)
            else:
                parts.append(f'{sym}{cnt}')

        base = ''.join(parts)

        # Charge notation matching molmass conventions:
        # neutral -> "C6H12O6"
        # charge=1 -> "[C6H12O6]+"
        # charge=-1 -> "[C6H12O6]-"
        # charge=2 -> "[C6H12O6]2+"
        # charge=-2 -> "[C6H12O6]2-"
        if self._charge == 0:
            return base

        sign = '+' if self._charge > 0 else '-'
        abs_charge = abs(self._charge)
        if abs_charge == 1:
            return f'[{base}]{sign}'
        return f'[{base}]{abs_charge}{sign}'
