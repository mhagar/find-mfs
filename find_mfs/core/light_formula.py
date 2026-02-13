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

    __slots__ = (
        '_elements',
        '_symbols',
        '_counts',
        '_charge',
        '_monoisotopic_mass',
        '_formula_str',
    )

    def __init__(
        self,
        elements: dict[str, int] | None = None,
        charge: int = 0,
        monoisotopic_mass: float = 0.0,
        symbols: list[str] | tuple[str, ...] | None = None,
        counts: list[int] | tuple[int, ...] | None = None,
    ):
        if elements is not None and (symbols is not None or counts is not None):
            raise ValueError(
                "Provide either `elements` or (`symbols`, `counts`), not both."
            )

        if elements is None:
            if symbols is None and counts is None:
                # Maintain current behavior for callers that pass no composition.
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
        self._formula_str: str | None = None

    @classmethod
    def from_counts(
        cls,
        symbols: list[str] | tuple[str, ...],
        counts: list[int] | tuple[int, ...],
        charge: int = 0,
        monoisotopic_mass: float = 0.0,
    ) -> 'LightFormula':
        """
        Construct from parallel symbols/counts with minimal overhead.

        This is intended for trusted hot paths (e.g. FormulaFinder) where
        symbols/counts shape is already validated upstream.
        """
        obj = cls.__new__(cls)
        obj._elements = None
        obj._symbols = symbols
        obj._counts = counts
        obj._charge = charge
        obj._monoisotopic_mass = monoisotopic_mass
        obj._formula_str = None
        return obj

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
        nonzero = {}
        for symbol, count in self._iter_nonzero_items():
            nonzero[symbol] = count
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
        total = 0
        for _, count in self._iter_nonzero_items():
            total += count
        return total

    @property
    def nominal_mass(self) -> int:
        """Nominal (integer) mass — sum of most abundant isotope masses rounded."""
        total = 0
        for symbol, count in self._iter_nonzero_items():
            total += round(ELEMENTS[symbol].isotopes[
                max(ELEMENTS[symbol].isotopes, key=lambda k: ELEMENTS[symbol].isotopes[k].abundance)
            ].mass) * count
        return total

    # ------------------------------------------------------------------
    # Composition — molmass-compatible
    # ------------------------------------------------------------------

    def composition(self) -> Composition:
        """Build a molmass.Composition matching Formula.composition() output."""
        nonzero_items: list[tuple[str, int]] = []

        # First pass: compute total mass for fractions
        total_mass = 0.0
        for symbol, count in self._iter_nonzero_items():
            nonzero_items.append((symbol, count))
            total_mass += ELEMENTS[symbol].mass * count

        # Build tuple list: (symbol, count, mass, fraction)
        items: list[tuple[str, int, float, float]] = []
        for symbol, count in nonzero_items:
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
            merged = {sym: cnt for sym, cnt in self._iter_nonzero_items()}
            for sym, cnt in other._iter_nonzero_items():
                merged[sym] = merged.get(sym, 0) + cnt
            return LightFormula(
                elements=merged,
                charge=self._charge + other._charge,
                monoisotopic_mass=self._monoisotopic_mass + other._monoisotopic_mass,
            )

        if isinstance(other, Formula):
            merged = {sym: cnt for sym, cnt in self._iter_nonzero_items()}
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
        nonzero = {}
        for symbol, count in self._iter_nonzero_items():
            nonzero[symbol] = count

        # Hill order: C first, H second, rest alphabetical
        sorted_symbols = sorted(
            nonzero,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )

        for sym in sorted_symbols:
            cnt = nonzero[sym]
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

    def _iter_nonzero_items(self):
        """
        Iterate (symbol, count) pairs for non-zero elements.

        Uses a compact symbols+counts backing when available to avoid
        materializing dictionaries for every candidate.
        """
        if self._elements is not None:
            for symbol, count in self._elements.items():
                if count > 0:
                    yield symbol, count
            return

        # Compact backing path
        symbols = self._symbols
        counts = self._counts
        if symbols is None or counts is None:
            return

        for i in range(len(symbols)):
            count = counts[i]
            if count > 0:
                yield symbols[i], count
