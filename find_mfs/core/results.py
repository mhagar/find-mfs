"""
This module has the FormulaSearchResults class, which contains
FormulaCandidate objects, and provides convenience methods for:
- filtering,
- display
- export
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, overload, TYPE_CHECKING

import numpy as np

from .finder import FormulaCandidate
from .light_formula import LightFormula
from ..utils.filtering import passes_octet_rule
from ..isotopes.results import SingleEnvelopeMatchResult

if TYPE_CHECKING:
    from ..isotopes import IsotopeMatchResult
    import pandas as pd


class _LazyBackend:
    """
    Stores raw numpy arrays and materializes FormulaCandidate on demand.

    This avoids eagerly constructing N LightFormula + N FormulaCandidate +
    N SingleEnvelopeMatchResult Python objects when the user may only
    inspect a few of them.
    """
    __slots__ = (
        '_counts', '_exact_masses', '_error_ppm', '_error_da',
        '_rdbe', '_iso_rmse', '_iso_match_frac', '_iso_n_matched',
        '_iso_peak_matches',
        '_symbols', '_charge', '_adduct', '_n_obs',
        '_charge_mass_offset', '_adduct_mass',
        '_cache',
    )

    def __init__(
        self,
        raw: dict,
        symbols: list[str],
        charge: int,
        adduct: str | None = None,
        n_obs: int = 0,
        charge_mass_offset: float = 0.0,
        adduct_mass: float = 0.0,
    ):
        self._counts = raw['counts']
        self._exact_masses = raw['exact_masses']
        self._error_ppm = raw['error_ppm']
        self._error_da = raw['error_da']
        self._rdbe = raw.get('rdbe')
        self._iso_rmse = raw.get('iso_rmse')
        self._iso_match_frac = raw.get('iso_match_frac')
        self._iso_n_matched = raw.get('iso_n_matched')
        self._iso_peak_matches = raw.get('iso_peak_matches')
        self._symbols = symbols
        self._charge = charge
        self._adduct = adduct
        self._n_obs = n_obs
        self._charge_mass_offset = charge_mass_offset
        self._adduct_mass = adduct_mass
        self._cache: dict[int, FormulaCandidate] = {}

    def __len__(self) -> int:
        return self._counts.shape[0]

    def _materialize(self, idx: int) -> FormulaCandidate:
        if idx in self._cache:
            return self._cache[idx]

        row_list = self._counts[idx].tolist()

        if self._adduct is not None:
            # Adduct path: core molecule is neutral
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=0,
                monoisotopic_mass=(
                    float(self._exact_masses[idx])
                    + self._charge_mass_offset
                    - self._adduct_mass
                ),
            )
        else:
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=self._charge,
                monoisotopic_mass=float(self._exact_masses[idx]),
            )

        isotope_result = None
        if self._iso_rmse is not None:
            if self._iso_peak_matches is not None:
                peak_matches = self._iso_peak_matches[idx].astype(bool)
            else:
                peak_matches = np.full(self._n_obs, self._iso_n_matched[idx] > 0)
            isotope_result = SingleEnvelopeMatchResult(
                num_peaks_matched=int(self._iso_n_matched[idx]),
                num_peaks_total=self._n_obs,
                intensity_rmse=float(self._iso_rmse[idx]),
                match_fraction=float(self._iso_match_frac[idx]),
                peak_matches=peak_matches,
                predicted_envelope=np.empty((0, 2), dtype=np.float64),
            )

        candidate = FormulaCandidate(
            formula=formula,
            error_ppm=float(self._error_ppm[idx]),
            error_da=float(self._error_da[idx]),
            rdbe=float(self._rdbe[idx]) if self._rdbe is not None else None,
            adduct=self._adduct,
            isotope_match_result=isotope_result,
        )
        self._cache[idx] = candidate
        return candidate

    def _reindex(self, idx) -> '_LazyBackend':
        """Return a new _LazyBackend reindexed by slice, boolean mask, or int array."""
        raw = {
            'counts': self._counts[idx],
            'exact_masses': self._exact_masses[idx],
            'error_ppm': self._error_ppm[idx],
            'error_da': self._error_da[idx],
        }
        if self._rdbe is not None:
            raw['rdbe'] = self._rdbe[idx]
        if self._iso_rmse is not None:
            raw['iso_rmse'] = self._iso_rmse[idx]
            raw['iso_match_frac'] = self._iso_match_frac[idx]
            raw['iso_n_matched'] = self._iso_n_matched[idx]
        if self._iso_peak_matches is not None:
            raw['iso_peak_matches'] = self._iso_peak_matches[idx]
        return _LazyBackend(
            raw=raw,
            symbols=self._symbols,
            charge=self._charge,
            adduct=self._adduct,
            n_obs=self._n_obs,
            charge_mass_offset=self._charge_mass_offset,
            adduct_mass=self._adduct_mass,
        )

    def _slice(self, s: slice) -> '_LazyBackend':
        """Return a new _LazyBackend for a slice of the data."""
        return self._reindex(s)

    def _filter_by_mask(self, mask: np.ndarray) -> '_LazyBackend':
        """Return a new _LazyBackend filtered by boolean mask."""
        return self._reindex(mask)


@dataclass
class FormulaSearchResults:
    """
    Container for formula search results with filtering and display methods

    This class wraps a list of FormulaCandidate objects and provides:
    - Iterator/indexing support for easy access to MF candidates
    - Post-hoc filtering methods that return new FormulaSearchResults
    - Formatted representation in response to `print()`
    - Formatted table output via to_table()
    - Optional pandas DataFrame export

    Attributes:
        candidates: List of formula candidates (may be empty if using lazy backend)
        query_mass: The mass that was searched
        query_params: Dictionary of search parameters used

    Example:
        >>> finder: 'FormulaFinder'
        >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
        >>> print(results)  # Gives a summary
        >>> for candidate in results:  # Iterate
        ...     print(candidate.formula)
        >>> # Post-hoc filter:
        >>> filtered: FormulaSearchResults = results.filter_by_rdbe(0, 10)
    """
    candidates: list[FormulaCandidate]
    query_mass: float
    query_params: dict = field(default_factory=dict)
    _backend: _LazyBackend | None = field(default=None, repr=False)

    def __len__(self) -> int:
        if self._backend is not None:
            return len(self._backend)
        return len(self.candidates)

    def __iter__(self):
        if self._backend is not None:
            return (self._backend._materialize(i) for i in range(len(self._backend)))
        return iter(self.candidates)

    @overload
    def __getitem__(self, idx: int) -> FormulaCandidate: ...

    @overload
    def __getitem__(self, idx: slice) -> 'FormulaSearchResults': ...

    def __getitem__(
        self,
        idx: int | slice,
    ) -> 'FormulaCandidate | FormulaSearchResults':
        if self._backend is not None:
            if isinstance(idx, slice):
                return FormulaSearchResults(
                    candidates=[],
                    query_mass=self.query_mass,
                    query_params=self.query_params,
                    _backend=self._backend._slice(idx),
                )
            if idx < 0:
                idx += len(self._backend)
            return self._backend._materialize(idx)

        if isinstance(idx, slice):
            return FormulaSearchResults(
                candidates=self.candidates[idx],
                query_mass=self.query_mass,
                query_params=self.query_params,
            )
        return self.candidates[idx]

    def __repr__(self) -> str:
        n_results = len(self)
        summary = self._summary_line(n_results)

        if n_results == 0:
            return summary

        lines = [summary, "", self.to_table(max_rows=5)]
        return "\n".join(lines)

    # === FORMATTING METHODS ===
    def _summary_line(self, n_results: int) -> str:
        """Build the header line, including adduct notation when present."""
        adduct = self.query_params.get('adduct')
        charge = self.query_params.get('charge', 0)
        parts = [
            f"query_mass={self.query_mass:.4f}",
            f"n_results={n_results}",
        ]
        if adduct is not None:
            adduct_part = adduct if adduct.startswith('-') else f'+{adduct}'
            sign = '+' if charge > 0 else '-' if charge < 0 else ''
            abs_charge = abs(charge)
            charge_str = f'{abs_charge}{sign}' if abs_charge > 1 else sign
            parts.append(f"adduct=[M{adduct_part}]{charge_str}")
        return f"FormulaSearchResults({', '.join(parts)})"

    def to_table(
        self,
        max_rows: Optional[int] = None
    ) -> str:
        """
        Return formatted table of all candidates

        Args:
            max_rows: Maximum number of rows to display. None shows all.

        Returns:
            Formatted string table
        """
        n = len(self)
        if n == 0:
            return "No candidates found."

        show_n = n if max_rows is None else min(n, max_rows)

        # Materialize only the rows we need to display
        candidates_to_show = [self[i] for i in range(show_n)]

        # Check if any candidates have isotope/fragment matching results
        has_isotope_results = any(
            c.isotope_match_result is not None for c in candidates_to_show
        )
        # Build header dynamically
        header = f"{'Formula':<25} {'Error (ppm)':<15} {'Error (Da)':<15} {'RDBE':<10}"
        sep_len = 70

        if has_isotope_results:
            header += f" {'Iso. Matches':<15}"
            header += f"{'Iso. RMSE':<10}"
            sep_len += 26

        lines: list[str] = [header, "-" * sep_len]

        # Build rows
        for candidate in candidates_to_show:
            formula_str = candidate.formula.formula
            rdbe_str = f"{candidate.rdbe:.1f}" if candidate.rdbe is not None else "N/A"

            iso_match_str = ""
            iso_score_str = ""
            if candidate.isotope_match_result is not None:
                iso_match_str = (f"{candidate.isotope_match_result.num_peaks_matched}"
                                 f"/{candidate.isotope_match_result.num_peaks_total}")
                iso_score_str = f"{candidate.isotope_match_result.intensity_rmse:.4f}"

            if has_isotope_results:
                lines.append(
                    f"{formula_str:<25} {candidate.error_ppm:>14.2f} "
                    f"{candidate.error_da:>14.6f} {rdbe_str:>9} {iso_match_str:>13} {iso_score_str:>9}"
                )
            else:
                lines.append(
                    f"{formula_str:<25} {candidate.error_ppm:>14.2f} "
                    f"{candidate.error_da:>14.6f} {rdbe_str:>9}"
                )

        if max_rows is not None and n > max_rows:
            lines.append(f"... and {n - max_rows} more")

        return "\n".join(lines)

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert results to pandas DataFrame, if pandas is installed.

        The DataFrame will include isotope matching scores if available,
        matching the columns shown in to_table() and __repr__().

        Returns:
            pandas.DataFrame with columns for formula, errors, RDBE, and
            isotope scores (if isotope matching was performed)

        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        # Fast path: read directly from backend arrays
        if self._backend is not None:
            b = self._backend
            n = len(b)
            data = {
                'formula': [
                    b._materialize(i).formula.formula for i in range(n)
                ],
                'error_ppm': b._error_ppm.tolist(),
                'error_da': b._error_da.tolist(),
                'rdbe': b._rdbe.tolist() if b._rdbe is not None else [None] * n,
                'mass': b._exact_masses.tolist(),
            }
            if b._iso_rmse is not None:
                data['isotope_intensity_rmse'] = b._iso_rmse.tolist()
                data['isotope_match_fraction'] = b._iso_match_frac.tolist()
            return pd.DataFrame(data)

        data = []
        for candidate in self.candidates:
            row = {
                'formula': candidate.formula.formula,
                'error_ppm': candidate.error_ppm,
                'error_da': candidate.error_da,
                'rdbe': candidate.rdbe,
                'mass': candidate.formula.monoisotopic_mass,
            }

            if candidate.isotope_match_result is not None:
                if isinstance(candidate.isotope_match_result, SingleEnvelopeMatchResult):
                    row['isotope_intensity_rmse'] = candidate.isotope_match_result.intensity_rmse
                    row['isotope_match_fraction'] = candidate.isotope_match_result.match_fraction

            data.append(row)

        return pd.DataFrame(data)

    # === SORTING METHODS ===
    def sort_by_error(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by absolute mass error (Da).

        Args:
            reverse: If True, sort in descending order (largest error first)

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            b = self._backend
            order = np.argsort(np.abs(b._error_da))
            if reverse:
                order = order[::-1]
            new_backend = b._reindex(order)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params=self.query_params, _backend=new_backend,
            )

        return FormulaSearchResults(
            candidates=sorted(self.candidates, reverse=reverse),
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def sort_by_rmse(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by isotope intensity RMSE.

        Candidates without isotope match results are placed at the end.

        Args:
            reverse: If True, sort in descending order (largest RMSE first)

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            b = self._backend
            if b._iso_rmse is None:
                # No isotope data — sorting by RMSE is a no-op
                return self
            order = np.argsort(b._iso_rmse)
            if reverse:
                order = order[::-1]
            new_backend = b._reindex(order)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params=self.query_params, _backend=new_backend,
            )

        with_iso = [c for c in self.candidates if c.isotope_match_result is not None]
        without_iso = [c for c in self.candidates if c.isotope_match_result is None]

        sorted_with = sorted(
            with_iso,
            key=lambda x: x.isotope_match_result.intensity_rmse,
            reverse=reverse,
        )

        return FormulaSearchResults(
            candidates=sorted_with + without_iso,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    # === FILTERING METHODS ===
    def filter_by_rdbe(
        self,
        min_rdbe: float,
        max_rdbe: float
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by RDBE range

        Args:
            min_rdbe: Minimum RDBE value (inclusive)
            max_rdbe: Maximum RDBE value (inclusive)

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        if self._backend is not None:
            b = self._backend
            if b._rdbe is None:
                # No RDBE data — cannot filter, return empty
                return FormulaSearchResults(
                    candidates=[], query_mass=self.query_mass,
                    query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
                )
            mask = (b._rdbe >= min_rdbe) & (b._rdbe <= max_rdbe)
            new_backend = b._filter_by_mask(mask)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
                _backend=new_backend,
            )

        filtered = [
            c for c in self.candidates
            if c.rdbe is not None and min_rdbe <= c.rdbe <= max_rdbe
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'filter_rdbe': (min_rdbe, max_rdbe),
            }
        )

    def filter_by_octet(self) -> 'FormulaSearchResults':
        """
        Filter candidates to only those passing the octet rule.

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        # Octet filtering requires materializing formulas
        filtered = [
            c for c in self
            if passes_octet_rule(c.formula)
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'check_octet': True,
            }
        )

    def filter_by_error(
        self,
        max_ppm: Optional[float] = None,
        max_da: Optional[float] = None
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by maximum error.

        At least one of max_ppm or max_da must be specified.

        Args:
            max_ppm: Maximum absolute error in ppm
            max_da: Maximum absolute error in Da

        Returns:
            New FormulaSearchResults with filtered candidates

        Raises:
            ValueError: If neither max_ppm nor max_da is specified
        """
        if max_ppm is None and max_da is None:
            raise ValueError(
                "At least one of max_ppm or max_da must be specified"
            )

        if self._backend is not None:
            b = self._backend
            mask = np.ones(len(b), dtype=bool)
            if max_ppm is not None:
                mask &= np.abs(b._error_ppm) <= max_ppm
            if max_da is not None:
                mask &= np.abs(b._error_da) <= max_da
            new_backend = b._filter_by_mask(mask)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'max_error_ppm': max_ppm, 'max_error_da': max_da,
                },
                _backend=new_backend,
            )

        filtered = []
        for c in self.candidates:
            passes = True
            if max_ppm is not None and abs(c.error_ppm) > max_ppm:
                passes = False
            if max_da is not None and abs(c.error_da) > max_da:
                passes = False
            if passes:
                filtered.append(c)

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'max_error_ppm': max_ppm,
                'max_error_da': max_da
            }
        )

    def filter_by_isotope_quality(
        self,
        max_match_rmse: Optional[float] = 1.0,
        min_match_fraction: Optional[float] = 0.0,
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by isotope match quality.

        Uses isotope matching results to filter candidate formulae.

        Args:
            max_match_rmse: Maximum isotope envelope RMSE.
                Example: 0.05 means the total error in isotope envelope can't
                exceed 5%.
                Default: 1.0 (total error can't exceed 100%)

            min_match_fraction: Minimum fraction of peaks matched (0.0-1.0)
                Example: 0.8 means at least 80% of peaks must match
                Default: 0.0 (no filter)

        Returns:
            New FormulaSearchResults with filtered candidates

        Raises:
            ValueError: If neither parameter is specified or if candidates
                don't have isotope match results
        """
        if self._backend is not None and self._backend._iso_rmse is not None:
            b = self._backend
            mask = (b._iso_rmse <= max_match_rmse) & (b._iso_match_frac >= min_match_fraction)
            new_backend = b._filter_by_mask(mask)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'min_match_fraction': min_match_fraction,
                    'max_match_rmse': max_match_rmse,
                },
                _backend=new_backend,
            )

        filtered = []
        for c in self.candidates:
            if c.isotope_match_result is None:
                continue
            if c.isotope_match_result.match_fraction < min_match_fraction:
                continue
            if c.isotope_match_result.intensity_rmse > max_match_rmse:
                continue
            filtered.append(c)

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'min_match_fraction': min_match_fraction,
                'max_match_rmse': max_match_rmse,
            }
        )

    def get_isotope_details(
        self,
        index: int
    ) -> 'IsotopeMatchResult | None':
        """
        Get detailed isotope matching information for a specific MF candidate.

        Args:
            index: Index of the candidate to inspect

        Returns:
            IsotopeMatchResult (SingleEnvelopeMatchResult)
            with detailed per-peak information, or None if no isotope matching
            was performed for this candidate

        Example:
            >>> finder: 'FormulaFinder'
            >>> results = finder.find_formulae(...)
            >>> details = results.get_isotope_details(0)
            >>> if details:
            ...     print(f"Matched {details.num_peaks_matched}/{details.num_peaks_total}")
            ...     print(f"Per-peak: {details.peak_matches}")
        """
        n = len(self)
        if index < 0 or index >= n:
            raise IndexError(
                f"Index {index} out of range for {n} candidates"
            )

        return self[index].isotope_match_result

    def top(
        self,
        n: int = 10,
    ) -> 'FormulaSearchResults':
        """
        Return top N candidates by error.

        Args:
            n: Number of top candidates to return

        Returns:
            New FormulaSearchResults with top N candidates
        """
        return self[:n]
