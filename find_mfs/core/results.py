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

from .finder import FormulaCandidate
from ..utils.filtering import passes_octet_rule
from ..isotopes.results import SingleEnvelopeMatchResult

if TYPE_CHECKING:
    from ..isotopes import IsotopeMatchResult
    import pandas as pd


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
        candidates: List of formula candidates
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

    def __len__(self) -> int:
        """
        Return number of MF candidates
        """
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    @overload
    def __getitem__(self, idx: int) -> FormulaCandidate: ...

    @overload
    def __getitem__(self, idx: slice) -> 'FormulaSearchResults': ...

    def __getitem__(
        self,
        idx: int | slice,
    ) -> 'FormulaCandidate | FormulaSearchResults':
        """
        Either returns a formula candidate, or a new
        FormulaSearchResults instance if given a slice

        Args:
            idx: index or slice of list

        Returns:
            Either a FormulaCandidate object, or
            another FormulaSearchResults instance
        """
        if isinstance(idx, slice):
            return FormulaSearchResults(
                candidates=self.candidates[idx],
                query_mass=self.query_mass,
                query_params=self.query_params,
            )

        return self.candidates[idx]

    def __repr__(self) -> str:
        """
        Text summary and top candidates
        """
        n_results = len(self.candidates)
        summary = self._summary_line(n_results)

        if n_results == 0:
            return summary

        # Show top 5 candidates
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
        if len(self.candidates) == 0:
            return "No candidates found."

        candidates_to_show = self.candidates
        if max_rows is not None:
            candidates_to_show = self.candidates[:max_rows]

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

            # Format isotope score if available
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

        if max_rows is not None and len(self.candidates) > max_rows:
            lines.append(f"... and {len(self.candidates) - max_rows} more")

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

        data = []
        for candidate in self.candidates:
            row = {
                'formula': candidate.formula.formula,
                'error_ppm': candidate.error_ppm,
                'error_da': candidate.error_da,
                'rdbe': candidate.rdbe,
                'mass': candidate.formula.monoisotopic_mass,
            }

            # Add isotope matching score if available
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
        filtered = [
            c for c in self.candidates
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

        filtered = []
        for c in self.candidates:
            if c.isotope_match_result is None:
                # Skip candidates without isotope results
                continue

            if c.isotope_match_result.match_fraction < min_match_fraction:
                # Skip candidates with lower match fraction than threshold
                continue

            if c.isotope_match_result.intensity_rmse > max_match_rmse:
                # Skip candidates with larger RSME than threshold
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

        if index < 0 or index >= len(self.candidates):
            raise IndexError(
                f"Index {index} out of range for {len(self.candidates)} candidates"
            )

        return self.candidates[index].isotope_match_result

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
        return FormulaSearchResults(
            candidates=self.candidates[:n],
            query_mass=self.query_mass,
            query_params=self.query_params
        )
