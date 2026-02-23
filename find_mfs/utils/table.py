"""
Table and DataFrame rendering utilities for FormulaSearchResults.

Columns are defined as _ColumnSpec objects.
To add a new column, append a _ColumnSpec to _COLUMNS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.finder import FormulaCandidate


@dataclass
class _ColumnSpec:
    header: str
    width: int
    align: Literal['<', '>']                      # '<' for left, '>' for right
    enabled: Callable[[list['FormulaCandidate']], bool]
    value: Callable[['FormulaCandidate'], str]    # formatted string for table
    df_key: str                                       # column name in DataFrame
    df_value: Callable[['FormulaCandidate'], object]  # raw value for DataFrame


def _iso_match_str(c: 'FormulaCandidate') -> str:
    if c.isotope_match_result is None:
        return ""
    r = c.isotope_match_result
    return f"{r.num_peaks_matched}/{r.num_peaks_total}"


def _iso_rmse_str(c: 'FormulaCandidate') -> str:
    if c.isotope_match_result is None:
        return ""
    return f"{c.isotope_match_result.intensity_rmse:.4f}"


def _has_isotope(cs: list['FormulaCandidate']) -> bool:
    return any(c.isotope_match_result is not None for c in cs)


def _has_prior(cs: list['FormulaCandidate']) -> bool:
    return any(c.prior_score is not None for c in cs)


_COLUMNS: list[_ColumnSpec] = [
    _ColumnSpec(
        header='Formula', width=25, align='<',
        enabled=lambda cs: True,
        value=lambda c: c.formula.formula,
        df_key='formula', df_value=lambda c: c.formula.formula,
    ),
    _ColumnSpec(
        header='Error (ppm)', width=15, align='>',
        enabled=lambda cs: True,
        value=lambda c: f"{c.error_ppm:.2f}",
        df_key='error_ppm', df_value=lambda c: c.error_ppm,
    ),
    _ColumnSpec(
        header='Error (Da)', width=15, align='>',
        enabled=lambda cs: True,
        value=lambda c: f"{c.error_da:.6f}",
        df_key='error_da', df_value=lambda c: c.error_da,
    ),
    _ColumnSpec(
        header='RDBE', width=10, align='>',
        enabled=lambda cs: True,
        value=lambda c: f"{c.rdbe:.1f}" if c.rdbe is not None else "N/A",
        df_key='rdbe', df_value=lambda c: c.rdbe,
    ),
    _ColumnSpec(
        header='Iso. Matches', width=15, align='>',
        enabled=_has_isotope,
        value=_iso_match_str,
        df_key='isotope_matches',
        df_value=lambda c: _iso_match_str(c) or None,
    ),
    _ColumnSpec(
        header='Iso. RMSE', width=10, align='>',
        enabled=_has_isotope,
        value=_iso_rmse_str,
        df_key='isotope_rmse',
        df_value=lambda c: c.isotope_match_result.intensity_rmse if c.isotope_match_result else None,
    ),
    _ColumnSpec(
        header='Prior', width=10, align='>',
        enabled=_has_prior,
        value=lambda c: f"{c.prior_score:.2f}" if c.prior_score is not None else "",
        df_key='prior_score', df_value=lambda c: c.prior_score,
    ),
]


def render_table(
    candidates: list['FormulaCandidate'],
    max_rows: int | None = None,
    total: int | None = None,
) -> str:
    """
    Render a text table of formula candidates.

    Args:
        candidates: Candidates to display (already sliced if needed).
        max_rows: If given and total > max_rows, appends a "... and N more" line.
        total: Total number of candidates before slicing (for the truncation line).

    Returns:
        Formatted string table.
    """
    if not candidates:
        return "No candidates found."

    active = [col for col in _COLUMNS if col.enabled(candidates)]

    header = " ".join(f"{col.header:{col.align}{col.width}}" for col in active)
    sep = "-" * len(header)

    rows = [
        " ".join(f"{col.value(c):{col.align}{col.width}}" for col in active)
        for c in candidates
    ]

    lines = [header, sep] + rows

    if max_rows is not None and total is not None and total > max_rows:
        lines.append(f"... and {total - max_rows} more")

    return "\n".join(lines)


def render_dataframe(candidates: list['FormulaCandidate']) -> 'pd.DataFrame':
    """
    Render formula candidates as a pandas DataFrame.

    Args:
        candidates: Candidates to include.

    Returns:
        pandas DataFrame with the same columns as the text table.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install with: pip install pandas"
        )

    active = [col for col in _COLUMNS if col.enabled(candidates)]

    data = [
        {col.df_key: col.df_value(c) for col in active}
        for c in candidates
    ]

    return pd.DataFrame(data)
