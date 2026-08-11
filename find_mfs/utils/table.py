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


def _has_field(field: str) -> Callable[[list['FormulaCandidate']], bool]:
    return lambda cs: any(getattr(c, field) is not None for c in cs)


def _fmt_field(field: str) -> Callable[['FormulaCandidate'], str]:
    def _value(c: 'FormulaCandidate') -> str:
        v = getattr(c, field)
        return f"{v:.2f}" if v is not None else ""
    return _value


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
        header='Chem.Prior', width=12, align='>',
        enabled=_has_field('chem_logprior'),
        value=_fmt_field('chem_logprior'),
        df_key='chem_logprior', df_value=lambda c: c.chem_logprior,
    ),
    _ColumnSpec(
        header='Iso.LL', width=10, align='>',
        enabled=_has_field('iso_loglik'),
        value=_fmt_field('iso_loglik'),
        df_key='iso_loglik', df_value=lambda c: c.iso_loglik,
    ),
    _ColumnSpec(
        header='Mass.LL', width=10, align='>',
        enabled=_has_field('mass_loglik'),
        value=_fmt_field('mass_loglik'),
        df_key='mass_loglik', df_value=lambda c: c.mass_loglik,
    ),
    _ColumnSpec(
        header='MS2.LL', width=10, align='>',
        enabled=_has_field('ms2_loglik'),
        value=_fmt_field('ms2_loglik'),
        df_key='ms2_loglik', df_value=lambda c: c.ms2_loglik,
    ),
    _ColumnSpec(
        header='Log.Post', width=12, align='>',
        enabled=_has_field('log_posterior'),
        value=_fmt_field('log_posterior'),
        df_key='log_posterior', df_value=lambda c: c.log_posterior,
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


def render_dataframe(
    candidates: list['FormulaCandidate']
) -> 'pd.DataFrame':
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
