"""Exact full-row duplicates.

Reports how many rows are redundant copies of another row. Column-name
duplicates belong in shape profiling; partial duplicates (ignore column X)
are a judgment call for the checks layer.

Whitespace is stripped and blank/null cells compare equal, but case is kept:
``"Alice"`` and ``"alice"`` are different rows. Treating case as noise is a
judgment call for the checks layer, not this profiler.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqa.profiling._nulls import is_missing


@dataclass
class DuplicateProfile:
    row_count: int
    unique_row_count: int
    duplicate_row_count: int  # extras beyond the first copy of each row
    duplicate_rate: float  # duplicate_row_count / row_count; 0.0 if no rows
    duplicate_group_count: int  # distinct row contents that appear more than once


def profile_duplicates(df: pd.DataFrame) -> DuplicateProfile:
    """Measure exact full-row duplicates."""
    row_count = len(df)
    if row_count == 0:
        return DuplicateProfile(
            row_count=0,
            unique_row_count=0,
            duplicate_row_count=0,
            duplicate_rate=0.0,
            duplicate_group_count=0,
        )

    comparable = _comparable_frame(df)
    # keep="first": True marks every extra copy after the first occurrence
    is_extra = comparable.duplicated(keep="first")
    duplicate_row_count = int(is_extra.sum())
    unique_row_count = row_count - duplicate_row_count

    counts = comparable.value_counts(dropna=False)
    duplicate_group_count = int((counts > 1).sum())

    return DuplicateProfile(
        row_count=row_count,
        unique_row_count=unique_row_count,
        duplicate_row_count=duplicate_row_count,
        duplicate_rate=duplicate_row_count / row_count,
        duplicate_group_count=duplicate_group_count,
    )


def _comparable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cells so blanks and nulls compare equal, then strip text.

    Uses positional column labels so duplicate header names do not collapse
    columns before the comparison.
    """
    columns: dict[int, list[str | None]] = {}
    for position in range(len(df.columns)):
        columns[position] = [
            None if is_missing(value) else str(value).strip()
            for value in df.iloc[:, position]
        ]
    return pd.DataFrame(columns)
