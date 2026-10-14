"""How often each present value appears in one named column.

A plain frequency fact for class-imbalance and leakage checks. Missing cells
follow the shared ``_nulls`` rules. The returned list is capped so a column
with huge cardinality cannot blow up a report; full ``distinct_count`` is
still reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from dqa.exceptions import ColumnNotFoundError
from dqa.profiling._nulls import present_values

DEFAULT_MAX_VALUES = 100


@dataclass
class ValueCount:
    value: str
    count: int


@dataclass
class ValueCountsProfile:
    column_name: str
    column_position: int
    non_missing_count: int
    distinct_count: int
    values: list[ValueCount]  # most common first; may be truncated
    truncated: bool


def profile_value_counts(
    df: pd.DataFrame,
    column: str,
    *,
    max_values: int = DEFAULT_MAX_VALUES,
) -> ValueCountsProfile:
    """Count present values in ``column``, sorted most → least common.

    If the name appears more than once, the first matching column is used
    (same rule as ``by_name`` elsewhere). Raises ``ColumnNotFoundError`` when
    the name is absent.
    """
    position = _first_position(df, column)
    values, _null_count = present_values(df.iloc[:, position])
    counts = Counter(values)
    distinct_count = len(counts)
    ranked = counts.most_common()
    truncated = distinct_count > max_values
    kept = ranked[:max_values]

    return ValueCountsProfile(
        column_name=column,
        column_position=position,
        non_missing_count=len(values),
        distinct_count=distinct_count,
        values=[ValueCount(value=value, count=count) for value, count in kept],
        truncated=truncated,
    )


def _first_position(df: pd.DataFrame, column: str) -> int:
    for position, name in enumerate(df.columns):
        if str(name) == column:
            return position
    available = [str(name) for name in df.columns]
    raise ColumnNotFoundError(column, available=available)
