"""Numeric summary statistics for each column.

Only values that parse as numbers (see :mod:`dqa.profiling._numbers`) contribute
to the stats. Standard deviation is the *sample* form (``ddof=1``), the usual
choice when describing an observed dataset rather than a whole population.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqa.profiling._nulls import present_values
from dqa.profiling._numbers import parse_number


@dataclass
class ColumnStatisticsProfile:
    column_name: str
    non_missing_count: int
    numeric_count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q1: float | None
    q3: float | None


@dataclass
class StatisticsProfile:
    columns: list[ColumnStatisticsProfile]

    def by_name(self, name: str) -> ColumnStatisticsProfile:
        """The profile for ``name``, or the first of them if the name repeats."""
        for profile in self.columns:
            if profile.column_name == name:
                return profile
        raise KeyError(name)


def profile_statistics(df: pd.DataFrame) -> StatisticsProfile:
    """Measure numeric summaries for every column."""
    return StatisticsProfile(
        columns=[
            _profile_column(str(column), df.iloc[:, position])
            for position, column in enumerate(df.columns)
        ]
    )


def _profile_column(name: str, series: pd.Series) -> ColumnStatisticsProfile:
    values, _null_count = present_values(series)
    non_missing_count = len(values)
    numbers = [n for value in values if (n := parse_number(value)) is not None]
    numeric_count = len(numbers)

    # Keep every column in the list (aligned with shape/missing/cardinality).
    # None means "no numeric summary is defined", not "forgot to compute".
    if numeric_count == 0:
        return ColumnStatisticsProfile(
            column_name=name,
            non_missing_count=non_missing_count,
            numeric_count=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            std=None,
            q1=None,
            q3=None,
        )

    numeric = pd.Series(numbers, dtype=float)
    # Sample std needs at least two values; with one, pandas returns NaN.
    std = float(numeric.std(ddof=1)) if numeric_count >= 2 else None
    return ColumnStatisticsProfile(
        column_name=name,
        non_missing_count=non_missing_count,
        numeric_count=numeric_count,
        min=float(numeric.min()),
        max=float(numeric.max()),
        mean=float(numeric.mean()),
        median=float(numeric.median()),
        std=std,
        q1=float(numeric.quantile(0.25)),
        q3=float(numeric.quantile(0.75)),
    )
