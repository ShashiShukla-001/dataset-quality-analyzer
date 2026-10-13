"""How many distinct values each column has among its present cells.

Reports counts and rates only. Whether a column is "constant" or an ID column
is a quality-check concern, not this profiler's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqa.profiling._nulls import present_values


@dataclass
class ColumnCardinalityProfile:
    column_name: str
    non_missing_count: int
    distinct_count: int
    distinct_rate: float  # distinct_count / non_missing_count


@dataclass
class CardinalityProfile:
    columns: list[ColumnCardinalityProfile]

    def by_name(self, name: str) -> ColumnCardinalityProfile:
        """The profile for ``name``, or the first of them if the name repeats."""
        for profile in self.columns:
            if profile.column_name == name:
                return profile
        raise KeyError(name)


def profile_cardinality(df: pd.DataFrame) -> CardinalityProfile:
    """Measure distinct values per column, excluding missing cells."""
    columns: list[ColumnCardinalityProfile] = []
    for position, column in enumerate(df.columns):
        name = str(column)
        values, _null_count = present_values(df.iloc[:, position])
        non_missing_count = len(values)
        distinct_count = len(set(values))
        if non_missing_count == 0:
            # No present values to divide by; 0.0 means "no distinctness to measure"
            distinct_rate = 0.0
        else:
            distinct_rate = distinct_count / non_missing_count
        columns.append(
            ColumnCardinalityProfile(
                column_name=name,
                non_missing_count=non_missing_count,
                distinct_count=distinct_count,
                distinct_rate=distinct_rate,
            )
        )
    return CardinalityProfile(columns=columns)
