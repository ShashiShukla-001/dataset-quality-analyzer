"""How much of the dataset is missing, by column and overall.

Reports facts only: counts and rates. Whether a rate is "too high" is a
quality-check concern, not this profiler's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqa.profiling._nulls import missing_mask


@dataclass
class ColumnMissingProfile:
    column_name: str
    null_count: int
    null_rate: float  # null_count / row_count; 0.0 when the frame has no rows


@dataclass
class MissingProfile:
    """Per-column missingness plus dataset-wide totals.

    Columns are a list so duplicate names are kept. Ranking "worst columns"
    is presentation — sort by ``null_rate`` in the report layer.
    """

    columns: list[ColumnMissingProfile]
    row_count: int
    column_count: int
    total_cells: int
    total_null_cells: int
    overall_null_rate: float

    def by_name(self, name: str) -> ColumnMissingProfile:
        """The profile for ``name``, or the first of them if the name repeats."""
        for profile in self.columns:
            if profile.column_name == name:
                return profile
        raise KeyError(name)


def profile_missing(df: pd.DataFrame) -> MissingProfile:
    """Measure missing values for every column and for the dataset as a whole."""
    row_count = len(df)
    column_count = len(df.columns)
    total_cells = row_count * column_count

    columns: list[ColumnMissingProfile] = []
    total_null_cells = 0

    for position, column in enumerate(df.columns):
        name = str(column)
        null_count = int(missing_mask(df.iloc[:, position]).sum())
        total_null_cells += null_count
        null_rate = (null_count / row_count) if row_count else 0.0
        columns.append(
            ColumnMissingProfile(
                column_name=name,
                null_count=null_count,
                null_rate=null_rate,
            )
        )

    overall_null_rate = (total_null_cells / total_cells) if total_cells else 0.0
    return MissingProfile(
        columns=columns,
        row_count=row_count,
        column_count=column_count,
        total_cells=total_cells,
        total_null_cells=total_null_cells,
        overall_null_rate=overall_null_rate,
    )
