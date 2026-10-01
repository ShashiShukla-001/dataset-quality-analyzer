"""The basic dimensions of a dataset: how many rows, columns, and their names."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass
class ShapeProfile:
    row_count: int
    column_count: int
    column_names: list[str]
    duplicate_column_names: list[str]


def profile_shape(df: pd.DataFrame) -> ShapeProfile:
    """Measure the shape of a dataset."""
    # Headers are not covered by the loader's dtype=str: Excel hands back int
    # and datetime headers, and callers may pass a frame we never loaded.
    names = [str(column) for column in df.columns]
    counts = Counter(names)

    return ShapeProfile(
        row_count=len(df),
        column_count=len(df.columns),
        column_names=names,
        duplicate_column_names=sorted(
            name for name, count in counts.items() if count > 1
        ),
    )
