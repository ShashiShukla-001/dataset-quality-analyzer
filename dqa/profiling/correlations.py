"""Pairwise Pearson correlations between numeric columns.

For each pair, only rows where both values are numeric contribute, and
``pair_count`` records how many those were. Whether a correlation is "too
high" (leakage, redundancy) is a quality-check concern, not this profiler's.

The number of pairs grows with the square of the numeric columns — a known
limit; there is no cap yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dqa.profiling._nulls import is_missing
from dqa.profiling._numbers import parse_number


@dataclass
class CorrelationPair:
    column_a: str
    column_b: str
    position_a: int
    position_b: int
    pair_count: int
    pearson: float | None


@dataclass
class CorrelationProfile:
    pairs: list[CorrelationPair]


def profile_correlations(df: pd.DataFrame) -> CorrelationProfile:
    """Measure Pearson correlation for every pair of numeric columns."""
    names: dict[int, str] = {}
    columns: dict[int, pd.Series] = {}

    for position, column in enumerate(df.columns):
        numeric = _numeric_column(df.iloc[:, position])
        # columns with no numeric values never appear in a pair
        if int(numeric.notna().sum()) == 0:
            continue
        names[position] = str(column)
        columns[position] = numeric

    if len(columns) < 2:
        return CorrelationProfile(pairs=[])

    frame = pd.DataFrame(columns)
    # min_periods=2: one shared row cannot define a correlation
    matrix = frame.corr(method="pearson", min_periods=2)
    positions = list(frame.columns)

    pairs: list[CorrelationPair] = []
    for i, position_a in enumerate(positions):
        for position_b in positions[i + 1 :]:
            shared = frame[position_a].notna() & frame[position_b].notna()
            pair_count = int(shared.sum())
            raw = matrix.loc[position_a, position_b]
            # 0.0 would claim "no linear relationship"; NaN means undefined
            # (constant column, or fewer than 2 shared numeric rows).
            pearson = None if pd.isna(raw) else float(raw)
            pairs.append(
                CorrelationPair(
                    column_a=names[position_a],
                    column_b=names[position_b],
                    position_a=position_a,
                    position_b=position_b,
                    pair_count=pair_count,
                    pearson=pearson,
                )
            )

    return CorrelationProfile(pairs=pairs)


def _numeric_column(series: pd.Series) -> pd.Series:
    """Align floats to the series index; non-numeric / missing become NaN."""
    values: list[float] = []
    for value in series:
        if is_missing(value):
            values.append(float("nan"))
            continue
        parsed = parse_number(str(value).strip())
        values.append(float("nan") if parsed is None else parsed)
    return pd.Series(values, index=series.index, dtype=float)
