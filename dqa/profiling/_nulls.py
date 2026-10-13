"""Shared definition of "missing" for every profiler that needs it.

A cell is missing when it is a real null (``pd.isna``) or holds only
whitespace after stripping. Placeholder words like ``"NA"`` or ``"unknown"``
are left alone — deciding those mean missing is a quality-check concern.
"""

from __future__ import annotations

import pandas as pd


def is_missing(value: object) -> bool:
    """True when ``value`` should count as absent for profiling."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def present_values(series: pd.Series) -> tuple[list[str], int]:
    """Split a column into stripped present values and a missing count."""
    values: list[str] = []
    null_count = 0
    for value in series:
        if is_missing(value):
            null_count += 1
        else:
            values.append(str(value).strip())
    return values, null_count


def missing_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of missing cells, aligned to ``series.index``."""
    return series.map(is_missing)
