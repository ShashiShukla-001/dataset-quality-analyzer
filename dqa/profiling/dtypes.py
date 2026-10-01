"""What type each column looks like, and how cleanly it fits that type.

Inference guesses, it does not judge: every column gets a best-guess type plus
the fraction of its non-null values that actually match. A column of integers
with a few stray words is reported as ``integer`` at a match rate below 1.0,
leaving the quality-check layer to decide whether that is bad enough to flag.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

# int() and float() are too permissive to use as tests: int("1_000") is 1000,
# int() accepts non-ASCII digits, and float() accepts "nan" and "inf".
_INTEGER = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")

# A date has to look like one before pandas is allowed near it: two digit
# groups joined by - or /, or a month name next to a number. Dots are not
# accepted as separators because "1.2.3" is far more often a version string.
_DATE_SHAPED = re.compile(r"(?<![0-9])[0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4}(?![0-9])")
_MONTH_NAME = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.IGNORECASE
)
_HAS_DIGIT = re.compile(r"[0-9]")

# Whole words only. "0"/"1" are indistinguishable from integers, and single
# letters collide with real data: T/A/G/C are DNA bases, not booleans.
_BOOLEANS = frozenset({"true", "false", "yes", "no"})


@dataclass
class ColumnTypeProfile:
    column_name: str
    inferred_type: str  # integer | float | boolean | date | text | empty
    match_rate: float  # share of non-null values matching inferred_type
    null_count: int


@dataclass
class DTypeProfile:
    """One profile per column, in the order the columns appear.

    A list rather than a dict keyed by name: column names are not unique, and
    a dict would quietly drop every column whose name is already taken.
    """

    columns: list[ColumnTypeProfile]

    def by_name(self, name: str) -> ColumnTypeProfile:
        """The profile for `name`, or the first of them if the name repeats."""
        for profile in self.columns:
            if profile.column_name == name:
                return profile
        raise KeyError(name)


def profile_dtypes(df: pd.DataFrame) -> DTypeProfile:
    """Infer a type for every column."""
    return DTypeProfile(
        columns=[
            # positional: df[name] returns a DataFrame when names repeat
            _profile_column(str(column), df.iloc[:, position])
            for position, column in enumerate(df.columns)
        ]
    )


def _profile_column(name: str, series: pd.Series) -> ColumnTypeProfile:
    values, null_count = _present_values(series)
    if not values:
        return ColumnTypeProfile(name, "empty", 1.0, null_count)

    # insertion order breaks ties, so "1" is an integer rather than a float
    rates = {
        "integer": _rate(values, _is_integer),
        "float": _rate(values, _is_float),
        "boolean": _rate(values, _is_boolean),
        "date": _date_rate(values),
    }
    best = max(rates, key=rates.__getitem__)
    if rates[best] == 0.0:
        return ColumnTypeProfile(name, "text", 1.0, null_count)
    return ColumnTypeProfile(name, best, rates[best], null_count)


def _present_values(series: pd.Series) -> tuple[list[str], int]:
    """Split a column into its real values and a count of the missing ones.

    A cell holding only whitespace is missing in every sense that matters, so
    it is counted as null rather than as the text "". Placeholder words like
    "NA" or "unknown" are left alone: deciding those mean missing is a
    judgement about the data, not a fact about it.
    """
    values = []
    null_count = 0
    for value in series:
        text = "" if pd.isna(value) else str(value).strip()
        if text:
            values.append(text)
        else:
            null_count += 1
    return values, null_count


def _rate(values: list[str], matches: Callable[[str], bool]) -> float:
    return sum(1 for value in values if matches(value)) / len(values)


def _is_integer(value: str) -> bool:
    return _INTEGER.fullmatch(value) is not None


def _is_float(value: str) -> bool:
    return _FLOAT.fullmatch(value) is not None


def _is_boolean(value: str) -> bool:
    return value.lower() in _BOOLEANS


def _is_date_shaped(value: str) -> bool:
    if _DATE_SHAPED.search(value):
        return True
    # "March" alone parses as year 1, so a month name needs a number with it
    return bool(_MONTH_NAME.search(value) and _HAS_DIGIT.search(value))


def _date_rate(values: list[str]) -> float:
    # Screening first keeps pandas away from values it would misread ("2024"
    # as a date, "1,234" as a date) and keeps the slow per-value dateutil
    # fallback off columns that are plainly not dates at all.
    candidates = [value for value in values if _is_date_shaped(value)]
    if not candidates:
        return 0.0

    # format="mixed" parses each value on its own; without it pandas infers one
    # format from the first value, making the match rate depend on row order.
    # utc=True is required too: mixed offsets raise even under errors="coerce".
    try:
        parsed = pd.to_datetime(
            pd.Series(candidates), errors="coerce", format="mixed", utc=True
        )
    except (ValueError, TypeError):
        return 0.0
    return int(parsed.notna().sum()) / len(values)
