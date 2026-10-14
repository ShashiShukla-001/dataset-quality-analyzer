import pandas as pd
import pytest

from dqa import ColumnNotFoundError
from dqa.profiling import profile_value_counts



def test_counts_sorted_most_to_least():
    df = pd.DataFrame({"label": ["a", "b", "a", "a", "b", "c"]})

    profile = profile_value_counts(df, "label")

    assert profile.non_missing_count == 6
    assert profile.distinct_count == 3
    assert [(v.value, v.count) for v in profile.values] == [
        ("a", 3),
        ("b", 2),
        ("c", 1),
    ]
    assert profile.truncated is False
    assert profile.column_position == 0


def test_missing_cells_excluded():
    df = pd.DataFrame({"label": ["a", "", "  ", None, "a", "b"]})

    profile = profile_value_counts(df, "label")

    assert profile.non_missing_count == 3
    assert [(v.value, v.count) for v in profile.values] == [("a", 2), ("b", 1)]


def test_case_is_kept():
    df = pd.DataFrame({"label": ["Yes", "yes", "Yes"]})

    profile = profile_value_counts(df, "label")

    assert profile.distinct_count == 2
    assert profile.values[0].value == "Yes"
    assert profile.values[0].count == 2


def test_unknown_column_raises():
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})

    with pytest.raises(ColumnNotFoundError, match="missing") as exc_info:
        profile_value_counts(df, "missing")

    assert exc_info.value.column == "missing"
    assert exc_info.value.available == ["a", "b"]


def test_duplicate_names_use_first_match():
    df = pd.DataFrame([["x", "y"], ["x", "z"]], columns=["a", "a"])

    profile = profile_value_counts(df, "a")

    assert profile.column_position == 0
    assert profile.distinct_count == 1
    assert [(v.value, v.count) for v in profile.values] == [("x", 2)]


def test_list_is_capped_but_distinct_count_is_full():
    df = pd.DataFrame({"id": [str(i) for i in range(150)]})

    profile = profile_value_counts(df, "id", max_values=10)

    assert profile.distinct_count == 150
    assert len(profile.values) == 10
    assert profile.truncated is True


def test_default_max_values_caps_at_100():
    df = pd.DataFrame({"id": [str(i) for i in range(150)]})

    profile = profile_value_counts(df, "id")  # no max_values → default 100

    assert profile.distinct_count == 150
    assert len(profile.values) == 100
    assert profile.truncated is True


def test_empty_present_values():
    df = pd.DataFrame({"label": ["", None, "  "]})

    profile = profile_value_counts(df, "label")

    assert profile.non_missing_count == 0
    assert profile.distinct_count == 0
    assert profile.values == []
    assert profile.truncated is False

def test_exactly_max_values_is_not_truncated():
    df = pd.DataFrame({"id": [str(i) for i in range(10)]})
    profile = profile_value_counts(df, "id", max_values=10)
    assert len(profile.values) == 10
    assert profile.truncated is False
