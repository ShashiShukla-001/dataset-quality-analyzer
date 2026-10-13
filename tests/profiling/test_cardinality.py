import pandas as pd
import pytest

from dqa.profiling import profile_cardinality, profile_missing


def test_normal_column_distinct_count_and_rate():
    df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})

    profile = profile_cardinality(df).by_name("color")

    assert profile.non_missing_count == 4
    assert profile.distinct_count == 3
    assert profile.distinct_rate == pytest.approx(0.75)


def test_case_is_kept():
    df = pd.DataFrame({"flag": ["Yes", "yes"]})

    profile = profile_cardinality(df).by_name("flag")

    assert profile.distinct_count == 2


def test_whitespace_is_stripped_before_counting():
    df = pd.DataFrame({"x": ["a", " a ", "a"]})

    profile = profile_cardinality(df).by_name("x")

    assert profile.distinct_count == 1
    assert profile.non_missing_count == 3


def test_missing_cells_are_excluded():
    df = pd.DataFrame({"note": ["a", "", "  ", None, "b"]})

    profile = profile_cardinality(df).by_name("note")

    assert profile.non_missing_count == 2
    assert profile.distinct_count == 2


def test_all_missing_column():
    df = pd.DataFrame({"empty": ["", None, "  "]})

    profile = profile_cardinality(df).by_name("empty")

    assert profile.non_missing_count == 0
    assert profile.distinct_count == 0
    assert profile.distinct_rate == 0.0


def test_duplicate_column_names_are_both_kept():
    df = pd.DataFrame([["a", "x"], ["a", "y"]], columns=["a", "a"])

    profile = profile_cardinality(df)

    assert len(profile.columns) == 2
    assert [c.distinct_count for c in profile.columns] == [1, 2]
    assert profile.by_name("a").distinct_count == 1


def test_zero_row_frame():
    df = pd.DataFrame({"a": [], "b": []})

    profile = profile_cardinality(df)

    assert len(profile.columns) == 2
    assert profile.by_name("a").non_missing_count == 0
    assert profile.by_name("a").distinct_count == 0
    assert profile.by_name("a").distinct_rate == 0.0


def test_non_missing_count_agrees_with_missing_profile():
    df = pd.DataFrame(
        {
            "age": ["30", "", None, "25"],
            "name": ["Alice", "  ", "Bob", "Carol"],
        }
    )

    cardinality = profile_cardinality(df)
    missing = profile_missing(df)
    row_count = len(df)

    for column in cardinality.columns:
        assert column.non_missing_count == row_count - missing.by_name(
            column.column_name
        ).null_count


def test_by_name_rejects_unknown_column():
    profile = profile_cardinality(pd.DataFrame({"age": ["30"]}))

    with pytest.raises(KeyError):
        profile.by_name("missing")
