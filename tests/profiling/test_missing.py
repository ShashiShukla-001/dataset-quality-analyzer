import pandas as pd
import pytest

from dqa.profiling import profile_dtypes, profile_missing


def test_no_missing_values():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": ["30", "25"]})

    profile = profile_missing(df)

    assert profile.row_count == 2
    assert profile.column_count == 2
    assert profile.total_cells == 4
    assert profile.total_null_cells == 0
    assert profile.overall_null_rate == 0.0
    assert profile.by_name("name").null_count == 0
    assert profile.by_name("age").null_rate == 0.0


def test_per_column_counts_and_rates():
    df = pd.DataFrame(
        {
            "a": ["1", None, "3", "4"],
            "b": ["x", "y", None, None],
        }
    )

    profile = profile_missing(df)

    assert profile.by_name("a") == profile.columns[0]
    assert profile.columns[0].null_count == 1
    assert profile.columns[0].null_rate == 0.25
    assert profile.columns[1].null_count == 2
    assert profile.columns[1].null_rate == 0.5
    assert profile.total_null_cells == 3
    assert profile.overall_null_rate == pytest.approx(3 / 8)


def test_whitespace_only_cells_count_as_missing():
    df = pd.DataFrame({"note": ["ok", "", "  ", None]})

    profile = profile_missing(df).by_name("note")

    assert profile.null_count == 3
    assert profile.null_rate == 0.75


def test_all_blank_column_has_rate_one():
    df = pd.DataFrame({"empty": ["", "  ", None]})

    profile = profile_missing(df).by_name("empty")

    assert profile.null_count == 3
    assert profile.null_rate == 1.0


def test_placeholder_words_are_not_treated_as_missing():
    df = pd.DataFrame({"status": ["NA", "N/A", "unknown"]})

    profile = profile_missing(df).by_name("status")

    assert profile.null_count == 0
    assert profile.null_rate == 0.0


def test_null_counts_agree_with_dtype_profile():
    df = pd.DataFrame(
        {
            "age": ["30", "", None, "25"],
            "name": ["Alice", "  ", "Bob", "Carol"],
        }
    )

    missing = profile_missing(df)
    dtypes = profile_dtypes(df)

    for column in missing.columns:
        assert column.null_count == dtypes.by_name(column.column_name).null_count


def test_duplicate_column_names_are_all_kept():
    df = pd.DataFrame([["1", None]], columns=["a", "a"])

    profile = profile_missing(df)

    assert len(profile.columns) == 2
    assert [c.null_count for c in profile.columns] == [0, 1]
    assert profile.by_name("a").null_count == 0  # the first one


def test_zero_row_frame_has_zero_rates():
    df = pd.DataFrame({"a": [], "b": []})

    profile = profile_missing(df)

    assert profile.row_count == 0
    assert profile.total_cells == 0
    assert profile.total_null_cells == 0
    assert profile.overall_null_rate == 0.0
    assert profile.by_name("a").null_rate == 0.0


def test_by_name_rejects_a_column_that_is_not_there():
    profile = profile_missing(pd.DataFrame({"age": ["30"]}))

    with pytest.raises(KeyError):
        profile.by_name("missing")
