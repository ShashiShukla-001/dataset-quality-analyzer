import pandas as pd
import pytest

from dqa.profiling import profile_cardinality, profile_statistics


def test_clean_integer_column_hand_calculated():
    df = pd.DataFrame({"n": ["1", "2", "3", "4", "5"]})

    profile = profile_statistics(df).by_name("n")

    assert profile.non_missing_count == 5
    assert profile.numeric_count == 5
    assert profile.min == 1.0
    assert profile.max == 5.0
    assert profile.mean == 3.0
    assert profile.median == 3.0
    assert profile.q1 == 2.0
    assert profile.q3 == 4.0
    assert profile.std == pytest.approx(1.5811388300841898)


def test_stray_words_lower_numeric_count():
    df = pd.DataFrame({"n": ["1", "2", "oops", "4"]})

    profile = profile_statistics(df).by_name("n")

    assert profile.non_missing_count == 4
    assert profile.numeric_count == 3
    assert profile.mean == pytest.approx(7 / 3)


def test_overflow_to_inf_is_not_a_number():
    # float("1e999") is inf; that must not poison mean/std
    profile = profile_statistics(pd.DataFrame({"n": ["1e999", "1"]})).by_name("n")

    assert profile.numeric_count == 1
    assert profile.mean == 1.0
    assert profile.std is None


def test_aggregate_overflow_is_undefined_not_infinite():
    # Each value is finite, but mean/std of five 1e308 values overflow.
    # Median of identical values is the value itself (no interpolate), so it stays.
    profile = profile_statistics(
        pd.DataFrame({"n": ["1e308", "1e308", "1e308", "1e308", "1e308"]})
    ).by_name("n")

    assert profile.numeric_count == 5
    assert profile.min == 1e308
    assert profile.max == 1e308
    assert profile.median == 1e308
    assert profile.mean is None
    assert profile.std is None


def test_whitespace_and_null_excluded():
    df = pd.DataFrame({"n": ["10", "", "  ", None, "20"]})

    profile = profile_statistics(df).by_name("n")

    assert profile.non_missing_count == 2
    assert profile.numeric_count == 2
    assert profile.mean == 15.0


def test_all_text_column_keeps_slot_with_none_stats():
    # Include every column so the list stays aligned with the DataFrame;
    # None means no numeric summary, not that the column was forgotten.
    df = pd.DataFrame({"name": ["Alice", "Bob"]})

    profile = profile_statistics(df).by_name("name")

    assert profile.non_missing_count == 2
    assert profile.numeric_count == 0
    assert profile.min is None
    assert profile.max is None
    assert profile.mean is None
    assert profile.median is None
    assert profile.std is None
    assert profile.q1 is None
    assert profile.q3 is None


def test_single_value_column_has_no_std():
    # Sample std is undefined with one observation; leave it None, not 0.0.
    df = pd.DataFrame({"n": ["7", "7", "7"]})

    profile = profile_statistics(df).by_name("n")

    assert profile.numeric_count == 3
    assert profile.min == 7.0
    assert profile.max == 7.0
    assert profile.mean == 7.0
    assert profile.std == 0.0  # three identical values: sample std is 0

    single = profile_statistics(pd.DataFrame({"n": ["7"]})).by_name("n")
    assert single.numeric_count == 1
    assert single.mean == 7.0
    assert single.std is None


def test_duplicate_column_names_are_both_kept():
    df = pd.DataFrame([["1", "a"], ["2", "b"]], columns=["x", "x"])

    profile = profile_statistics(df)

    assert len(profile.columns) == 2
    assert profile.columns[0].numeric_count == 2
    assert profile.columns[1].numeric_count == 0
    assert profile.by_name("x").mean == 1.5


def test_zero_row_frame():
    df = pd.DataFrame({"a": []})

    profile = profile_statistics(df).by_name("a")

    assert profile.non_missing_count == 0
    assert profile.numeric_count == 0
    assert profile.mean is None


def test_non_missing_count_agrees_with_cardinality():
    df = pd.DataFrame(
        {
            "age": ["30", "", None, "25"],
            "name": ["Alice", "  ", "Bob", "Carol"],
        }
    )

    statistics = profile_statistics(df)
    cardinality = profile_cardinality(df)

    for column in statistics.columns:
        assert (
            column.non_missing_count
            == cardinality.by_name(column.column_name).non_missing_count
        )


def test_by_name_rejects_unknown_column():
    profile = profile_statistics(pd.DataFrame({"n": ["1"]}))

    with pytest.raises(KeyError):
        profile.by_name("missing")
