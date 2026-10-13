import pandas as pd
import pytest

from dqa.profiling import DuplicateProfile, profile_duplicates


def test_no_duplicates():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": ["30", "25"]})

    assert profile_duplicates(df) == DuplicateProfile(
        row_count=2,
        unique_row_count=2,
        duplicate_row_count=0,
        duplicate_rate=0.0,
        duplicate_group_count=0,
    )


def test_exact_duplicate_rows():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "age": ["30", "25", "30"],
        }
    )

    profile = profile_duplicates(df)

    assert profile.row_count == 3
    assert profile.unique_row_count == 2
    assert profile.duplicate_row_count == 1
    assert profile.duplicate_rate == pytest.approx(1 / 3)
    assert profile.duplicate_group_count == 1


def test_multiple_copies_and_groups():
    # A appears 3 times, B twice, C once → 3 extras, 2 groups
    df = pd.DataFrame(
        {
            "x": ["A", "A", "A", "B", "B", "C"],
            "y": ["1", "1", "1", "2", "2", "3"],
        }
    )

    profile = profile_duplicates(df)

    assert profile.unique_row_count == 3
    assert profile.duplicate_row_count == 3
    assert profile.duplicate_group_count == 2
    assert profile.duplicate_rate == pytest.approx(0.5)


def test_whitespace_and_null_count_as_the_same_missing_cell():
    # same content once the project's missing rule is applied
    df = pd.DataFrame(
        {
            "name": ["Alice", "Alice"],
            "note": ["", None],
        }
    )

    profile = profile_duplicates(df)

    assert profile.duplicate_row_count == 1
    assert profile.unique_row_count == 1
    # needs dropna=False on value_counts; without it this group is dropped
    assert profile.duplicate_group_count == 1


def test_partial_overlap_is_not_a_duplicate():
    # identical except one column — not an exact row duplicate
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "name": ["Alice", "Alice"],
        }
    )

    profile = profile_duplicates(df)

    assert profile.duplicate_row_count == 0
    assert profile.duplicate_group_count == 0


def test_empty_frame():
    df = pd.DataFrame({"a": [], "b": []})

    assert profile_duplicates(df) == DuplicateProfile(
        row_count=0,
        unique_row_count=0,
        duplicate_row_count=0,
        duplicate_rate=0.0,
        duplicate_group_count=0,
    )


def test_duplicate_column_names_still_compare_all_columns():
    # two columns both named "a"; both participate in the row comparison
    df = pd.DataFrame([["1", "x"], ["1", "x"], ["1", "y"]], columns=["a", "a"])

    profile = profile_duplicates(df)

    assert profile.row_count == 3
    assert profile.unique_row_count == 2
    assert profile.duplicate_row_count == 1
