import pandas as pd

from dqa.profiling import ShapeProfile, profile_shape


def test_normal_dataframe():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": ["30", "25"]})

    assert profile_shape(df) == ShapeProfile(
        row_count=2,
        column_count=2,
        column_names=["name", "age"],
        duplicate_column_names=[],
    )


def test_empty_dataframe_keeps_its_columns():
    df = pd.DataFrame({"name": [], "age": []})

    profile = profile_shape(df)

    assert profile.row_count == 0
    assert profile.column_count == 2
    assert profile.column_names == ["name", "age"]


def test_duplicate_column_names_are_reported():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])

    profile = profile_shape(df)

    assert profile.column_count == 3
    assert profile.column_names == ["a", "a", "b"]
    assert profile.duplicate_column_names == ["a"]


def test_non_string_column_names_become_strings():
    # what Excel actually hands back when the header row holds numbers or dates
    df = pd.DataFrame({2024: ["a"], pd.Timestamp("2024-01-01"): ["b"]})

    assert profile_shape(df).column_names == ["2024", "2024-01-01 00:00:00"]
