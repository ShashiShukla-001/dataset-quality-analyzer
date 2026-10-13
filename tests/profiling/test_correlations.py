import pandas as pd
import pytest

from dqa.profiling import profile_correlations


def test_perfect_positive_and_negative_correlation():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3"],
            "b": ["2", "4", "6"],
            "c": ["3", "2", "1"],
        }
    )

    pairs = {(p.column_a, p.column_b): p for p in profile_correlations(df).pairs}

    assert pairs[("a", "b")].pearson == pytest.approx(1.0)
    assert pairs[("a", "b")].pair_count == 3
    assert pairs[("a", "c")].pearson == pytest.approx(-1.0)
    assert pairs[("b", "c")].pearson == pytest.approx(-1.0)
    assert len(pairs) == 3  # each pair once


def test_stray_words_lower_pair_count():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "oops", "4"],
            "b": ["2", "4", "6", "8"],
        }
    )

    pair = profile_correlations(df).pairs[0]

    assert pair.pair_count == 3
    assert pair.pearson == pytest.approx(1.0)


def test_whitespace_and_null_excluded_from_pair():
    df = pd.DataFrame(
        {
            "a": ["1", "", None, "4"],
            "b": ["2", "  ", "6", "8"],
        }
    )

    pair = profile_correlations(df).pairs[0]

    assert pair.pair_count == 2
    assert pair.pearson == pytest.approx(1.0)


def test_constant_column_has_undefined_correlation():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3"],
            "k": ["5", "5", "5"],
        }
    )

    pair = profile_correlations(df).pairs[0]

    assert pair.pair_count == 3
    assert pair.pearson is None


def test_all_text_column_appears_in_no_pairs():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3"],
            "name": ["Alice", "Bob", "Carol"],
            "b": ["2", "4", "6"],
        }
    )

    profile = profile_correlations(df)

    assert len(profile.pairs) == 1
    assert profile.pairs[0].column_a == "a"
    assert profile.pairs[0].column_b == "b"


def test_duplicate_column_names_use_positions():
    df = pd.DataFrame([["1", "2"], ["2", "4"], ["3", "6"]], columns=["x", "x"])

    pair = profile_correlations(df).pairs[0]

    assert pair.column_a == "x"
    assert pair.column_b == "x"
    assert pair.position_a == 0
    assert pair.position_b == 1
    assert pair.pearson == pytest.approx(1.0)


def test_zero_row_frame():
    df = pd.DataFrame({"a": [], "b": []})

    assert profile_correlations(df).pairs == []


def test_single_shared_row_is_undefined():
    df = pd.DataFrame(
        {
            "a": ["1", "oops", "oops"],
            "b": ["2", "oops", "oops"],
        }
    )

    pair = profile_correlations(df).pairs[0]

    assert pair.pair_count == 1
    assert pair.pearson is None


def test_shared_rows_constant_on_overlap_is_undefined():
    # three shared numeric rows; b is constant on those rows → no pearson

    df = pd.DataFrame(
        {
            "a": ["1", "2", "3", "oops"],
            "b": ["2", "2", "2", "9"],
        }
    )
    pair = profile_correlations(df).pairs[0]

    assert pair.pair_count == 3
    assert pair.pearson is None




