import time

import pandas as pd
import pytest

from dqa.profiling import profile_dtypes
from dqa.profiling.dtypes import _is_date_shaped


def profile_one(values):
    """Profile a single column built from `values`."""
    return profile_dtypes(pd.DataFrame({"col": values})).by_name("col")


def test_integer_column():
    profile = profile_one(["1", "-2", "+3"])

    assert profile.inferred_type == "integer"
    assert profile.match_rate == 1.0
    assert profile.null_count == 0


def test_float_column():
    assert profile_one(["1.5", "2.0", "1e3", ".5"]).inferred_type == "float"


def test_whole_numbers_prefer_integer_over_float():
    # "1" matches both; the more specific type wins the tie
    assert profile_one(["1", "2", "3"]).inferred_type == "integer"


def test_mixed_integers_and_decimals_are_float():
    assert profile_one(["1", "2.5", "3"]).inferred_type == "float"


def test_boolean_column():
    assert profile_one(["true", "FALSE", "Yes", "no"]).inferred_type == "boolean"


@pytest.mark.parametrize(
    "label, values",
    [
        ("dna bases", ["T", "A", "G", "C"]),
        ("grades", ["A", "B", "F", "Y", "N"]),
        ("single-letter flags", ["Y", "N", "Y"]),
    ],
)
def test_single_letters_are_text_not_booleans(label, values):
    assert profile_one(values).inferred_type == "text", label


def test_zero_and_one_stay_integer():
    # ambiguous with booleans; reading them as integers loses nothing
    assert profile_one(["0", "1", "1", "0"]).inferred_type == "integer"


def test_date_column():
    profile = profile_one(["2024-01-01", "2024-02-15", "Jan 5 2024"])

    assert profile.inferred_type == "date"
    assert profile.match_rate == 1.0


def test_date_match_rate_does_not_depend_on_row_order():
    values = ["2024-01-01", "Jan 5 2024", "nope"]

    forwards = profile_one(values)
    backwards = profile_one(list(reversed(values)))

    assert forwards.inferred_type == backwards.inferred_type == "date"
    assert forwards.match_rate == backwards.match_rate == pytest.approx(2 / 3)


def test_mixed_timezone_offsets_do_not_crash():
    # pandas raises on mixed offsets even under errors="coerce"
    profile = profile_one(["2024-01-01T00:00:00+00:00", "2024-06-01T00:00:00+05:30"])

    assert profile.inferred_type == "date"
    assert profile.match_rate == 1.0


def test_text_column():
    profile = profile_one(["apple", "banana", "cherry"])

    assert profile.inferred_type == "text"
    assert profile.match_rate == 1.0


def test_all_null_column_is_empty():
    profile = profile_one([None, None, None])

    assert profile.inferred_type == "empty"
    assert profile.match_rate == 1.0
    assert profile.null_count == 3


def test_partial_match_is_preserved_not_flattened_to_text():
    # three integers and one stray word: the ambiguity survives as a number
    profile = profile_one(["1", "2", "3", "oops"])

    assert profile.inferred_type == "integer"
    assert profile.match_rate == 0.75


def test_nulls_are_excluded_from_the_match_rate():
    profile = profile_one(["1", "2", None, None])

    assert profile.inferred_type == "integer"
    assert profile.match_rate == 1.0
    assert profile.null_count == 2


def test_whitespace_only_cells_count_as_missing():
    profile = profile_one(["1", "", "   ", None])

    assert profile.inferred_type == "integer"
    assert profile.match_rate == 1.0
    assert profile.null_count == 3


def test_column_of_blanks_is_empty_not_text():
    profile = profile_one(["", "  ", None])

    assert profile.inferred_type == "empty"
    assert profile.null_count == 3


def test_placeholder_words_are_not_treated_as_missing():
    # "NA" means missing to a human, but that is a judgement for the checks
    # layer; here it is just text
    profile = profile_one(["NA", "N/A", "unknown"])

    assert profile.inferred_type == "text"
    assert profile.null_count == 0


def test_years_are_integers_not_dates():
    # pandas alone reads "2024" as 2024-01-01
    assert profile_one(["2024", "2025", "2026"]).inferred_type == "integer"


def test_month_names_are_text_not_dates():
    # pandas alone reads "March" as year 1
    assert profile_one(["March", "April", "May"]).inferred_type == "text"


@pytest.mark.parametrize(
    "label, values",
    [
        ("phone numbers", ["555-1234", "555-9876"]),
        ("id codes", ["12-345", "12-346"]),
        ("version strings", ["1.2.3", "2.0.1"]),
        ("times of day", ["12:30", "09:15"]),
        ("ratios", ["1/2", "3/4"]),
        ("numeric ranges", ["10-20", "30-40"]),
        ("part numbers", ["A1", "B2"]),
        ("thousands separators", ["1,234", "5,678"]),
        ("long numeric ids", ["1234567-12-12", "7654321-01-01"]),
    ],
)
def test_digit_bearing_text_is_not_mistaken_for_dates(label, values):
    assert profile_one(values).inferred_type != "date", label


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-01", "2024-02-15"],
        ["01/02/2024", "03/04/2024"],
        ["2024-1-1", "2024-2-5"],
        ["Jan 5 2024", "Feb 6 2024"],
        ["5 March 2024", "6 April 2024"],
        ["2024-01-01T10:30:00", "2024-02-01T11:45:00"],
    ],
)
def test_real_dates_are_still_recognised(values):
    profile = profile_one(values)

    assert profile.inferred_type == "date"
    assert profile.match_rate == 1.0


def test_unparseable_date_shaped_values_lower_the_match_rate():
    # right shape, impossible date: screened in, then coerced to NaT
    profile = profile_one(["2024-01-01", "99/99/9999"])

    assert profile.inferred_type == "date"
    assert profile.match_rate == 0.5


@pytest.mark.parametrize("value", ["SKU-1", "12-345", "1,234", "12:30", "1/2"])
def test_non_dates_never_reach_the_date_parser(value):
    # the screen is what keeps the slow per-value dateutil path off these
    assert not _is_date_shaped(value)


def test_large_text_column_profiles_quickly():
    # before the date screen, 200k SKU-style values took ~3s in dateutil
    df = pd.DataFrame({"col": [f"SKU-{i}" for i in range(200_000)]})

    start = time.perf_counter()
    profile = profile_dtypes(df)
    elapsed = time.perf_counter() - start

    assert profile.by_name("col").inferred_type == "text"
    assert elapsed < 2.0


@pytest.mark.parametrize("value", ["1_000", "\u0661\u0662\u0663", "nan", "inf"])
def test_values_python_would_wrongly_parse_are_text(value):
    # int("1_000") is 1000, int() takes non-ASCII digits, float() takes "nan"
    assert profile_one([value]).inferred_type == "text"


def test_profiles_every_column_in_order():
    df = pd.DataFrame({"age": ["30", "25"], "name": ["Alice", "Bob"]})

    profile = profile_dtypes(df)

    assert [c.column_name for c in profile.columns] == ["age", "name"]
    assert profile.by_name("age").inferred_type == "integer"
    assert profile.by_name("name").inferred_type == "text"


def test_duplicate_column_names_are_all_kept():
    df = pd.DataFrame([["1", "apple"]], columns=["a", "a"])

    profile = profile_dtypes(df)

    assert len(profile.columns) == 2
    assert [c.inferred_type for c in profile.columns] == ["integer", "text"]
    assert profile.by_name("a").inferred_type == "integer"  # the first one


def test_by_name_rejects_a_column_that_is_not_there():
    profile = profile_dtypes(pd.DataFrame({"age": ["30"]}))

    with pytest.raises(KeyError):
        profile.by_name("missing")
