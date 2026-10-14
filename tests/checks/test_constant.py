from dqa.checks import CheckConfig, Severity, check_constant
from dqa.checks.constant import CHECK_NAME
from dqa.profiling.cardinality import CardinalityProfile, ColumnCardinalityProfile


def make_cardinality(
    columns: list[tuple[str, int, int]],
) -> CardinalityProfile:
    """Build a CardinalityProfile as (name, non_missing_count, distinct_count)."""
    profiles = []
    for name, non_missing, distinct in columns:
        rate = (distinct / non_missing) if non_missing else 0.0
        profiles.append(
            ColumnCardinalityProfile(
                column_name=name,
                non_missing_count=non_missing,
                distinct_count=distinct,
                distinct_rate=rate,
            )
        )
    return CardinalityProfile(columns=profiles)


def test_constant_column_is_warning():
    findings = check_constant(
        make_cardinality([("status", 100, 1), ("id", 100, 100)])
    )

    assert len(findings) == 1
    assert findings[0].check_name == CHECK_NAME
    assert findings[0].column == "status"
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["distinct_count"] == 1


def test_varying_column_is_silent():
    assert check_constant(make_cardinality([("id", 50, 50)])) == []


def test_fully_empty_column_is_skipped():
    # Decision: distinct_count == 0 is the missing-data check's job, not ours.
    findings = check_constant(make_cardinality([("empty", 0, 0), ("ok", 10, 5)]))

    assert findings == []


def test_single_present_value_is_skipped_by_default():
    # With one present value every column looks constant; that finding is noise.
    findings = check_constant(make_cardinality([("lonely", 1, 1)]))

    assert findings == []


def test_custom_min_rows_allows_stricter_skip():
    profile = make_cardinality([("almost", 3, 1)])
    assert len(check_constant(profile)) == 1

    findings = check_constant(profile, CheckConfig(min_rows_for_constant_check=5))
    assert findings == []


def test_duplicate_names_use_column_position():
    findings = check_constant(
        make_cardinality([("a", 10, 5), ("a", 10, 1)])
    )

    assert len(findings) == 1
    assert findings[0].column == "a"
    assert findings[0].column_position == 1


def test_exactly_min_rows_is_checked():
    findings = check_constant(make_cardinality([("pair", 2, 1)]))
    assert len(findings) == 1