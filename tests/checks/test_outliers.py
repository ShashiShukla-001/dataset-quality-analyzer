from dqa.checks import CheckConfig, Severity, check_outliers
from dqa.checks.outliers import CHECK_NAME
from dqa.profiling.statistics import ColumnStatisticsProfile, StatisticsProfile


def make_stats(
    name: str,
    numeric_count: int,
    q1: float | None,
    q3: float | None,
    min_value: float | None,
    max_value: float | None,
    *,
    non_missing_count: int | None = None,
) -> StatisticsProfile:
    """One-column StatisticsProfile for outlier tests."""
    return make_stats_columns(
        [(name, numeric_count, q1, q3, min_value, max_value, non_missing_count)]
    )


def make_stats_columns(
    columns: list[tuple],
) -> StatisticsProfile:
    profiles = []
    for item in columns:
        name, numeric_count, q1, q3, min_value, max_value = item[:6]
        non_missing = item[6] if len(item) > 6 and item[6] is not None else numeric_count
        profiles.append(
            ColumnStatisticsProfile(
                column_name=name,
                non_missing_count=non_missing if non_missing is not None else numeric_count,
                numeric_count=numeric_count,
                min=min_value,
                max=max_value,
                mean=None,
                median=None,
                std=None,
                q1=q1,
                q3=q3,
            )
        )
    return StatisticsProfile(columns=profiles)


def test_no_outliers():
    # q1=10, q3=20 → IQR=10; fences [-5, 35]; min/max inside
    assert check_outliers(make_stats("n", 10, 10.0, 20.0, 0.0, 30.0)) == []


def test_high_outlier():
    findings = check_outliers(make_stats("n", 10, 10.0, 20.0, 0.0, 40.0))

    assert len(findings) == 1
    assert findings[0].check_name == CHECK_NAME
    assert findings[0].severity == Severity.INFO
    assert findings[0].evidence["max"] == 40.0
    assert findings[0].evidence["upper_fence"] == 35.0


def test_low_outlier():
    findings = check_outliers(make_stats("n", 10, 10.0, 20.0, -10.0, 30.0))

    assert len(findings) == 1
    assert findings[0].evidence["min"] == -10.0
    assert findings[0].evidence["lower_fence"] == -5.0


def test_value_exactly_on_fence_is_not_an_outlier():
    # min == lower fence (-5); max inside — strict < / > means no finding
    assert check_outliers(make_stats("n", 10, 10.0, 20.0, -5.0, 30.0)) == []


def test_zero_iqr_is_skipped():
    # Columns where at least half the values are identical are skipped, because
    # the fences collapse (IQR == 0) and every different value would be flagged.
    assert check_outliers(make_stats("n", 10, 5.0, 5.0, 5.0, 9.0)) == []


def test_too_few_numbers_is_skipped():
    assert check_outliers(make_stats("n", 3, 10.0, 20.0, 0.0, 100.0)) == []


def test_all_text_column_is_skipped():
    assert (
        check_outliers(make_stats("name", 0, None, None, None, None, non_missing_count=10))
        == []
    )


def test_duplicate_names_use_column_position():
    profile = make_stats_columns(
        [
            ("x", 10, 10.0, 20.0, 0.0, 30.0, None),
            ("x", 10, 10.0, 20.0, 0.0, 40.0, None),
        ]
    )
    findings = check_outliers(profile)

    assert len(findings) == 1
    assert findings[0].column == "x"
    assert findings[0].column_position == 1


def test_custom_k_widens_fences():
    profile = make_stats("n", 10, 10.0, 20.0, 0.0, 40.0)
    assert len(check_outliers(profile)) == 1

    # k=3 → upper = 20 + 30 = 50; max 40 is inside
    assert check_outliers(profile, CheckConfig(outlier_iqr_multiplier=3.0)) == []

def test_max_exactly_on_upper_fence_is_not_an_outlier():
    # q1=10, q3=20, IQR=10, upper fence = 35
    assert check_outliers(make_stats("n", 10, 10.0, 20.0, 0.0, 35.0)) == []


def test_both_fences_exceeded_in_one_finding():
    findings = check_outliers(make_stats("n", 10, 10.0, 20.0, -10.0, 40.0))

    assert len(findings) == 1
    assert findings[0].evidence["min"] == -10.0
    assert findings[0].evidence["max"] == 40.0
    assert findings[0].evidence["lower_fence"] == -5.0
    assert findings[0].evidence["upper_fence"] == 35.0