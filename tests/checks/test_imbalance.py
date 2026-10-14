from dqa.checks import CheckConfig, Severity, check_imbalance
from dqa.checks.imbalance import CHECK_NAME
from dqa.profiling.value_counts import ValueCount, ValueCountsProfile


def make_counts(
    counts: list[int],
    *,
    column_name: str = "target",
    column_position: int = 0,
    truncated: bool = False,
    distinct_count: int | None = None,
) -> ValueCountsProfile:
    """Build a ValueCountsProfile from class sizes (already most → least)."""
    values = [
        ValueCount(value=f"c{i}", count=count) for i, count in enumerate(counts)
    ]
    return ValueCountsProfile(
        column_name=column_name,
        column_position=column_position,
        non_missing_count=sum(counts),
        distinct_count=distinct_count if distinct_count is not None else len(counts),
        values=values,
        truncated=truncated,
    )


def test_balanced_classes_no_finding():
    assert check_imbalance(make_counts([10, 10, 10])) == []


def test_warning_band():
    # 80/20 → ratio 4.0 is the warning boundary; 40/10 = 4.0
    findings = check_imbalance(make_counts([40, 10]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["ratio"] == 4.0


def test_error_band():
    # 95/5 → ratio 19.0 is the error boundary under imbalance_error_ratio=19
    findings = check_imbalance(make_counts([95, 5]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert findings[0].evidence["ratio"] == 19.0


def test_exact_warning_boundary():
    findings = check_imbalance(make_counts([80, 20]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["ratio"] == 4.0
    assert findings[0].evidence["threshold"] == 4.0


def test_exact_error_boundary():
    findings = check_imbalance(make_counts([95, 5]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert findings[0].evidence["ratio"] == 19.0
    assert findings[0].evidence["threshold"] == 19.0


def test_just_under_error_is_warning():
    # 90/10 = 9.0: above warning, below error
    findings = check_imbalance(make_counts([90, 10]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["ratio"] == 9.0


def test_single_class_skipped():
    assert check_imbalance(make_counts([100])) == []


def test_too_many_classes_skipped():
    assert check_imbalance(make_counts([10] * 21)) == []


def test_truncated_profile_skipped():
    # Even with an extreme ratio in the kept list, truncation means we cannot
    # trust the smallest class — skip rather than invent a wrong finding.
    profile = make_counts([1000, 1], truncated=True, distinct_count=150)
    assert check_imbalance(profile) == []


def test_custom_config():
    profile = make_counts([80, 20])  # ratio 4.0
    assert check_imbalance(profile, CheckConfig(imbalance_warning_ratio=5.0)) == []


def test_finding_carries_column_name_and_position():
    # 90/10 = 9.0: clearly in the warning band, not on the 4.0 boundary
    findings = check_imbalance(
        make_counts([90, 10], column_name="label", column_position=3)
    )

    assert findings[0].check_name == CHECK_NAME
    assert findings[0].column == "label"
    assert findings[0].column_position == 3
    assert findings[0].evidence["largest_class"] == "c0"
    assert findings[0].evidence["smallest_class"] == "c1"
    assert findings[0].evidence["class_count"] == 2
