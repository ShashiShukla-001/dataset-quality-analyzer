from dqa.checks import CheckConfig, Severity, check_missing
from dqa.profiling.missing import ColumnMissingProfile, MissingProfile


def make_profile(
    row_count: int,
    null_counts: list[int],
    names: list[str] | None = None,
) -> MissingProfile:
    """Build a MissingProfile without going through a DataFrame."""
    column_count = len(null_counts)
    if names is None:
        names = [f"c{i}" for i in range(column_count)]
    columns = [
        ColumnMissingProfile(
            column_name=name,
            null_count=count,
            null_rate=(count / row_count) if row_count else 0.0,
        )
        for name, count in zip(names, null_counts, strict=True)
    ]
    total_null_cells = sum(null_counts)
    total_cells = row_count * column_count
    return MissingProfile(
        columns=columns,
        row_count=row_count,
        column_count=column_count,
        total_cells=total_cells,
        total_null_cells=total_null_cells,
        overall_null_rate=(total_null_cells / total_cells) if total_cells else 0.0,
    )


def test_one_empty_column_does_not_also_trigger_residual_warning():
    # 1 of 10 fully missing → residual after worst is 0%
    findings = check_missing(make_profile(2, [0] * 9 + [2]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert findings[0].column == "c9"


def test_warning_band_error_band_and_clean_column():
    # 5/25=0.20 warning, 15/25=0.60 error, 0 clean
    #a residual dataset finding also appears here (5/50 = 10%); this test only looks at column findings
    findings = check_missing(make_profile(25, [5, 15, 0], names=["warn", "bad", "ok"]))

    by_column = {f.column: f for f in findings if f.column is not None}
    assert set(by_column) == {"warn", "bad"}
    assert by_column["warn"].severity == Severity.WARNING
    assert by_column["bad"].severity == Severity.ERROR


def test_exact_warning_boundary():
    findings = check_missing(make_profile(25, [5], names=["edge"]))

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["null_rate"] == 0.20
    assert findings[0].evidence["threshold"] == 0.20


def test_clean_dataset_gives_no_findings():
    assert check_missing(make_profile(10, [0, 0, 0])) == []


def test_duplicate_names_use_column_position():
    findings = check_missing(make_profile(4, [0, 2], names=["a", "a"]))

    assert len(findings) == 1
    assert findings[0].column == "a"
    assert findings[0].column_position == 1
    assert findings[0].severity == Severity.ERROR


def test_custom_config_controls_thresholds():
    profile = make_profile(10, [3], names=["a"])  # 30%: warning under defaults
    findings = check_missing(
        profile,
        CheckConfig(missing_warning_rate=0.50, missing_error_rate=0.80),
    )

    assert findings == []


def test_spread_below_column_threshold_only_residual_warning():
    # 10 columns × 3/25 = 12% each: under column warning, residual stays ≥ 10%
    findings = check_missing(make_profile(25, [3] * 10))

    assert len(findings) == 1
    assert findings[0].column is None
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["threshold"] == CheckConfig().residual_missing_warning_rate


def test_single_column_dataset_has_no_residual_finding():
    findings = check_missing(make_profile(10, [10], names=["only"]))

    assert len(findings) == 1
    assert findings[0].column == "only"
    assert findings[0].severity == Severity.ERROR


def test_two_empty_columns_still_keep_residual_warning():
    # Decision: remove only the single worst column. A second fully empty
    # column leaves residual ≈ 11% (≥ 10%), so the overall WARNING stays —
    # missingness is no longer explained by one column alone.
    findings = check_missing(make_profile(2, [0] * 8 + [2, 2]))

    residual = [f for f in findings if f.column is None]
    column_errors = [f for f in findings if f.severity == Severity.ERROR]

    assert len(column_errors) == 2
    assert len(residual) == 1
    assert residual[0].severity == Severity.WARNING

def test_exact_residual_boundary():
    # each column 10% (under the column warning), residual 2/20 = exactly 0.10
    findings = check_missing(make_profile(10, [1, 1, 1]))

    assert len(findings) == 1
    assert findings[0].column is None
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["null_rate"] == 0.10
    assert findings[0].evidence["threshold"] == 0.10
