"""Flag columns whose min/max sit strictly beyond Tukey IQR fences.

Uses only StatisticsProfile facts (q1, q3, min, max). That detects whether
any extreme exists, not how many — a count can be added as a profiler fact later.

Known limits:

- A mostly-text column with a few numbers can still get stats, and we do not
  yet require a minimum numeric share. ``min_numeric_for_outliers`` is the
  only size guard for that case.
- Columns where at least half the values are identical are skipped, because
  the fences collapse (IQR == 0) and every different value would be flagged.
"""

from __future__ import annotations

from dqa.checks.config import CheckConfig
from dqa.checks.findings import Finding, Severity
from dqa.profiling.statistics import StatisticsProfile

CHECK_NAME = "outliers"


def check_outliers(
    profile: StatisticsProfile,
    config: CheckConfig | None = None,
) -> list[Finding]:
    """Judge outliers from a StatisticsProfile. Does not touch the DataFrame."""
    config = config or CheckConfig()
    findings: list[Finding] = []

    for position, column in enumerate(profile.columns):
        finding = _column_finding(column, position, config)
        if finding is not None:
            findings.append(finding)

    return findings


def _column_finding(column, position: int, config: CheckConfig) -> Finding | None:
    if column.numeric_count < config.min_numeric_for_outliers:
        return None
    if column.q1 is None or column.q3 is None or column.min is None or column.max is None:
        return None

    iqr = column.q3 - column.q1
    # Columns where at least half the values are identical are skipped, because
    # the fences collapse (IQR == 0) and every different value would be flagged.
    if iqr == 0:
        return None

    k = config.outlier_iqr_multiplier
    lower = column.q1 - k * iqr
    upper = column.q3 + k * iqr

    # Strict inequality: a value exactly on the fence is not an outlier
    # (standard Tukey definition). Missing-data thresholds use >= because those
    # are policy cutoffs ("at or above this rate is a problem"), not fence edges.
    low = column.min < lower
    high = column.max > upper
    if not low and not high:
        return None

    sides = []
    if low:
        sides.append("below lower fence")
    if high:
        sides.append("above upper fence")
    side_text = " and ".join(sides)

    return Finding(
        check_name=CHECK_NAME,
        severity=Severity.INFO,
        message=(
            f"Column {column.column_name!r} has values {side_text} "
            f"(k={k}, IQR={iqr})"
        ),
        column=column.column_name,
        column_position=position,
        evidence={
            "numeric_count": column.numeric_count,
            "q1": column.q1,
            "q3": column.q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "min": column.min,
            "max": column.max,
            "k": k,
        },
    )
