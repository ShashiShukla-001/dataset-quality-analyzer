"""Flag columns (and the dataset) whose missing rates cross configured thresholds."""

from __future__ import annotations

from dqa.checks.config import CheckConfig
from dqa.checks.findings import Finding, Severity
from dqa.profiling.missing import MissingProfile

CHECK_NAME = "missing_data"


def check_missing(
    profile: MissingProfile,
    config: CheckConfig | None = None,
) -> list[Finding]:
    """Judge missing-value rates from a MissingProfile. Does not touch the DataFrame."""
    config = config or CheckConfig()
    findings: list[Finding] = []

    for position, column in enumerate(profile.columns):
        finding = _column_finding(
            column.column_name,
            position,
            column.null_count,
            column.null_rate,
            config,
        )
        if finding is not None:
            findings.append(finding)

    spread = _spread_missing(profile)
    if spread is not None and spread[1] >= config.residual_missing_warning_rate:
        residual_count, residual_rate = spread
        findings.append(
            Finding(
                check_name=CHECK_NAME,
                severity=Severity.WARNING,
                message=(
                    f"Missingness remains at {residual_rate:.0%} after excluding the "
                    f"worst column, at or above the "
                    f"{config.residual_missing_warning_rate:.0%} warning threshold"
                ),
                column=None,
                column_position=None,
                evidence={
                    "null_count": residual_count,
                    "null_rate": residual_rate,
                    "threshold": config.residual_missing_warning_rate,
                },
            )
        )

    return findings


def _spread_missing(profile: MissingProfile) -> tuple[int, float] | None:
    """Missing count and rate after removing the single worst column.

    Returns None when there aren't at least two columns, so "spread thin"
    isn't meaningful and one bad column can't double-count as a dataset finding.
    """
    if profile.column_count < 2 or profile.row_count == 0:
        return None

    worst = max(column.null_count for column in profile.columns)
    residual_count = profile.total_null_cells - worst
    residual_cells = profile.total_cells - profile.row_count
    residual_rate = residual_count / residual_cells if residual_cells else 0.0
    return residual_count, residual_rate


def _column_finding(
    name: str,
    position: int,
    null_count: int,
    null_rate: float,
    config: CheckConfig,
) -> Finding | None:
    if null_rate >= config.missing_error_rate:
        severity = Severity.ERROR
        threshold = config.missing_error_rate
        label = "error"
    elif null_rate >= config.missing_warning_rate:
        severity = Severity.WARNING
        threshold = config.missing_warning_rate
        label = "warning"
    else:
        return None

    return Finding(
        check_name=CHECK_NAME,
        severity=severity,
        message=(
            f"Column {name!r} missing rate is {null_rate:.0%}, "
            f"at or above the {threshold:.0%} {label} threshold"
        ),
        column=name,
        column_position=position,
        evidence={
            "null_count": null_count,
            "null_rate": null_rate,
            "threshold": threshold,
        },
    )
