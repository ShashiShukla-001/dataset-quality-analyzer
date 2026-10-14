"""Flag columns that never change among their present values."""

from __future__ import annotations

from dqa.checks.config import CheckConfig
from dqa.checks.findings import Finding, Severity
from dqa.profiling.cardinality import CardinalityProfile

CHECK_NAME = "constant_column"


def check_constant(
    profile: CardinalityProfile,
    config: CheckConfig | None = None,
) -> list[Finding]:
    """Judge constant columns from a CardinalityProfile. Does not touch the DataFrame.

    Only ``distinct_count == 1`` is flagged. Low-variance cutoffs are skipped:
    any rate or count threshold would be an unjustified guess (and would mislabel
    healthy boolean columns). Fully empty columns (``distinct_count == 0``) are
    left to the missing-data check.
    """
    config = config or CheckConfig()
    findings: list[Finding] = []

    for position, column in enumerate(profile.columns):
        if column.non_missing_count < config.min_rows_for_constant_check:
            continue
        if column.distinct_count != 1:
            continue
        findings.append(
            Finding(
                check_name=CHECK_NAME,
                severity=Severity.WARNING,
                message=(
                    f"Column {column.column_name!r} is constant: "
                    f"{column.non_missing_count} present values, only 1 distinct"
                ),
                column=column.column_name,
                column_position=position,
                evidence={
                    "non_missing_count": column.non_missing_count,
                    "distinct_count": column.distinct_count,
                    "min_rows": config.min_rows_for_constant_check,
                },
            )
        )

    return findings
