"""Assemble profiles and check findings into one quality report.

Entry points:
- ``analyze_dataset`` — any DataFrame (tests, library callers)
- ``analyze_file`` — load then analyze; copies loader metadata only (never
  embeds ``LoadResult``, which holds a DataFrame and would break JSON)

When ``target`` is set, ``profile_value_counts`` runs first so a bad name
raises ``ColumnNotFoundError`` before checks. Leakage then uses that
profile's ``column_position`` so duplicate target names resolve correctly.

When ``target`` is None, imbalance and leakage each get one INFO finding with
``evidence["reason"] == "no_target"`` so “did not run” is not silent.
Other checks that skip for size/cardinality still return ``[]``; saying why
for those remains a future report concern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from dqa.checks import (
    CheckConfig,
    Finding,
    Severity,
    check_constant,
    check_imbalance,
    check_leakage,
    check_missing,
    check_outliers,
)
from dqa.checks.imbalance import CHECK_NAME as IMBALANCE_CHECK
from dqa.checks.leakage import CHECK_NAME as LEAKAGE_CHECK
from dqa.loader import FileSource, LoadResult, load_dataset
from dqa.profiling import (
    CardinalityProfile,
    CorrelationProfile,
    DuplicateProfile,
    DTypeProfile,
    MissingProfile,
    ShapeProfile,
    StatisticsProfile,
    ValueCountsProfile,
    profile_cardinality,
    profile_correlations,
    profile_dtypes,
    profile_duplicates,
    profile_missing,
    profile_shape,
    profile_statistics,
    profile_value_counts,
)


@dataclass
class FileMetadata:
    source_name: str | None
    file_format: str | None
    sheet_name: str | None
    available_sheets: list[str] | None
    row_count: int
    column_count: int


@dataclass
class SeverityCounts:
    info: int
    warning: int
    error: int


@dataclass
class QualityReport:
    metadata: FileMetadata
    target: str | None
    shape: ShapeProfile
    dtypes: DTypeProfile
    missing: MissingProfile
    duplicates: DuplicateProfile
    cardinality: CardinalityProfile
    statistics: StatisticsProfile
    correlations: CorrelationProfile
    value_counts: ValueCountsProfile | None
    findings: list[Finding]
    severity_counts: SeverityCounts

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (``str`` Enum severities serialize cleanly)."""
        return asdict(self)


def analyze_file(
    file: FileSource,
    *,
    sheet_name: str | None = None,
    target: str | None = None,
    config: CheckConfig | None = None,
    filename: str | None = None,
) -> QualityReport:
    """Load a file, then run the full analyze pipeline."""
    loaded = load_dataset(file, sheet_name=sheet_name, filename=filename)
    return analyze_dataset(
        loaded.dataframe,
        target=target,
        config=config,
        source_name=loaded.source_name,
        file_format=loaded.file_format,
        sheet_name=loaded.sheet_name,
        available_sheets=loaded.available_sheets,
    )


def analyze_dataset(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    config: CheckConfig | None = None,
    source_name: str | None = None,
    file_format: str | None = None,
    sheet_name: str | None = None,
    available_sheets: list[str] | None = None,
) -> QualityReport:
    """Profile a DataFrame, run checks, and assemble a QualityReport."""
    config = config or CheckConfig()

    # Target existence first: raises ColumnNotFoundError before any check runs.
    value_counts: ValueCountsProfile | None = None
    if target is not None:
        value_counts = profile_value_counts(df, target)

    shape = profile_shape(df)
    dtypes = profile_dtypes(df)
    missing = profile_missing(df)
    duplicates = profile_duplicates(df)
    cardinality = profile_cardinality(df)
    statistics = profile_statistics(df)
    correlations = profile_correlations(df)

    findings: list[Finding] = []
    findings.extend(check_missing(missing, config))
    findings.extend(check_constant(cardinality, config))
    findings.extend(check_outliers(statistics, config))

    if target is None:
        findings.append(_skipped_check(IMBALANCE_CHECK, "no_target"))
        findings.append(_skipped_check(LEAKAGE_CHECK, "no_target"))
    else:
        assert value_counts is not None
        findings.extend(check_imbalance(value_counts, config))
        findings.extend(
            check_leakage(
                correlations,
                target,
                target_position=value_counts.column_position,
                config=config,
            )
        )

    findings.sort(key=lambda f: f.severity.rank, reverse=True)
    counts = _severity_counts(findings)

    return QualityReport(
        metadata=FileMetadata(
            source_name=source_name,
            file_format=file_format,
            sheet_name=sheet_name,
            available_sheets=available_sheets,
            row_count=len(df),
            column_count=len(df.columns),
        ),
        target=target,
        shape=shape,
        dtypes=dtypes,
        missing=missing,
        duplicates=duplicates,
        cardinality=cardinality,
        statistics=statistics,
        correlations=correlations,
        value_counts=value_counts,
        findings=findings,
        severity_counts=counts,
    )


def _skipped_check(check_name: str, reason: str) -> Finding:
    return Finding(
        check_name=check_name,
        severity=Severity.INFO,
        message=f"Check {check_name!r} was skipped ({reason})",
        column=None,
        column_position=None,
        evidence={"reason": reason},
    )


def _severity_counts(findings: list[Finding]) -> SeverityCounts:
    counts = {Severity.INFO: 0, Severity.WARNING: 0, Severity.ERROR: 0}
    for finding in findings:
        counts[finding.severity] += 1
    return SeverityCounts(
        info=counts[Severity.INFO],
        warning=counts[Severity.WARNING],
        error=counts[Severity.ERROR],
    )
