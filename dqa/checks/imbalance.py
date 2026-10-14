"""Flag a skewed class distribution on a target column.

Measure is largest_count / smallest_count: balanced targets score 1.0 regardless
of class count. Skips single-class, high-cardinality, and truncated profiles
(truncation would hide the true smallest class).
"""

from __future__ import annotations

from dqa.checks.config import CheckConfig
from dqa.checks.findings import Finding, Severity
from dqa.profiling.value_counts import ValueCountsProfile

CHECK_NAME = "class_imbalance"


def check_imbalance(
    profile: ValueCountsProfile,
    config: CheckConfig | None = None,
) -> list[Finding]:
    """Judge class imbalance from a ValueCountsProfile. Does not touch the DataFrame."""
    config = config or CheckConfig()

    if profile.truncated:
        return []
    if profile.distinct_count < 2:
        return []
    if profile.distinct_count > config.max_classes_for_imbalance:
        return []
    if not profile.values:
        return []

    largest = profile.values[0]  # sorted most → least
    smallest = min(profile.values, key=lambda item: item.count)
    ratio = largest.count / smallest.count

    if ratio >= config.imbalance_error_ratio:
        severity = Severity.ERROR
        threshold = config.imbalance_error_ratio
        label = "error"
    elif ratio >= config.imbalance_warning_ratio:
        severity = Severity.WARNING
        threshold = config.imbalance_warning_ratio
        label = "warning"
    else:
        return []

    return [
        Finding(
            check_name=CHECK_NAME,
            severity=severity,
            message=(
                f"Target {profile.column_name!r} class imbalance ratio is {ratio:.1f} "
                f"(largest {largest.value!r}={largest.count}, "
                f"smallest {smallest.value!r}={smallest.count}), "
                f"at or above the {threshold:g} {label} threshold"
            ),
            column=profile.column_name,
            column_position=profile.column_position,
            evidence={
                "ratio": ratio,
                "largest_class": largest.value,
                "largest_count": largest.count,
                "smallest_class": smallest.value,
                "smallest_count": smallest.count,
                "class_count": profile.distinct_count,
                "threshold": threshold,
            },
        )
    ]
