"""Thresholds and policy knobs for quality checks.

Judgment lives here, not inside profilers. Callers pass a ``CheckConfig`` into
the checks layer; profiling stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckConfig:
    """Immutable check policy.

    Frozen so a check cannot mutate a threshold mid-run (e.g. tightening a
    cutoff after seeing the first column) and silently change later findings.
    To use different thresholds, pass a new ``CheckConfig`` instance.
    """

    # Warn when a column is ~20%+ empty: noticeable, still often usable.
    missing_warning_rate: float = 0.20
    # Error when a column is ~50%+ empty: half the signal is gone.
    missing_error_rate: float = 0.50
    # Warn when missingness stays ~10%+ after dropping the worst column — i.e.
    # holes are spread across the table, not explained by one empty column alone.
    residual_missing_warning_rate: float = 0.10
    # Need at least this many present values before "constant" means anything.
    # Based on non_missing_count per column, not dataset row_count: one present
    # value in a mostly-empty column is a missing-data issue, not a constant one.
    min_rows_for_constant_check: int = 2
    # Tukey fence multiplier: outlier if outside [q1 - k*IQR, q3 + k*IQR].
    outlier_iqr_multiplier: float = 1.5
    # Quartiles on fewer than this many numbers are too noisy to trust.
    min_numeric_for_outliers: int = 4
    # Target looks like a class label only up to this many distinct values.
    max_classes_for_imbalance: int = 20
    # largest/smallest class count; 4.0 ≈ 80/20 for binary targets.
    imbalance_warning_ratio: float = 4.0
    # 19.0 so a classic 95/5 split (ratio 19) is an ERROR under >=, not stuck
    # as a WARNING under a 20.0 cutoff that only fires at 95.2/4.8.
    imbalance_error_ratio: float = 19.0
