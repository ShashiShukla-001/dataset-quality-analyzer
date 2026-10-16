"""Flag features whose numeric correlation with the target looks like leakage.

Scope is Pearson pairs only (option A). A broader “predicts the target”
profiler can come later.

If no *usable* correlation pair involves the target (text target, typo, empty
profile, or every pair undefined), returns one INFO finding that leakage
could not be assessed — an empty list would look like “no leakage found”,
which is false reassurance. Whether the target column exists on the
DataFrame belongs in the report step (``ColumnNotFoundError``). How other
checks report “did not run, and why” is also a report-step decision; this
INFO finding is the first small case.
"""

from __future__ import annotations

from dqa.checks.config import CheckConfig
from dqa.checks.findings import Finding, Severity
from dqa.profiling.correlations import CorrelationPair, CorrelationProfile

CHECK_NAME = "possible_leakage"


def check_leakage(
    profile: CorrelationProfile,
    target: str,
    *,
    target_position: int | None = None,
    config: CheckConfig | None = None,
) -> list[Finding]:
    """Judge possible leakage from correlations involving ``target``."""
    config = config or CheckConfig()
    threshold = config.leakage_correlation_threshold
    findings: list[Finding] = []
    saw_usable_target_pair = False

    for pair in profile.pairs:
        role = _target_role(pair, target, target_position)
        if role is None:
            continue
        if pair.pearson is None:
            continue
        # Only a defined pearson means the check could actually assess this pair.
        saw_usable_target_pair = True
        if abs(pair.pearson) < threshold:
            continue

        if role == "a":
            feature_name, feature_position = pair.column_b, pair.position_b
        else:
            feature_name, feature_position = pair.column_a, pair.position_a

        findings.append(
            Finding(
                check_name=CHECK_NAME,
                severity=Severity.WARNING,
                message=(
                    f"Possible leakage: column {feature_name!r} has "
                    f"|pearson|={abs(pair.pearson):.3f} with target {target!r} "
                    f"(threshold {threshold:g})"
                ),
                column=feature_name,
                column_position=feature_position,
                evidence={
                    "target": target,
                    "target_position": (
                        pair.position_a if role == "a" else pair.position_b
                    ),
                    "pearson": pair.pearson,
                    "abs_pearson": abs(pair.pearson),
                    "pair_count": pair.pair_count,
                    "threshold": threshold,
                },
            )
        )

    if not saw_usable_target_pair:
        return [
            Finding(
                check_name=CHECK_NAME,
                severity=Severity.INFO,
                message=(
                    f"Leakage could not be assessed for target {target!r}: "
                    f"not numeric, not found, or its correlations are undefined"
                ),
                column=None,
                column_position=None,
                evidence={"target": target, "reason": "target_not_in_pairs"},
            )
        ]

    return findings


def _target_role(
    pair: CorrelationPair,
    target: str,
    target_position: int | None,
) -> str | None:
    """Return ``'a'`` or ``'b'`` if that side is the target, else None."""
    if target_position is not None:
        if pair.position_a == target_position and pair.column_a == target:
            return "a"
        if pair.position_b == target_position and pair.column_b == target:
            return "b"
        return None
    if pair.column_a == target:
        return "a"
    if pair.column_b == target:
        return "b"
    return None
