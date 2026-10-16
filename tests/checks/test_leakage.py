from dqa.checks import CheckConfig, Severity, check_leakage
from dqa.checks.leakage import CHECK_NAME
from dqa.profiling.correlations import CorrelationPair, CorrelationProfile


def make_pairs(*pairs: CorrelationPair) -> CorrelationProfile:
    return CorrelationProfile(pairs=list(pairs))


def pair(
    a: str,
    b: str,
    pearson: float | None,
    *,
    position_a: int = 0,
    position_b: int = 1,
    pair_count: int = 100,
) -> CorrelationPair:
    return CorrelationPair(
        column_a=a,
        column_b=b,
        position_a=position_a,
        position_b=position_b,
        pair_count=pair_count,
        pearson=pearson,
    )


def test_target_as_first_column_of_pair():
    profile = make_pairs(pair("target", "feature", 0.98, position_a=0, position_b=1))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].check_name == CHECK_NAME
    assert findings[0].severity == Severity.WARNING
    assert findings[0].column == "feature"
    assert findings[0].column_position == 1
    assert "possible leakage" in findings[0].message.lower()


def test_target_as_second_column_of_pair():
    profile = make_pairs(pair("feature", "target", 0.99, position_a=0, position_b=2))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].column == "feature"
    assert findings[0].column_position == 0
    assert findings[0].evidence["target_position"] == 2


def test_no_pair_involves_target_gives_info():
    profile = make_pairs(pair("x", "y", 0.99))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert findings[0].column is None
    assert findings[0].evidence["reason"] == "target_not_in_pairs"


def test_empty_pairs_gives_info():
    findings = check_leakage(make_pairs(), "target")

    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert findings[0].column is None


def test_target_present_but_under_threshold_is_silent():
    # check ran (target is in a pair); no leaky column → empty list is honest
    profile = make_pairs(
        pair("target", "feature", 0.50),
        pair("x", "y", 0.99),
    )

    assert check_leakage(profile, "target") == []


def test_all_target_pairs_undefined_gives_info():
    # constant / undefined correlations: check could not assess anything
    profile = make_pairs(pair("target", "feature", None))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert findings[0].evidence["reason"] == "target_not_in_pairs"


def test_undefined_pair_beside_defined_low_pair_is_silent():
    # at least one usable target pair → check ran; under threshold → []
    profile = make_pairs(
        pair("target", "a", None, position_a=0, position_b=1),
        pair("target", "b", 0.50, position_a=0, position_b=2),
    )

    assert check_leakage(profile, "target") == []


def test_exact_threshold_is_a_warning():
    profile = make_pairs(pair("target", "feature", 0.95))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["abs_pearson"] == 0.95
    assert findings[0].evidence["threshold"] == 0.95


def test_just_under_threshold_is_silent():
    profile = make_pairs(pair("target", "feature", 0.949))

    assert check_leakage(profile, "target") == []


def test_negative_correlation_uses_absolute_value():
    profile = make_pairs(pair("target", "feature", -0.96))

    findings = check_leakage(profile, "target")

    assert len(findings) == 1
    assert findings[0].evidence["abs_pearson"] == 0.96


def test_custom_threshold():
    profile = make_pairs(pair("target", "feature", 0.91))

    assert check_leakage(profile, "target") == []
    findings = check_leakage(
        profile, "target", config=CheckConfig(leakage_correlation_threshold=0.90)
    )
    assert len(findings) == 1


def test_target_position_disambiguates_duplicate_names():
    profile = make_pairs(
        pair("target", "a", 0.99, position_a=0, position_b=1),
        pair("b", "target", 0.99, position_a=1, position_b=2),
    )

    findings = check_leakage(profile, "target", target_position=2)

    assert len(findings) == 1
    assert findings[0].column == "b"
    assert findings[0].evidence["target_position"] == 2
