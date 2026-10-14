from dqa.checks import Severity


def test_severity_rank_sorts_info_then_warning_then_error():
    levels = [Severity.ERROR, Severity.INFO, Severity.WARNING]

    assert [s.name for s in sorted(levels, key=lambda s: s.rank)] == [
        "INFO",
        "WARNING",
        "ERROR",
    ]


def test_plain_sorted_is_alphabetical_not_severity_order():
    # documents the trap: str Enum compares as text, so never use plain sorted
    levels = [Severity.ERROR, Severity.INFO, Severity.WARNING]

    assert [s.value for s in sorted(levels)] == ["error", "info", "warning"]
