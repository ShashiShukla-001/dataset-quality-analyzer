"""Shared shape for every quality-check result.

All checks emit the same ``Finding`` type so the report, API, and UI never
need a switch on check-specific result classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How serious a finding is.

    Three levels are enough to sort a report and still easy to explain.
    Using an Enum stops typos like ``"warnnig"`` from slipping through.

    Because this is a ``str`` Enum, plain ``sorted(...)`` orders alphabetically
    (error, info, warning). Always sort with ``key=lambda s: s.rank`` (or
    ``key=lambda f: f.severity.rank`` on findings) so the report order is
    info < warning < error.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Numeric order for sorting: info=0, warning=1, error=2."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}


@dataclass
class Finding:
    check_name: str
    severity: Severity
    message: str
    column: str | None = None
    column_position: int | None = None  # distinguishes duplicate column names
    evidence: dict[str, Any] = field(default_factory=dict)
