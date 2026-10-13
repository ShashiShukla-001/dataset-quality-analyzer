"""Shared rules for recognising and parsing numeric text.

int() and float() are too permissive: int("1_000") is 1000, int() accepts
non-ASCII digits, and float() accepts "nan" and "inf". Every profiler that
asks "is this a number?" should go through here so the answers stay aligned.
"""

from __future__ import annotations

import math
import re

_INTEGER = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def is_integer(value: str) -> bool:
    return _INTEGER.fullmatch(value) is not None


def is_float(value: str) -> bool:
    return _FLOAT.fullmatch(value) is not None


def parse_number(text: str) -> float | None:
    """Parse ``text`` as a finite float, or return None when it is not usable.

    Overflows like ``"1e999"`` become ``inf`` under ``float()``; those are not
    real measurements, so they are rejected the same way non-numeric text is.
    """
    if not is_float(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None
