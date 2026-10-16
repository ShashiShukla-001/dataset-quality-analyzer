"""HTTP client for the DQA FastAPI service.

Kept separate from Streamlit so the request/response handling can be unit-tested
with a stubbed ``post`` callable.
"""

from __future__ import annotations

from typing import Any, Callable

import requests

PostFn = Callable[..., Any]


class AnalyzeError(Exception):
    """Raised when the API returns a non-success status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def analyze_upload(
    base_url: str,
    filename: str,
    content: bytes,
    *,
    target: str | None = None,
    sheet_name: str | None = None,
    timeout: float = 120.0,
    post: PostFn | None = None,
) -> dict[str, Any]:
    """POST a file to ``/analyze`` and return the report dict.

    ``post`` defaults to ``requests.post``; tests pass a stub.
    """
    post_fn = post or requests.post
    url = base_url.rstrip("/") + "/analyze"
    data: dict[str, str] = {}
    if target:
        data["target"] = target
    if sheet_name:
        data["sheet_name"] = sheet_name

    response = post_fn(
        url,
        files={"file": (filename, content)},
        data=data,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise AnalyzeError(response.status_code, _error_detail(response))
    return response.json()


def _error_detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return (getattr(response, "text", None) or f"HTTP {response.status_code}").strip()

    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        if isinstance(detail, str):
            return detail
        return str(detail)
    return str(body)
