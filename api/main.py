"""FastAPI app: upload a dataset, get a quality report.

``MAX_UPLOAD_MB`` (25 MB) caps how much we copy into memory for analysis.
The loader reads everything as text, so peak process memory is several times
the file size; 25 MB keeps that within a comfortable single-process budget.

This check runs only while we re-read the already-received upload. FastAPI/
Starlette may have spooled the body to disk before our handler runs, so a
huge request can still fill disk first. A real upload cap belongs in front of
the app (for example nginx ``client_max_body_size``), optionally with an early
``Content-Length`` reject. Trusting ``Content-Length`` alone is not enough.
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import Request

from dqa import (
    ColumnNotFoundError,
    DatasetLoadError,
    DQAError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
    analyze_file,
)

MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

app = FastAPI(title="Dataset Quality Analyzer", version="0.1.0")


@app.exception_handler(DQAError)
async def dqa_error_handler(_request: Request, exc: DQAError) -> JSONResponse:
    if isinstance(exc, UnsupportedFileFormatError):
        status = 415
    elif isinstance(exc, (DatasetLoadError, EmptyDatasetError)):
        status = 422
    elif isinstance(exc, ColumnNotFoundError):
        status = 400
    else:
        status = 400
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    target: str | None = Form(None),
    sheet_name: str | None = Form(None),
) -> dict:
    """Analyze an uploaded CSV/Excel file. Plain ``def`` so FastAPI runs it
    in a thread pool; profiling is CPU-bound and must not block the event loop.
    """
    payload = _read_upload_limited(file)
    stream = io.BytesIO(payload)
    report = analyze_file(
        stream,
        filename=file.filename,
        target=target or None,
        sheet_name=sheet_name or None,
    )
    return report.to_dict()


def _read_upload_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {MAX_UPLOAD_MB} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)
