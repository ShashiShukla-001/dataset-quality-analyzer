"""Loading CSV and Excel files into pandas DataFrames.

This is the only module allowed to know about pandas-level errors; it
translates them into the exceptions defined in :mod:`dqa.exceptions`.

Every column is read as string (``dtype=str``). Letting pandas infer dtypes
would discard information before profiling ever sees it — a zip code like
"02134" would arrive as the integer 2134. Deciding what a column really
holds is the profiler's job, not the loader's. Blank cells are still read as
real nulls, so missing-value counts stay accurate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

from dqa.exceptions import (
    DatasetLoadError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
)

CSV_EXTENSIONS = {".csv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}

FileSource = str | Path | IO[bytes]


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    source_name: str
    file_format: str
    sheet_name: str | None
    available_sheets: list[str] | None
    row_count: int
    column_count: int


def load_dataset(
    file: FileSource,
    *,
    sheet_name: str | None = None,
    filename: str | None = None,
) -> LoadResult:
    """Load a CSV or Excel file.

    `file` may be a path or an open binary file object (as uploaded through
    FastAPI or Streamlit). For Excel files, `sheet_name` selects a sheet;
    the first sheet is used when it is not given.

    When `file` is a stream with no usable ``.name`` (common for upload
    buffers), pass ``filename=`` so the format can still be detected from
    the extension. A stream's own ``.name`` wins when present.
    """
    source_name = _source_name(file, filename=filename)
    extension = Path(source_name).suffix.lower()

    if extension in CSV_EXTENSIONS:
        return _load_csv(file, source_name)
    if extension in EXCEL_EXTENSIONS:
        return _load_excel(file, source_name, sheet_name)
    raise UnsupportedFileFormatError(path=source_name, extension=extension)


def _source_name(file: FileSource, *, filename: str | None = None) -> str:
    if isinstance(file, (str, Path)):
        return str(file)

    name = getattr(file, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(filename, str) and filename:
        return filename
    raise UnsupportedFileFormatError(
        "Unsupported file format; the uploaded file has no name to identify it by"
    )


def _load_csv(file: FileSource, source_name: str) -> LoadResult:
    try:
        frame = pd.read_csv(file, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("File is empty", path=source_name) from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise DatasetLoadError(f"Could not read CSV file: {exc}", path=source_name) from exc

    return _build_result(frame, source_name, "csv", None, None)


def _load_excel(
    file: FileSource, source_name: str, sheet_name: str | None
) -> LoadResult:
    try:
        workbook = pd.ExcelFile(file)
    except (ValueError, ImportError, OSError) as exc:
        raise DatasetLoadError(f"Could not open Excel file: {exc}", path=source_name) from exc

    with workbook:
        available_sheets = list(workbook.sheet_names)
        if not available_sheets:
            raise EmptyDatasetError("Excel file contains no sheets", path=source_name)

        selected_sheet = available_sheets[0] if sheet_name is None else sheet_name
        if selected_sheet not in available_sheets:
            raise DatasetLoadError(
                f"Sheet {selected_sheet!r} not found; "
                f"available sheets: {', '.join(available_sheets)}",
                path=source_name,
            )

        try:
            frame = workbook.parse(selected_sheet, dtype=str)
        except (ValueError, OSError) as exc:
            raise DatasetLoadError(
                f"Could not parse sheet {selected_sheet!r}: {exc}", path=source_name
            ) from exc

    return _build_result(frame, source_name, "excel", selected_sheet, available_sheets)


def _build_result(
    frame: pd.DataFrame,
    source_name: str,
    file_format: str,
    sheet_name: str | None,
    available_sheets: list[str] | None,
) -> LoadResult:
    row_count = len(frame)
    column_count = len(frame.columns)
    if row_count == 0 or column_count == 0:
        raise EmptyDatasetError(path=source_name)

    return LoadResult(
        dataframe=frame,
        source_name=source_name,
        file_format=file_format,
        sheet_name=sheet_name,
        available_sheets=available_sheets,
        row_count=row_count,
        column_count=column_count,
    )
