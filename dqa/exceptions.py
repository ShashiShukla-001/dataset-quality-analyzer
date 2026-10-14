"""Domain-specific exceptions for Dataset Quality Analyzer.

All layers above the loader (tests, FastAPI, Streamlit) should catch and
handle only these exceptions — never pandas or low-level I/O errors directly.
"""


class DQAError(Exception):
    """Base exception for all Dataset Quality Analyzer errors."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class UnsupportedFileFormatError(DQAError):
    """Raised when the file extension or format is not CSV or Excel."""

    def __init__(
        self,
        message: str = "Unsupported file format; expected CSV or Excel",
        *,
        path: str | None = None,
        extension: str | None = None,
    ) -> None:
        super().__init__(message, path=path)
        self.extension = extension


class DatasetLoadError(DQAError):
    """Raised when a file cannot be read or parsed (corrupt, malformed,
    truncated, bad encoding, permission denied, etc)."""

    def __init__(
        self,
        message: str = "Could not read or parse file",
        *,
        path: str | None = None,
    ) -> None:
        super().__init__(message, path=path)


class EmptyDatasetError(DQAError):
    """Raised when a dataset has no rows, no columns, or both."""

    def __init__(
        self,
        message: str = "Dataset contains no usable data",
        *,
        path: str | None = None,
    ) -> None:
        super().__init__(message, path=path)


class ColumnNotFoundError(DQAError):
    """Raised when a requested column name is not present in the dataset."""

    def __init__(
        self,
        column: str,
        *,
        available: list[str] | None = None,
    ) -> None:
        if available:
            message = (
                f"Column {column!r} not found; "
                f"available columns: {', '.join(available)}"
            )
        else:
            message = f"Column {column!r} not found in dataset"
        super().__init__(message)
        self.column = column
        self.available = available