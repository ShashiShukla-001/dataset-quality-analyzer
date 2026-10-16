"""Dataset Quality Analyzer — core library."""

from dqa.checks import CheckConfig, Finding, Severity
from dqa.exceptions import (
    ColumnNotFoundError,
    DatasetLoadError,
    DQAError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
)
from dqa.loader import LoadResult, load_dataset
from dqa.report import QualityReport, analyze_dataset, analyze_file

__all__ = [
    "CheckConfig",
    "ColumnNotFoundError",
    "DQAError",
    "DatasetLoadError",
    "EmptyDatasetError",
    "Finding",
    "LoadResult",
    "QualityReport",
    "Severity",
    "UnsupportedFileFormatError",
    "analyze_dataset",
    "analyze_file",
    "load_dataset",
]
