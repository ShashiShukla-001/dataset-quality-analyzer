"""Turn a quality-report dict into tables for the UI.

Pure functions, no Streamlit, so the mapping from API JSON to what the page
shows can be tested without a browser.
"""

from __future__ import annotations

from typing import Any


def findings_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "severity": finding["severity"],
            "check": finding["check_name"],
            "column": finding["column"],
            "message": finding["message"],
        }
        for finding in report["findings"]
    ]


def overview_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """File facts plus the dataset-wide rates a user scans first."""
    metadata = report["metadata"]
    shape = report["shape"]
    missing = report["missing"]
    duplicates = report["duplicates"]
    rows = [
        {"item": "Source", "value": metadata.get("source_name") or ""},
        {"item": "Format", "value": metadata.get("file_format") or ""},
        {"item": "Rows", "value": str(shape["row_count"])},
        {"item": "Columns", "value": str(shape["column_count"])},
        {"item": "Overall missing rate", "value": str(missing["overall_null_rate"])},
        {"item": "Duplicate row count", "value": str(duplicates["duplicate_row_count"])},
        {"item": "Duplicate rate", "value": str(duplicates["duplicate_rate"])},
    ]
    if metadata.get("sheet_name"):
        rows.append({"item": "Sheet", "value": metadata["sheet_name"]})
    duplicate_names = shape.get("duplicate_column_names") or []
    if duplicate_names:
        rows.append({"item": "Duplicate column names", "value": ", ".join(duplicate_names)})
    return rows


def dtype_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column": column["column_name"],
            "inferred_type": column["inferred_type"],
            "match_rate": column["match_rate"],
            "null_count": column["null_count"],
        }
        for column in report["dtypes"]["columns"]
    ]


def missing_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column": column["column_name"],
            "null_count": column["null_count"],
            "null_rate": column["null_rate"],
        }
        for column in report["missing"]["columns"]
    ]


def cardinality_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column": column["column_name"],
            "non_missing_count": column["non_missing_count"],
            "distinct_count": column["distinct_count"],
            "distinct_rate": column["distinct_rate"],
        }
        for column in report["cardinality"]["columns"]
    ]


def statistics_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column": column["column_name"],
            "numeric_count": column["numeric_count"],
            "min": column["min"],
            "max": column["max"],
            "mean": column["mean"],
            "median": column["median"],
            "std": column["std"],
            "q1": column["q1"],
            "q3": column["q3"],
        }
        for column in report["statistics"]["columns"]
    ]


def correlation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column_a": pair["column_a"],
            "column_b": pair["column_b"],
            "pair_count": pair["pair_count"],
            "pearson": pair["pearson"],
        }
        for pair in report["correlations"]["pairs"]
    ]


def value_count_rows(report: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Rows for the target frequency table, or None when no target was set."""
    profile = report.get("value_counts")
    if not profile:
        return None
    return [
        {"value": item["value"], "count": item["count"]}
        for item in profile["values"]
    ]
