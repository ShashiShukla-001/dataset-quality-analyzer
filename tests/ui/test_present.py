from ui.present import (
    cardinality_rows,
    correlation_rows,
    dtype_rows,
    findings_rows,
    missing_rows,
    overview_rows,
    statistics_rows,
    value_count_rows,
)


def _report(**overrides):
    report = {
        "metadata": {
            "source_name": "people.csv",
            "file_format": "csv",
            "sheet_name": None,
            "available_sheets": None,
            "row_count": 2,
            "column_count": 2,
        },
        "shape": {
            "row_count": 2,
            "column_count": 2,
            "column_names": ["age", "label"],
            "duplicate_column_names": [],
        },
        "dtypes": {
            "columns": [
                {
                    "column_name": "age",
                    "inferred_type": "integer",
                    "match_rate": 1.0,
                    "null_count": 0,
                }
            ]
        },
        "missing": {
            "columns": [{"column_name": "age", "null_count": 0, "null_rate": 0.0}],
            "overall_null_rate": 0.0,
        },
        "duplicates": {"duplicate_row_count": 0, "duplicate_rate": 0.0},
        "cardinality": {
            "columns": [
                {
                    "column_name": "age",
                    "non_missing_count": 2,
                    "distinct_count": 2,
                    "distinct_rate": 1.0,
                }
            ]
        },
        "statistics": {
            "columns": [
                {
                    "column_name": "age",
                    "numeric_count": 2,
                    "min": 1.0,
                    "max": 2.0,
                    "mean": 1.5,
                    "median": 1.5,
                    "std": 0.7,
                    "q1": 1.0,
                    "q3": 2.0,
                }
            ]
        },
        "correlations": {
            "pairs": [
                {
                    "column_a": "age",
                    "column_b": "score",
                    "pair_count": 2,
                    "pearson": 0.5,
                }
            ]
        },
        "value_counts": None,
        "findings": [
            {
                "severity": "warning",
                "check_name": "constant_column",
                "column": "age",
                "message": "constant",
            }
        ],
    }
    report.update(overrides)
    return report


def test_findings_rows_keep_api_order():
    rows = findings_rows(_report())

    assert rows == [
        {
            "severity": "warning",
            "check": "constant_column",
            "column": "age",
            "message": "constant",
        }
    ]


def test_overview_includes_sheet_and_duplicate_names_when_present():
    report = _report()
    report["metadata"]["sheet_name"] = "sales"
    report["shape"]["duplicate_column_names"] = ["id", "id"]

    items = {row["item"]: row["value"] for row in overview_rows(report)}

    assert items["Source"] == "people.csv"
    assert items["Format"] == "csv"
    assert items["Rows"] == "2"
    assert items["Sheet"] == "sales"
    assert items["Duplicate column names"] == "id, id"


def test_overview_omits_sheet_for_csv():
    items = {row["item"] for row in overview_rows(_report())}

    assert "Sheet" not in items
    assert "Duplicate column names" not in items


def test_profile_tables_use_column_names():
    report = _report()

    assert dtype_rows(report)[0]["inferred_type"] == "integer"
    assert missing_rows(report)[0]["null_rate"] == 0.0
    assert cardinality_rows(report)[0]["distinct_count"] == 2
    assert statistics_rows(report)[0]["mean"] == 1.5
    assert correlation_rows(report)[0]["pearson"] == 0.5


def test_value_counts_absent_without_target():
    assert value_count_rows(_report()) is None


def test_value_counts_rows_when_target_set():
    report = _report(
        value_counts={
            "column_name": "label",
            "values": [{"value": "x", "count": 2}, {"value": "y", "count": 1}],
        }
    )

    assert value_count_rows(report) == [
        {"value": "x", "count": 2},
        {"value": "y", "count": 1},
    ]
