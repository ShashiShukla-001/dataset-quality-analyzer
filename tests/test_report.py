import json

import pandas as pd
import pytest

from dqa import ColumnNotFoundError
from dqa.checks import Severity
from dqa.report import analyze_dataset, analyze_file


def test_clean_dataset_has_no_error_or_warning_findings():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3", "4"],
            "b": ["2", "4", "6", "8"],
            "label": ["x", "y", "x", "y"],
        }
    )

    report = analyze_dataset(df, target="label")

    assert report.severity_counts.error == 0
    assert report.severity_counts.warning == 0


def test_messy_dataset_findings_sorted_most_serious_first():
    df = pd.DataFrame(
        {
            "const": ["same"] * 8,
            "empty": [None] * 8,
            "ok": [str(i) for i in range(8)],
            "label": ["a"] * 7 + ["b"],  # ratio 7 → imbalance WARNING
        }
    )

    report = analyze_dataset(df, target="label")
    ranks = [f.severity.rank for f in report.findings]

    assert ranks == sorted(ranks, reverse=True)
    assert report.severity_counts.error >= 1  # empty column
    assert report.findings[0].severity == Severity.ERROR


def test_no_target_adds_two_info_findings():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["3", "4"]})

    report = analyze_dataset(df, target=None)
    skipped = [f for f in report.findings if f.evidence.get("reason") == "no_target"]

    assert len(skipped) == 2
    assert {f.check_name for f in skipped} == {"class_imbalance", "possible_leakage"}
    assert all(f.severity == Severity.INFO for f in skipped)
    assert report.value_counts is None


def test_bad_target_raises_before_checks():
    df = pd.DataFrame({"a": ["1", "2"]})

    with pytest.raises(ColumnNotFoundError, match="nope"):
        analyze_dataset(df, target="nope")


def test_report_to_dict_is_json_serializable():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "3"],
            "label": ["x", "y", "x"],
        }
    )
    report = analyze_dataset(
        df,
        target="label",
        source_name="demo.csv",
        file_format="csv",
    )

    payload = json.dumps(report.to_dict())
    assert "demo.csv" in payload
    assert "findings" in payload


def test_analyze_file_csv_metadata(tmp_path):
    path = tmp_path / "people.csv"
    pd.DataFrame({"a": ["1", "2"], "label": ["x", "y"]}).to_csv(path, index=False)

    report = analyze_file(path, target="label")

    assert report.metadata.source_name.endswith("people.csv")
    assert report.metadata.file_format == "csv"
    assert report.metadata.row_count == 2
    assert report.metadata.column_count == 2
    assert report.metadata.sheet_name is None


def test_analyze_file_excel_sheet_metadata(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": ["1"], "label": ["x"]}).to_excel(
            writer, sheet_name="first", index=False
        )
        pd.DataFrame({"b": ["2"]}).to_excel(writer, sheet_name="second", index=False)

    report = analyze_file(path, sheet_name="first", target="label")

    assert report.metadata.source_name.endswith("book.xlsx")
    assert report.metadata.file_format == "excel"
    assert report.metadata.sheet_name == "first"
    assert report.metadata.available_sheets == ["first", "second"]
    assert report.metadata.row_count == 1
    assert report.metadata.column_count == 2


def test_report_surfaces_leakage_warning():
    df = pd.DataFrame(
        {
            "y": ["1", "2", "3", "4", "5"],
            "leak": ["2", "4", "6", "8", "10"],
            "noise": ["a", "b", "c", "d", "e"],
        }
    )

    report = analyze_dataset(df, target="y")
    leaks = [
        f
        for f in report.findings
        if f.check_name == "possible_leakage" and f.severity == Severity.WARNING
    ]

    assert len(leaks) == 1
    assert leaks[0].column == "leak"
