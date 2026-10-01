import pandas as pd
import pytest

from dqa import (
    DatasetLoadError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
    load_dataset,
)


@pytest.fixture
def workbook(tmp_path):
    """A two-sheet .xlsx file."""
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame({"b": [3], "c": [4]}).to_excel(
            writer, sheet_name="sales", index=False
        )
    return path


def test_valid_csv_loads(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("zip,name,age\n02134,Alice,30\n02134,Bob,25\n")

    result = load_dataset(path)

    assert result.row_count == 2
    assert result.column_count == 3
    assert result.file_format == "csv"
    assert result.sheet_name is None
    # leading zero survives: the loader must not let pandas infer 02134 -> 2134
    assert result.dataframe["zip"].tolist() == ["02134", "02134"]


def test_blank_cell_is_a_real_null(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text("name,age\nAlice,30\nBob,\n")

    frame = load_dataset(path).dataframe

    assert frame["age"].isna().tolist() == [False, True]
    assert frame.isna().sum().sum() == 1


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")

    with pytest.raises(EmptyDatasetError, match="File is empty"):
        load_dataset(path)


def test_headers_only_csv_raises(tmp_path):
    path = tmp_path / "headers_only.csv"
    path.write_text("name,age\n")

    with pytest.raises(EmptyDatasetError, match="no usable data"):
        load_dataset(path)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("name,age\nAlice,30\n")

    with pytest.raises(UnsupportedFileFormatError):
        load_dataset(path)


def test_excel_loads_requested_sheet(workbook):
    result = load_dataset(workbook, sheet_name="sales")

    assert result.file_format == "excel"
    assert result.sheet_name == "sales"
    assert result.available_sheets == ["summary", "sales"]
    assert result.row_count == 1
    assert result.column_count == 2


def test_excel_unknown_sheet_raises(workbook):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(workbook, sheet_name="ghost")
