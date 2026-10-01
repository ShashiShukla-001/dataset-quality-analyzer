# Dataset Quality Analyzer

A tool that checks how good a dataset actually is before you use it for analysis or machine learning.

You give it a CSV or Excel file. It loads the data, profiles it (shape, column types, missing values, duplicates, cardinality, basic statistics, correlations), then runs quality checks on top of that (missing-data issues, constant columns, class imbalance, outliers, possible data leakage). All of it comes together as a single quality report.

**Status:** work in progress, being built step by step.

## Progress

Done so far:

- Dataset loader — CSV + Excel, with tests
- Profiling: shape & columns — in progress

Planned next:

- Profiling: data types, missing values, duplicates, cardinality, statistics, correlations
- Quality checks
- Report assembly
- FastAPI backend
- Streamlit UI

## Project layout

| Path | What lives there |
| --- | --- |
| `dqa/` | Core library (loading, profiling, checks, reporting). No framework code. |
| `api/` | FastAPI layer, built later |
| `ui/` | Streamlit layer, built later |
| `tests/` | Tests for `dqa/` |
| `data/` | Sample datasets used while developing |

The core library has no framework attached, so it can be tested and used on its own. The API and UI layers are thin wrappers over it.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -e .              # installs dqa plus pandas and openpyxl
pip install pytest            # for running the tests
```

## Running tests

```powershell
python -m pytest
```

## Usage

```python
from dqa import load_dataset

result = load_dataset("data/samples/people.csv")

result.dataframe        # the pandas DataFrame
result.row_count        # 2
result.column_count     # 3
result.file_format      # "csv"
result.sheet_name       # None for CSV, the sheet name for Excel
result.available_sheets # None for CSV, every sheet name for Excel
```

For Excel files you can pick a sheet; the first one is used by default.

```python
result = load_dataset("data/samples/book.xlsx", sheet_name="sales")
```

Every column is read as text, so values like the zip code `02134` are not silently
turned into the number `2134`. Working out what a column really holds is the
profiler's job, not the loader's. Blank cells are still read as real nulls, so
missing-value counts stay accurate.

Loading problems are raised as the tool's own exceptions rather than pandas ones,
so nothing above the loader needs to know pandas is involved:

| Exception | When |
| --- | --- |
| `UnsupportedFileFormatError` | Not a CSV or Excel file |
| `EmptyDatasetError` | The file is empty, or has headers but no rows |
| `DatasetLoadError` | The file is missing, unreadable, corrupt, or the sheet does not exist |

All three inherit from `DQAError`, so you can catch that one at the boundary.
