"""Streamlit front end. Talks to the FastAPI service over HTTP (not dqa directly).

Run the API first, then::

    streamlit run ui/app.py

First analysis runs without a target. The page then reruns so the target and
sheet dropdowns are filled from that response; choosing either and clicking
Analyze again sends the choice.

When the upload’s name or size changes, the cached report is dropped so the
target/sheet dropdowns cannot keep listing columns from a previous file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.client import AnalyzeError, analyze_upload
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

DEFAULT_API_URL = "http://127.0.0.1:8000"


def _show_table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.set_page_config(page_title="Dataset Quality Analyzer", layout="wide")
st.title("Dataset Quality Analyzer")

api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
uploaded = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xlsm", "xls"],
)

# Invalidate the report when the upload changes (name + size, not a content
# hash — enough to catch a different file, cheap enough to check every rerun).
if uploaded is None:
    st.session_state.pop("upload_identity", None)
    st.session_state.pop("report", None)
else:
    identity = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_identity") != identity:
        st.session_state["upload_identity"] = identity
        st.session_state.pop("report", None)
        st.session_state.pop("error", None)
        # Drop widget state too, or a column picked for the previous file
        # stays selected and Streamlit rejects it once it is no longer an option.
        st.session_state.pop("target_column", None)
        st.session_state.pop("sheet_name", None)

report = st.session_state.get("report")

target_options = ["(none)"]
sheet_options: list[str] = []
if report is not None:
    target_options += list(report["shape"]["column_names"])
    sheets = report["metadata"].get("available_sheets") or []
    sheet_options = list(sheets)

col_target, col_sheet = st.columns(2)
with col_target:
    selected_target = st.selectbox("Target column", target_options, key="target_column")
with col_sheet:
    if sheet_options:
        selected_sheet = st.selectbox("Sheet", sheet_options, key="sheet_name")
    else:
        selected_sheet = None
        st.selectbox("Sheet", ["(upload an Excel file)"], disabled=True)

if st.button("Analyze", type="primary", disabled=uploaded is None):
    assert uploaded is not None
    content = uploaded.getvalue()
    st.session_state["file_name"] = uploaded.name
    st.session_state["file_bytes"] = content

    target = None if selected_target == "(none)" else selected_target
    # First run (no report yet): omit sheet so the API picks the default;
    # later runs use the dropdown filled from available_sheets.
    sheet_name = selected_sheet if report is not None else None

    try:
        st.session_state["report"] = analyze_upload(
            api_url,
            uploaded.name,
            content,
            target=target,
            sheet_name=sheet_name,
        )
        st.session_state.pop("error", None)
    except AnalyzeError as exc:
        st.session_state["error"] = exc.detail
        st.session_state.pop("report", None)
    except requests.RequestException as exc:
        st.session_state["error"] = f"Could not reach API: {exc}"
        st.session_state.pop("report", None)
    # The dropdowns above were drawn from the previous report. Rerun so they
    # pick up the columns (or the error) before the user tries to choose one.
    st.rerun()

if error := st.session_state.get("error"):
    st.error(error)

report = st.session_state.get("report")
if report is None:
    st.stop()

counts = report["severity_counts"]
c1, c2, c3 = st.columns(3)
c1.metric("Errors", counts["error"])
c2.metric("Warnings", counts["warning"])
c3.metric("Info", counts["info"])

st.subheader("Overview")
_show_table(overview_rows(report))

st.subheader("Findings")
finding_rows = findings_rows(report)
if not finding_rows:
    st.write("No findings.")
else:
    _show_table(finding_rows)

with st.expander("Column types"):
    _show_table(dtype_rows(report))
with st.expander("Missing values"):
    _show_table(missing_rows(report))
with st.expander("Cardinality"):
    _show_table(cardinality_rows(report))
with st.expander("Statistics"):
    _show_table(statistics_rows(report))
with st.expander("Correlations"):
    pairs = correlation_rows(report)
    if not pairs:
        st.caption("No numeric pairs.")
    else:
        _show_table(pairs)

counts_rows = value_count_rows(report)
if counts_rows is not None:
    title = f"Value counts ({report['value_counts']['column_name']})"
    if report["value_counts"].get("truncated"):
        title += " — truncated"
    with st.expander(title):
        _show_table(counts_rows)
