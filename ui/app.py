"""Streamlit front end. Talks to the FastAPI service over HTTP (not dqa directly).

Run the API first, then::

    streamlit run ui/app.py

First analysis runs without a target so the response can fill the target and
sheet dropdowns; choosing either and clicking Analyze again re-runs with them.

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

DEFAULT_API_URL = "http://127.0.0.1:8000"

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

report = st.session_state.get("report")

target_options = ["(none)"]
sheet_options: list[str] = []
if report is not None:
    target_options += list(report["shape"]["column_names"])
    sheets = report["metadata"].get("available_sheets") or []
    sheet_options = list(sheets)

col_target, col_sheet = st.columns(2)
with col_target:
    selected_target = st.selectbox("Target column", target_options, index=0)
with col_sheet:
    if sheet_options:
        current_sheet = report["metadata"].get("sheet_name") if report else None
        default_idx = (
            sheet_options.index(current_sheet) if current_sheet in sheet_options else 0
        )
        selected_sheet = st.selectbox("Sheet", sheet_options, index=default_idx)
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

st.subheader("Findings")
findings = report["findings"]
if not findings:
    st.write("No findings.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "severity": f["severity"],
                    "check": f["check_name"],
                    "column": f["column"],
                    "message": f["message"],
                }
                for f in findings
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
