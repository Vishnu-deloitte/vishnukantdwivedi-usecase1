"""
streamlit_app.py
Streamlit UI — entry point for users of the Agentic Medallion Pipeline.

Workflow:
  1. Upload raw CSV files
  2. Enter a business question
  3. Run Phase 1 -> review/approve Bronze STTM
  4. Run Phase 2 -> review/approve Silver STTM
  5. Run Phase 3 -> review/approve Gold STTM
  6. Run Phase 4 -> view the executive HTML report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from core.config import UPLOADS_DIR
from agents.orchestrator import (
    run_until_bronze_sttm,
    run_bronze_to_silver_sttm,
    run_silver_to_gold_sttm,
    run_gold_and_report,
)

st.set_page_config(page_title="Agentic Medallion Pipeline", page_icon="🏅", layout="wide")
st.title("🏅 Agentic Medallion Pipeline")
st.caption("Upload raw CSVs, ask a business question, review each STTM checkpoint, and get an executive report.")

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None


def _save_uploaded_files(files) -> list:
    saved_paths = []
    for f in files:
        out_path = UPLOADS_DIR / f.name
        out_path.write_bytes(f.getbuffer())
        saved_paths.append(str(out_path))
    return saved_paths


def _show_sttm(path: str, label: str):
    st.subheader(label)
    try:
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Source file: `{path}` — edit the CSV directly if you want to change any rule before approving.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load STTM: {e}")


# ---------------------------------------------------------------------------
# Step 1 & 2 — Upload + business question
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Raw CSV files", type=["csv"], accept_multiple_files=True)
    st.header("2. Business question")
    business_intent = st.text_area(
        "What do you want to know?",
        placeholder='e.g. "Which product category had the highest sales in Q4?"',
    )
    start_disabled = not uploaded or not business_intent
    if st.button("🚀 Run Phase 1 — Profile & Bronze STTM", disabled=start_disabled, type="primary"):
        with st.spinner("Profiling data and generating Bronze STTM..."):
            input_paths = _save_uploaded_files(uploaded)
            st.session_state.pipeline_state = run_until_bronze_sttm(input_paths, business_intent)

state = st.session_state.pipeline_state

if state is None:
    st.info("Upload CSV files and enter a business question in the sidebar to begin.")
    st.stop()

if state.get("status") == "error":
    st.error(f"Pipeline failed at `{state.get('phase')}`: {state.get('error')}")
    st.json(state)
    st.stop()

st.success(f"Run ID: `{state['run_id']}`  •  Phase: `{state['phase']}`")

# ---------------------------------------------------------------------------
# Phase 1 result — approve Bronze STTM
# ---------------------------------------------------------------------------
if state["phase"] == "awaiting_bronze_approval":
    _show_sttm(state["sttm_bronze_path"], "⏸ Review Bronze STTM (ingestion rules)")
    if st.button("✅ Approve Bronze STTM & Run Phase 2"):
        with st.spinner("Ingesting Bronze layer and generating Silver STTM..."):
            st.session_state.pipeline_state = run_bronze_to_silver_sttm(state)
            st.rerun()

# ---------------------------------------------------------------------------
# Phase 2 result — approve Silver STTM
# ---------------------------------------------------------------------------
elif state["phase"] == "awaiting_silver_approval":
    st.write("Bronze Parquet files:", state["bronze_output_paths"])
    _show_sttm(state["sttm_silver_path"], "⏸ Review Silver STTM (cleansing rules)")
    if st.button("✅ Approve Silver STTM & Run Phase 3"):
        with st.spinner("Cleansing Silver layer and generating Gold STTM..."):
            st.session_state.pipeline_state = run_silver_to_gold_sttm(state)
            st.rerun()

# ---------------------------------------------------------------------------
# Phase 3 result — approve Gold STTM
# ---------------------------------------------------------------------------
elif state["phase"] == "awaiting_gold_approval":
    st.write("Silver Parquet files:", state["silver_output_paths"])
    _show_sttm(state["sttm_gold_path"], "⏸ Review Gold STTM (materialisation rules)")
    if st.button("✅ Approve Gold STTM & Run Phase 4"):
        with st.spinner("Materialising Gold tables and generating the report..."):
            st.session_state.pipeline_state = run_gold_and_report(state)
            st.rerun()

# ---------------------------------------------------------------------------
# Phase 4 result — final report
# ---------------------------------------------------------------------------
elif state["phase"] == "complete":
    st.write("Gold Parquet files:", state["gold_output_paths"])
    st.subheader("📊 Executive Report")
    report_html = Path(state["report_path"]).read_text()
    st.components.v1.html(report_html, height=900, scrolling=True)
    st.download_button(
        "⬇️ Download report.html",
        data=report_html,
        file_name=f"report_{state['run_id'][:8]}.html",
        mime="text/html",
    )
    if st.button("🔁 Start a new run"):
        st.session_state.pipeline_state = None
        st.rerun()
