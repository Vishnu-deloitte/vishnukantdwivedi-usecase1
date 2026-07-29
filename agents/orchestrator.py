"""
agents/orchestrator.py
Supervisor Orchestrator — the pipeline coordinator. Receives a phase goal,
dispatches specialist agents in sequence, and hands state back to the
Streamlit UI at each human-approval gate.

Entry points (called by the Streamlit UI):
    run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState
    run_bronze_to_silver_sttm(state) -> PipelineState
    run_silver_to_gold_sttm(state) -> PipelineState
    run_gold_and_report(state) -> PipelineState
"""
import uuid
from typing import List, Optional, TypedDict

from core.audit import AuditLogger
from core.memory import document_store

from agents.profiler import profile_multiple_datasets
from agents.sttm_generator import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from agents.bronze_agent import execute_bronze
from agents.silver_agent import execute_silver
from agents.gold_agent import execute_gold
from agents.reporter import generate_report


class PipelineState(TypedDict, total=False):
    run_id: str
    business_intent: str
    input_files: List[str]

    profile_path: str

    sttm_bronze_path: str
    bronze_output_paths: List[str]

    sttm_silver_path: str
    silver_output_paths: List[str]

    sttm_gold_path: str
    gold_output_paths: List[str]

    report_path: str
    report_json_path: str

    phase: str
    status: str
    error: Optional[str]


def _new_state(input_files: List[str], business_intent: str, run_id: Optional[str] = None) -> PipelineState:
    return PipelineState(
        run_id=run_id or str(uuid.uuid4()),
        business_intent=business_intent,
        input_files=input_files,
        phase="init",
        status="running",
        error=None,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Profile raw data, generate Bronze STTM
# ---------------------------------------------------------------------------
def run_until_bronze_sttm(uploaded_files: List[str], business_intent: str) -> PipelineState:
    state = _new_state(uploaded_files, business_intent)
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    audit.phase_start("phase_1_profile_bronze_sttm", input_files=uploaded_files, business_intent=business_intent)

    try:
        profile_path = profile_multiple_datasets(uploaded_files, business_intent, run_id)
        document_store.put(run_id, "profile_path", profile_path)

        sttm_bronze_path = generate_bronze_sttm(
            profile_path=profile_path,
            business_intent=business_intent,
            run_id=run_id,
            task_description="Generate Bronze ingestion rules from the data profile.",
        )

        state.update(
            profile_path=profile_path,
            sttm_bronze_path=sttm_bronze_path,
            phase="awaiting_bronze_approval",
            status="ok",
        )
        audit.phase_complete("phase_1_profile_bronze_sttm", output_paths=[profile_path, sttm_bronze_path])
    except Exception as e:  # noqa: BLE001
        state.update(phase="phase_1_profile_bronze_sttm", status="error", error=str(e))
        audit.phase_fail("phase_1_profile_bronze_sttm", error=str(e))

    return state


# ---------------------------------------------------------------------------
# Phase 2 — Ingest Bronze, generate Silver STTM
# ---------------------------------------------------------------------------
def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    audit.human_approval("bronze_sttm", approved=True, sttm_path=state.get("sttm_bronze_path"))
    audit.phase_start("phase_2_bronze_silver_sttm")

    try:
        bronze_output_paths = execute_bronze(
            input_files=state["input_files"],
            sttm_path=state["sttm_bronze_path"],
            run_id=run_id,
            task_description="Ingest raw CSV files into Bronze Parquet using the approved STTM.",
        )
        document_store.put(run_id, "bronze_output_paths", bronze_output_paths)

        sttm_silver_path = generate_silver_sttm(
            bronze_output_paths=bronze_output_paths,
            bronze_sttm_path=state["sttm_bronze_path"],
            business_intent=state["business_intent"],
            run_id=run_id,
            task_description="Generate Silver cleansing rules from the Bronze outputs.",
        )

        state.update(
            bronze_output_paths=bronze_output_paths,
            sttm_silver_path=sttm_silver_path,
            phase="awaiting_silver_approval",
            status="ok",
        )
        audit.phase_complete("phase_2_bronze_silver_sttm", output_paths=bronze_output_paths + [sttm_silver_path])
    except Exception as e:  # noqa: BLE001
        state.update(phase="phase_2_bronze_silver_sttm", status="error", error=str(e))
        audit.phase_fail("phase_2_bronze_silver_sttm", error=str(e))

    return state


# ---------------------------------------------------------------------------
# Phase 3 — Cleanse Silver, generate Gold STTM
# ---------------------------------------------------------------------------
def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    audit.human_approval("silver_sttm", approved=True, sttm_path=state.get("sttm_silver_path"))
    audit.phase_start("phase_3_silver_gold_sttm")

    try:
        silver_output_paths = execute_silver(
            input_files=state["bronze_output_paths"],
            sttm_path=state["sttm_silver_path"],
            run_id=run_id,
            task_description="Cleanse Bronze Parquet into trusted Silver Parquet using the approved STTM.",
        )
        document_store.put(run_id, "silver_output_paths", silver_output_paths)

        sttm_gold_path = generate_gold_sttm(
            silver_output_paths=silver_output_paths,
            silver_sttm_path=state["sttm_silver_path"],
            business_intent=state["business_intent"],
            run_id=run_id,
            task_description="Generate Gold materialisation rules that will answer the business question.",
        )

        state.update(
            silver_output_paths=silver_output_paths,
            sttm_gold_path=sttm_gold_path,
            phase="awaiting_gold_approval",
            status="ok",
        )
        audit.phase_complete("phase_3_silver_gold_sttm", output_paths=silver_output_paths + [sttm_gold_path])
    except Exception as e:  # noqa: BLE001
        state.update(phase="phase_3_silver_gold_sttm", status="error", error=str(e))
        audit.phase_fail("phase_3_silver_gold_sttm", error=str(e))

    return state


# ---------------------------------------------------------------------------
# Phase 4 — Materialise Gold, generate report
# ---------------------------------------------------------------------------
def run_gold_and_report(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    audit.human_approval("gold_sttm", approved=True, sttm_path=state.get("sttm_gold_path"))
    audit.phase_start("phase_4_gold_report")

    try:
        gold_output_paths = execute_gold(
            input_files=state["silver_output_paths"],
            sttm_path=state["sttm_gold_path"],
            business_intent=state["business_intent"],
            run_id=run_id,
            task_description="Materialise analytics-ready Gold Parquet tables using the approved STTM.",
        )
        document_store.put(run_id, "gold_output_paths", gold_output_paths)

        report_html_path = generate_report(
            gold_files=gold_output_paths,
            business_intent=state["business_intent"],
            run_id=run_id,
            task_description="Answer the business question with SQL over the Gold tables and produce an HTML report.",
        )
        report_json_path = report_html_path.replace(".html", ".json")

        state.update(
            gold_output_paths=gold_output_paths,
            report_path=report_html_path,
            report_json_path=report_json_path,
            phase="complete",
            status="ok",
        )
        audit.phase_complete("phase_4_gold_report", output_paths=gold_output_paths + [report_html_path])
    except Exception as e:  # noqa: BLE001
        state.update(phase="phase_4_gold_report", status="error", error=str(e))
        audit.phase_fail("phase_4_gold_report", error=str(e))

    return state
