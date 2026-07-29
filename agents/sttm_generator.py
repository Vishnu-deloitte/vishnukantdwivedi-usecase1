"""
agents/sttm_generator.py
Unified STTM (Source-to-Target Mapping) agent. The orchestrator tells this
agent which layer to generate (bronze / silver / gold); it inspects the
relevant context and calls the matching generation tool.

STTM CSV columns:
  source_schema, source_table, source_column,
  target_schema, target_table, target_column,
  transformation_type, transformation_logic
"""
import csv
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import STTM_DIR, make_llm
from core.observability import AgentTrace

STTM_COLUMNS = [
    "source_schema", "source_table", "source_column",
    "target_schema", "target_table", "target_column",
    "transformation_type", "transformation_logic",
]


def _write_sttm_csv(rows: List[dict], out_path: Path) -> str:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STTM_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in STTM_COLUMNS})
    return str(out_path)


def _parse_llm_json_rows(text: str) -> List[dict]:
    """Best-effort extraction of a JSON array of STTM row dicts from LLM output."""
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return []


def _run_sttm_loop(system_prompt: str, user_prompt: str, tools: list, trace: AgentTrace) -> List[dict]:
    llm = make_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    rows: List[dict] = []
    for _ in range(3):  # 2 steps needed (inspect→generate)
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)
        if ai_msg.content:
            trace.log_step("ai_reasoning", str(ai_msg.content))
            parsed = _parse_llm_json_rows(str(ai_msg.content))
            if parsed:
                rows = parsed
        if not getattr(ai_msg, "tool_calls", None):
            break
        for call in ai_msg.tool_calls:
            fn = tool_map[call["name"]]
            result = fn.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return rows


# ---------------------------------------------------------------------------
# BRONZE
# ---------------------------------------------------------------------------
def generate_bronze_sttm(profile_path: str, business_intent: str, run_id: str, task_description: str = "") -> str:
    """Generate sttm_bronze_{run_id[:8]}.csv from the data profile."""
    trace = AgentTrace("sttm_generator_bronze", run_id, input_data={"profile_path": profile_path})
    profile = json.loads(Path(profile_path).read_text())
    scratchpad = {"profile": profile}

    @tool
    def inspect_context_tool(confirmation: str = "inspect") -> str:
        """Preview the data profile for the requested layer (Bronze ingestion)."""
        trace.log_tool_call("inspect_context_tool")
        content = json.dumps(scratchpad["profile"].get("full_profile", {}))
        trace.log_tool_result("inspect_context_tool", content[:2000])
        return content

    @tool
    def generate_bronze_sttm_tool(rules_json: str) -> str:
        """Write the Bronze STTM rules. rules_json must be a JSON array of row objects with
        source_schema, source_table, source_column, target_schema, target_table, target_column,
        transformation_type (RENAME/CAST/METADATA), transformation_logic."""
        trace.log_tool_call("generate_bronze_sttm_tool")
        try:
            rows = json.loads(rules_json)
        except json.JSONDecodeError:
            rows = []
        scratchpad["rows"] = rows
        return json.dumps({"rows_received": len(rows)})

    tools = [inspect_context_tool, generate_bronze_sttm_tool]
    system_prompt = (
        "You are the STTM agent generating BRONZE ingestion rules: CSV -> Parquet.\n"
        "For every source column produce a row with transformation_type one of "
        "RENAME (snake_case target name), CAST (to_numeric/to_datetime/str), or PASSTHROUGH.\n"
        "Also add two METADATA rows per source table: _load_timestamp and _source_file.\n"
        "First call inspect_context_tool. Then reply with ONLY a JSON array of row objects "
        f"matching columns {STTM_COLUMNS}. target_schema='bronze'."
    )
    user_prompt = f"Business intent: {business_intent}\nTask: {task_description}\nGenerate the Bronze STTM."

    rows = _run_sttm_loop(system_prompt, user_prompt, tools, trace)
    if not rows:
        rows = _fallback_bronze_rows(profile)

    out_path = STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv"
    path_str = _write_sttm_csv(rows, out_path)
    trace.finish(output={"sttm_path": path_str, "row_count": len(rows)})
    return path_str


def _fallback_bronze_rows(profile: dict) -> List[dict]:
    """Deterministic fallback if the LLM doesn't return valid JSON."""
    rows = []
    for fname, info in profile.get("full_profile", {}).items():
        table = Path(fname).stem
        for col in info.get("columns", {}):
            rows.append({
                "source_schema": "raw", "source_table": table, "source_column": col,
                "target_schema": "bronze", "target_table": f"{table}_bronze", "target_column": col.strip().lower().replace(" ", "_"),
                "transformation_type": "RENAME", "transformation_logic": f"rename to snake_case: {col}",
            })
        for meta_col in ("_load_timestamp", "_source_file"):
            rows.append({
                "source_schema": "raw", "source_table": table, "source_column": "",
                "target_schema": "bronze", "target_table": f"{table}_bronze", "target_column": meta_col,
                "transformation_type": "METADATA",
                "transformation_logic": "UTC ISO timestamp at ingestion" if meta_col == "_load_timestamp" else "original source file path",
            })
    return rows


# ---------------------------------------------------------------------------
# SILVER
# ---------------------------------------------------------------------------
def generate_silver_sttm(bronze_output_paths: List[str], bronze_sttm_path: str, business_intent: str,
                          run_id: str, task_description: str = "") -> str:
    """Generate sttm_silver_{run_id[:8]}.csv from Bronze outputs + Bronze STTM."""
    trace = AgentTrace("sttm_generator_silver", run_id, input_data={"bronze_output_paths": bronze_output_paths})
    bronze_sttm_df = pd.read_csv(bronze_sttm_path) if Path(bronze_sttm_path).exists() else pd.DataFrame()
    scratchpad: dict = {}

    @tool
    def inspect_context_tool(confirmation: str = "inspect") -> str:
        """Preview Bronze Parquet schemas and null counts, plus the approved Bronze STTM."""
        trace.log_tool_call("inspect_context_tool")
        schemas = {}
        for p in bronze_output_paths:
            df = pd.read_parquet(p)
            schemas[Path(p).name] = {
                "columns": list(df.columns),
                "null_counts": df.isna().sum().to_dict(),
                "row_count": len(df),
            }
        scratchpad["schemas"] = schemas
        content = json.dumps(schemas, default=str)
        trace.log_tool_result("inspect_context_tool", content[:2000])
        return content

    @tool
    def generate_silver_sttm_tool(rules_json: str) -> str:
        """Write the Silver STTM rules (nulls, dedup, type standardisation, date formatting, surrogate key)."""
        trace.log_tool_call("generate_silver_sttm_tool")
        try:
            rows = json.loads(rules_json)
        except json.JSONDecodeError:
            rows = []
        scratchpad["rows"] = rows
        return json.dumps({"rows_received": len(rows)})

    tools = [inspect_context_tool, generate_silver_sttm_tool]
    system_prompt = (
        "You are the STTM agent generating SILVER cleansing rules: Bronze Parquet -> Silver Parquet.\n"
        "For each retained column, produce a row with transformation_type one of "
        "NULL_HANDLING (dropna/fillna:mean/median/mode/constant), DEDUP, TYPE_CAST, DATE_FORMAT (YYYY-MM-DD), "
        "TEXT_NORMALIZE (strip/lower/upper), or SURROGATE_KEY (first row per table: pk_<table>_silver_id).\n"
        "First call inspect_context_tool. Then reply with ONLY a JSON array of row objects "
        f"matching columns {STTM_COLUMNS}. target_schema='silver'."
    )
    user_prompt = f"Business intent: {business_intent}\nTask: {task_description}\nGenerate the Silver STTM."

    rows = _run_sttm_loop(system_prompt, user_prompt, tools, trace)
    if not rows:
        rows = _fallback_silver_rows(bronze_output_paths)

    out_path = STTM_DIR / f"sttm_silver_{run_id[:8]}.csv"
    path_str = _write_sttm_csv(rows, out_path)
    trace.finish(output={"sttm_path": path_str, "row_count": len(rows)})
    return path_str


def _fallback_silver_rows(bronze_output_paths: List[str]) -> List[dict]:
    rows = []
    for p in bronze_output_paths:
        df = pd.read_parquet(p)
        table = Path(p).stem.replace("_bronze", "")
        rows.append({
            "source_schema": "bronze", "source_table": Path(p).stem, "source_column": "",
            "target_schema": "silver", "target_table": f"{table}_silver", "target_column": f"pk_{table}_silver_id",
            "transformation_type": "SURROGATE_KEY", "transformation_logic": "monotonic integer id, first column",
        })
        for col in df.columns:
            if col.startswith("_"):
                continue
            t_type = "DATE_FORMAT" if "date" in col.lower() else ("NULL_HANDLING" if df[col].isna().any() else "TYPE_CAST")
            rows.append({
                "source_schema": "bronze", "source_table": Path(p).stem, "source_column": col,
                "target_schema": "silver", "target_table": f"{table}_silver", "target_column": col,
                "transformation_type": t_type,
                "transformation_logic": "standardise to YYYY-MM-DD" if t_type == "DATE_FORMAT"
                    else ("dropna" if t_type == "NULL_HANDLING" else "cast to correct dtype"),
            })
    return rows


# ---------------------------------------------------------------------------
# GOLD
# ---------------------------------------------------------------------------
def generate_gold_sttm(silver_output_paths: List[str], silver_sttm_path: str, business_intent: str,
                        run_id: str, task_description: str = "") -> str:
    """Generate sttm_gold_{run_id[:8]}.csv from Silver outputs + business intent."""
    trace = AgentTrace("sttm_generator_gold", run_id, input_data={"silver_output_paths": silver_output_paths})
    scratchpad: dict = {}

    @tool
    def inspect_context_tool(confirmation: str = "inspect") -> str:
        """Preview Silver schemas grouped by table, to design Gold joins/aggregations."""
        trace.log_tool_call("inspect_context_tool")
        schemas = {}
        for p in silver_output_paths:
            df = pd.read_parquet(p)
            schemas[Path(p).name] = {"columns": list(df.columns), "row_count": len(df)}
        scratchpad["schemas"] = schemas
        content = json.dumps(schemas, default=str)
        trace.log_tool_result("inspect_context_tool", content[:2000])
        return content

    @tool
    def generate_gold_sttm_tool(rules_json: str) -> str:
        """Write the Gold STTM rules (joins, aggregations, surrogate key) that answer the business question."""
        trace.log_tool_call("generate_gold_sttm_tool")
        try:
            rows = json.loads(rules_json)
        except json.JSONDecodeError:
            rows = []
        scratchpad["rows"] = rows
        return json.dumps({"rows_received": len(rows)})

    tools = [inspect_context_tool, generate_gold_sttm_tool]
    system_prompt = (
        "You are the STTM agent generating GOLD materialisation rules: Silver Parquet -> Gold Parquet.\n"
        "Design one or more Gold target tables that let the Reporter agent answer the business question. "
        "transformation_type is one of JOIN (outer join on matching _id columns), AGGREGATE "
        "(sum/avg/count/max/min via groupby), RENAME, or SURROGATE_KEY (first row per table: pk_gold_id).\n"
        "First call inspect_context_tool. Then reply with ONLY a JSON array of row objects "
        f"matching columns {STTM_COLUMNS}. target_schema='gold'."
    )
    user_prompt = f"Business intent: {business_intent}\nTask: {task_description}\nGenerate the Gold STTM."

    rows = _run_sttm_loop(system_prompt, user_prompt, tools, trace)
    if not rows:
        rows = _fallback_gold_rows(silver_output_paths)

    out_path = STTM_DIR / f"sttm_gold_{run_id[:8]}.csv"
    path_str = _write_sttm_csv(rows, out_path)
    trace.finish(output={"sttm_path": path_str, "row_count": len(rows)})
    return path_str


def _fallback_gold_rows(silver_output_paths: List[str]) -> List[dict]:
    rows = []
    target_table = "gold_summary"
    rows.append({
        "source_schema": "silver", "source_table": "", "source_column": "",
        "target_schema": "gold", "target_table": target_table, "target_column": "pk_gold_id",
        "transformation_type": "SURROGATE_KEY", "transformation_logic": "monotonic integer id, first column",
    })
    for p in silver_output_paths:
        df = pd.read_parquet(p)
        table = Path(p).stem
        for col in df.columns:
            if col.startswith("pk_"):
                continue
            agg = "SUM" if pd.api.types.is_numeric_dtype(df[col]) else "GROUP_BY"
            rows.append({
                "source_schema": "silver", "source_table": table, "source_column": col,
                "target_schema": "gold", "target_table": target_table, "target_column": col,
                "transformation_type": "AGGREGATE", "transformation_logic": agg,
            })
    return rows
