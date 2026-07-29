"""
agents/silver_agent.py
Silver layer cleansing agent — Bronze Parquet -> Silver Parquet.
Null handling, deduplication, type standardisation, date formatting
(-> YYYY-MM-DD), surrogate key pk_*_silver_id injected as first column,
column filtering to STTM-approved columns only.
"""
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import SILVER_DIR, make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Silver cleansing agent.
Follow THINK -> INSPECT -> PLAN -> ACT -> VERIFY:
1. Call inspect_task_tool to preview Bronze schemas, null counts, and the approved Silver STTM.
2. State a short plan for null handling, dedup, and type standardisation.
3. Call silver_ingestion_tool to execute the cleansing.
4. Reply with a one-line confirmation. Do not call more tools after that.
"""


def _apply_silver_rules(input_files: List[str], sttm_path: str, run_id: str) -> List[str]:
    sttm = pd.read_csv(sttm_path)
    output_paths = []

    for parquet_path in input_files:
        df = pd.read_parquet(parquet_path)
        source_table = Path(parquet_path).stem
        rules = sttm[(sttm["source_table"] == source_table) | (sttm["source_table"] == "")]
        target_table = rules["target_table"].iloc[0] if len(rules) and pd.notna(rules["target_table"].iloc[0]) \
            else source_table.replace("_bronze", "_silver")

        keep_cols = [c for c in rules["target_column"].dropna().unique().tolist()
                     if not str(c).startswith("pk_")]
        keep_cols = [c for c in keep_cols if c in df.columns] or [c for c in df.columns if not c.startswith("_")]
        df = df[keep_cols].copy()

        for _, r in rules.iterrows():
            t_type = r.get("transformation_type")
            col = r.get("target_column")
            if col not in df.columns:
                continue
            if t_type == "NULL_HANDLING":
                logic = str(r.get("transformation_logic", "")).lower()
                if "mean" in logic and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif "median" in logic and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif "mode" in logic:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "")
                else:
                    df = df.dropna(subset=[col])
            elif t_type == "DATE_FORMAT":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            elif t_type == "TEXT_NORMALIZE":
                logic = str(r.get("transformation_logic", "")).lower()
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.strip()
                    if "lower" in logic:
                        df[col] = df[col].str.lower()
                    elif "upper" in logic:
                        df[col] = df[col].str.upper()
            elif t_type == "TYPE_CAST":
                logic = str(r.get("transformation_logic", "")).lower()
                if "numeric" in logic:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.drop_duplicates()
        base_name = target_table if target_table.endswith("_silver") else f"{target_table}_silver"
        pk_col = f"pk_{base_name}_id"
        df.insert(0, pk_col, range(1, len(df) + 1))

        out_path = SILVER_DIR / f"{target_table}.parquet"
        df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def execute_silver(input_files: List[str], sttm_path: str, run_id: str, task_description: str = "") -> List[str]:
    """Runs the Silver agent. Returns the list of written Silver Parquet paths."""
    trace = AgentTrace("silver_agent", run_id, input_data={"input_files": input_files, "sttm_path": sttm_path})
    scratchpad: dict = {}

    @tool
    def inspect_task_tool(confirmation: str = "inspect") -> str:
        """Preview Bronze Parquet schemas, null counts, and the approved Silver STTM rules."""
        trace.log_tool_call("inspect_task_tool")
        schemas = {}
        for p in input_files:
            df = pd.read_parquet(p)
            schemas[Path(p).name] = {"columns": list(df.columns), "null_counts": df.isna().sum().to_dict()}
        sttm_preview = pd.read_csv(sttm_path).to_dict(orient="records")
        content = json.dumps({"bronze_schemas": schemas, "sttm_rules": sttm_preview[:50]}, default=str)
        trace.log_tool_result("inspect_task_tool", content[:2000])
        return content

    @tool
    def silver_ingestion_tool(confirmation: str = "execute") -> str:
        """Apply the approved Silver STTM cleansing rules and write Parquet files."""
        trace.log_tool_call("silver_ingestion_tool")
        paths = _apply_silver_rules(input_files, sttm_path, run_id)
        scratchpad["output_paths"] = paths
        content = json.dumps(paths)
        trace.log_tool_result("silver_ingestion_tool", content)
        return content

    tools = [inspect_task_tool, silver_ingestion_tool]
    llm = make_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {task_description}\nInput files: {input_files}\nBegin."),
    ]

    for _ in range(3):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)
        if ai_msg.content:
            trace.log_step("ai_reasoning", str(ai_msg.content))
        if not getattr(ai_msg, "tool_calls", None):
            break
        for call in ai_msg.tool_calls:
            fn = tool_map[call["name"]]
            result = fn.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    output_paths = scratchpad.get("output_paths") or _apply_silver_rules(input_files, sttm_path, run_id)
    trace.finish(output={"output_paths": output_paths})
    return output_paths
