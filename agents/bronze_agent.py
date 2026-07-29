"""
agents/bronze_agent.py
Bronze layer ingestion agent — CSV -> Parquet, columns renamed and typed,
_load_timestamp / _source_file metadata injected. No null handling here;
Bronze is a faithful raw copy.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import BRONZE_DIR, make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Bronze ingestion agent.
Follow THINK -> INSPECT -> PLAN -> ACT -> VERIFY:
1. Call inspect_task_tool to preview the input CSVs and the approved Bronze STTM rules.
2. State a short plan (which columns get renamed/cast, which metadata columns get added).
3. Call bronze_ingestion_tool to execute the ingestion.
4. When you get the result, reply with a one-line confirmation. Do not call more tools after that.
"""


def _apply_bronze_rules(input_files: List[str], sttm_path: str, run_id: str) -> List[str]:
    sttm = pd.read_csv(sttm_path)
    output_paths = []
    load_ts = datetime.now(timezone.utc).isoformat()

    for csv_path in input_files:
        df = pd.read_csv(csv_path)
        table_stem = Path(csv_path).stem
        rules = sttm[(sttm["source_table"] == table_stem) | (sttm["source_table"] == "")]

        rename_map = {}
        cast_map = {}
        for _, r in rules.iterrows():
            if r.get("transformation_type") == "RENAME" and pd.notna(r.get("source_column")) and r.get("source_column") != "":
                rename_map[r["source_column"]] = r["target_column"]
            if r.get("transformation_type") == "CAST" and pd.notna(r.get("source_column")):
                cast_map[r["source_column"]] = str(r.get("transformation_logic", ""))

        if rename_map:
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        for col, logic in cast_map.items():
            target_col = rename_map.get(col, col)
            if target_col not in df.columns:
                continue
            if "datetime" in logic.lower():
                df[target_col] = pd.to_datetime(df[target_col], errors="coerce")
            elif "numeric" in logic.lower():
                df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        df["_load_timestamp"] = load_ts
        df["_source_file"] = str(csv_path)

        out_path = BRONZE_DIR / f"{table_stem}_bronze.parquet"
        df.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def execute_bronze(input_files: List[str], sttm_path: str, run_id: str, task_description: str = "") -> List[str]:
    """Runs the Bronze agent. Returns the list of written Bronze Parquet paths."""
    trace = AgentTrace("bronze_agent", run_id, input_data={"input_files": input_files, "sttm_path": sttm_path})
    scratchpad: dict = {}

    @tool
    def inspect_task_tool(confirmation: str = "inspect") -> str:
        """Preview CSV file shapes and the approved Bronze STTM transformation rules."""
        trace.log_tool_call("inspect_task_tool")
        shapes = {}
        for p in input_files:
            df = pd.read_csv(p, nrows=20)
            shapes[Path(p).name] = {"columns": list(df.columns)}
        sttm_preview = pd.read_csv(sttm_path).to_dict(orient="records")
        content = json.dumps({"csv_shapes": shapes, "sttm_rules": sttm_preview[:50]})
        trace.log_tool_result("inspect_task_tool", content[:2000])
        return content

    @tool
    def bronze_ingestion_tool(confirmation: str = "execute") -> str:
        """Apply the approved Bronze STTM rules and write Parquet files."""
        trace.log_tool_call("bronze_ingestion_tool")
        paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        scratchpad["output_paths"] = paths
        content = json.dumps(paths)
        trace.log_tool_result("bronze_ingestion_tool", content)
        return content

    tools = [inspect_task_tool, bronze_ingestion_tool]
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

    output_paths = scratchpad.get("output_paths") or _apply_bronze_rules(input_files, sttm_path, run_id)
    trace.finish(output={"output_paths": output_paths})
    return output_paths
