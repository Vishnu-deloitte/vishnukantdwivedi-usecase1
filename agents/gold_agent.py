"""
agents/gold_agent.py
Gold layer materialisation agent — Silver Parquet -> analytics-ready
Gold Parquet. Multi-source joins (outer join on matching _id columns),
column renaming, groupby().agg() aggregations, pk_gold_id surrogate key.
One file per Gold target table defined in the STTM.
"""
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import GOLD_DIR, make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Gold materialisation agent.
Follow THINK -> INSPECT -> PLAN -> ACT -> VERIFY:
1. Call inspect_task_tool to preview Silver schemas and the approved Gold STTM, grouped by target table.
2. State a short plan for joins and aggregations that will answer the business question.
3. Call gold_ingestion_tool to execute the materialisation.
4. Reply with a one-line confirmation. Do not call more tools after that.
"""

_AGG_MAP = {"SUM": "sum", "AVG": "mean", "COUNT": "count", "MAX": "max", "MIN": "min"}


def _find_join_key(a: pd.DataFrame, b: pd.DataFrame) -> str:
    id_cols_a = [c for c in a.columns if c.lower().endswith("_id") and not c.lower().startswith("pk_")]
    id_cols_b = set(c for c in b.columns if c.lower().endswith("_id") and not c.lower().startswith("pk_"))
    for c in id_cols_a:
        if c in id_cols_b:
            return c
    return ""


def _apply_gold_rules(input_files: List[str], sttm_path: str, business_intent: str, run_id: str) -> List[str]:
    sttm = pd.read_csv(sttm_path)
    frames = {Path(p).stem: pd.read_parquet(p) for p in input_files}
    output_paths = []

    target_tables = [t for t in sttm["target_table"].dropna().unique() if t]
    if not target_tables:
        target_tables = ["gold_summary"]

    for target_table in target_tables:
        rules = sttm[sttm["target_table"] == target_table]

        # Merge all relevant silver frames on shared *_id columns (outer join)
        merged = None
        for name, df in frames.items():
            if merged is None:
                merged = df.copy()
            else:
                key = _find_join_key(merged, df)
                if key:
                    merged = merged.merge(df, on=key, how="outer", suffixes=("", f"_{name}"))
                else:
                    merged = merged.join(df.set_index(df.columns[0]), how="outer", rsuffix=f"_{name}")

        if merged is None:
            continue

        agg_rules = rules[rules["transformation_type"] == "AGGREGATE"]
        rename_rules = rules[rules["transformation_type"] == "RENAME"]

        for _, r in rename_rules.iterrows():
            src, tgt = r.get("source_column"), r.get("target_column")
            if src in merged.columns and pd.notna(tgt):
                merged = merged.rename(columns={src: tgt})

        if len(agg_rules):
            group_cols = [
                r["source_column"] for _, r in agg_rules.iterrows()
                if str(r.get("transformation_logic", "")).upper() == "GROUP_BY" and r["source_column"] in merged.columns
            ]
            agg_map = {}
            for _, r in agg_rules.iterrows():
                col = r.get("source_column")
                logic = str(r.get("transformation_logic", "")).upper()
                if col in merged.columns and logic in _AGG_MAP:
                    agg_map[col] = _AGG_MAP[logic]
            if group_cols and agg_map:
                result = merged.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
            else:
                result = merged
        else:
            result = merged

        result.insert(0, "pk_gold_id", range(1, len(result) + 1))
        out_path = GOLD_DIR / f"{target_table}.parquet"
        result.to_parquet(out_path, index=False)
        output_paths.append(str(out_path))

    return output_paths


def execute_gold(input_files: List[str], sttm_path: str, business_intent: str, run_id: str,
                  task_description: str = "") -> List[str]:
    """Runs the Gold agent. Returns the list of written Gold Parquet paths."""
    trace = AgentTrace("gold_agent", run_id, input_data={"input_files": input_files, "sttm_path": sttm_path})
    scratchpad: dict = {}

    @tool
    def inspect_task_tool(confirmation: str = "inspect") -> str:
        """Preview Silver schemas and the approved Gold STTM, grouped by target Gold table."""
        trace.log_tool_call("inspect_task_tool")
        schemas = {}
        for p in input_files:
            df = pd.read_parquet(p)
            schemas[Path(p).name] = list(df.columns)
        sttm_preview = pd.read_csv(sttm_path).to_dict(orient="records")
        content = json.dumps({"silver_schemas": schemas, "sttm_rules": sttm_preview[:50]}, default=str)
        trace.log_tool_result("inspect_task_tool", content[:2000])
        return content

    @tool
    def gold_ingestion_tool(confirmation: str = "execute") -> str:
        """Apply joins and aggregations from the approved Gold STTM and write Parquet files."""
        trace.log_tool_call("gold_ingestion_tool")
        paths = _apply_gold_rules(input_files, sttm_path, business_intent, run_id)
        scratchpad["output_paths"] = paths
        content = json.dumps(paths)
        trace.log_tool_result("gold_ingestion_tool", content)
        return content

    tools = [inspect_task_tool, gold_ingestion_tool]
    llm = make_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Business intent: {business_intent}\nTask: {task_description}\nInput files: {input_files}\nBegin."),
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

    output_paths = scratchpad.get("output_paths") or _apply_gold_rules(input_files, sttm_path, business_intent, run_id)
    trace.finish(output={"output_paths": output_paths})
    return output_paths
