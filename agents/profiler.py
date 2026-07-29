"""
agents/profiler.py
Data Profiler agent — profiles raw CSV files to understand structure,
semantics, and quality before any transformation rules are written.

Follows THINK -> INSPECT -> PLAN -> ACT -> VERIFY:
  inspect_files_tool  : lightweight preview (shape, columns, dtypes, samples)
  profiler_tool       : full statistics (nulls, uniques, min/max/mean)
"""
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.config import PROFILES_DIR, make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Profiler agent in a medallion data pipeline.
Your job: understand the raw CSV files well enough that downstream agents
can safely design transformation rules.

Always follow this loop:
1. Call inspect_files_tool first to preview shapes, columns, and dtypes.
2. Based on what you see, call profiler_tool to compute full statistics.
3. When profiler_tool returns, reply with a short natural-language summary
   of what you found (semantic meaning of columns, likely join keys, data
   quality issues). Do not call any more tools after that.
"""


def _light_inspect(csv_paths: List[str]) -> dict:
    out = {}
    for p in csv_paths:
        df = pd.read_csv(p, nrows=50)
        out[Path(p).name] = {
            "path": p,
            "shape_preview_rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample_values": {c: df[c].dropna().astype(str).head(3).tolist() for c in df.columns},
        }
    return out


def _full_profile(csv_paths: List[str]) -> dict:
    out = {}
    for p in csv_paths:
        df = pd.read_csv(p)
        col_stats = {}
        for c in df.columns:
            series = df[c]
            stats = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique()),
            }
            if pd.api.types.is_numeric_dtype(series):
                stats.update(
                    {
                        "min": float(series.min()) if series.notna().any() else None,
                        "max": float(series.max()) if series.notna().any() else None,
                        "mean": float(series.mean()) if series.notna().any() else None,
                    }
                )
            col_stats[c] = stats
        out[Path(p).name] = {
            "path": p,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": col_stats,
            "likely_id_columns": [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"],
        }
    return out


def profile_multiple_datasets(csv_paths: List[str], business_intent: str, run_id: str) -> str:
    """
    Runs the Profiler agent over the given CSV files and writes
    data/profiles/profile_combined_{run_id[:8]}.json

    Returns the path to the written profile JSON.
    """
    trace = AgentTrace("profiler", run_id, input_data={"csv_paths": csv_paths, "business_intent": business_intent})
    scratchpad: dict = {}

    @tool
    def inspect_files_tool(confirmation: str = "inspect") -> str:
        """Preview the raw CSV files: shape, column names, dtypes, and 3 sample values per column."""
        trace.log_tool_call("inspect_files_tool")
        result = _light_inspect(csv_paths)
        scratchpad["light_inspect"] = result
        content = json.dumps(result)
        trace.log_tool_result("inspect_files_tool", content[:2000])
        return content

    @tool
    def profiler_tool(confirmation: str = "profile") -> str:
        """Compute full column statistics: null count/%, unique count, min/max/mean for numeric columns."""
        trace.log_tool_call("profiler_tool")
        result = _full_profile(csv_paths)
        scratchpad["full_profile"] = result
        content = json.dumps(result)
        trace.log_tool_result("profiler_tool", content[:2000])
        return content

    tools = [inspect_files_tool, profiler_tool]
    llm = make_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Business question: {business_intent}\n"
                f"Files to profile: {csv_paths}\n"
                "Begin."
            )
        ),
    ]

    summary_text = ""
    for _ in range(3):  # bounded ReAct loop — 2 steps needed (inspect→profile)
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)
        if ai_msg.content:
            trace.log_step("ai_reasoning", str(ai_msg.content))
        if not getattr(ai_msg, "tool_calls", None):
            summary_text = str(ai_msg.content)
            break
        for call in ai_msg.tool_calls:
            fn = tool_map[call["name"]]
            result = fn.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    profile_payload = {
        "run_id": run_id,
        "business_intent": business_intent,
        "light_inspect": scratchpad.get("light_inspect", {}),
        "full_profile": scratchpad.get("full_profile", _full_profile(csv_paths)),
        "agent_summary": summary_text,
    }

    profile_path = PROFILES_DIR / f"profile_combined_{run_id[:8]}.json"
    profile_path.write_text(json.dumps(profile_payload, indent=2, default=str))

    trace.finish(output={"profile_path": str(profile_path)})
    return str(profile_path)
