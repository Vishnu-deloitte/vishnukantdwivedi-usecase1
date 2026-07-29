"""
core/observability.py
AgentTrace — captures a single agent invocation's full reasoning trace
(plan, tool calls, reasoning steps, output) and writes it to
data/traces/trace_{agent_name}_{run_id[:8]}.json
"""
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from core.config import TRACES_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentTrace:
    def __init__(self, agent_name: str, run_id: str, input_data: Optional[dict] = None):
        self.agent_name = agent_name
        self.run_id = run_id
        self.started_at = _now()
        self._start_time = time.time()
        self.input = input_data or {}
        self.plan: str = ""
        self.tool_calls: list = []
        self.reasoning_steps: list = []
        self.output: dict = {}
        self.status: str = "running"
        self.error: Optional[str] = None

    # -- recording -----------------------------------------------------
    def set_plan(self, plan: str) -> None:
        self.plan = plan
        self.reasoning_steps.append({"role": "ai_plan", "content": plan})

    def log_step(self, role: str, content: str) -> None:
        self.reasoning_steps.append({"role": role, "content": content})

    def log_tool_call(self, tool: str, args: Optional[dict] = None) -> None:
        self.tool_calls.append({"tool": tool, "args": args or {}})

    def log_tool_result(self, tool: str, content: str) -> None:
        self.reasoning_steps.append({"role": "tool_result", "tool": tool, "content": content})

    def finish(self, output: Optional[dict] = None, status: str = "success", error: Optional[str] = None) -> str:
        self.output = output or {}
        self.status = status
        self.error = error
        return self.save()

    # -- persistence -----------------------------------------------------
    def save(self) -> str:
        path = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        payload = {
            "agent": self.agent_name,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "input": self.input,
            "plan": self.plan,
            "tool_calls": self.tool_calls,
            "reasoning_steps": self.reasoning_steps,
            "output": self.output,
            "duration_seconds": round(time.time() - self._start_time, 2),
            "status": self.status,
            "error": self.error,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)
