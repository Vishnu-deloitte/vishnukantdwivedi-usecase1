"""
core/audit.py
AuditLogger — records phase start/complete/fail events to human-readable
JSON files under audit_logs/. Independent from the deep reasoning traces
in core/observability.py; this is the high-level "what happened" log.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.config import AUDIT_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    """One AuditLogger instance per pipeline run_id."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.path: Path = AUDIT_DIR / f"audit_{run_id[:8]}.json"
        if not self.path.exists():
            self._write([])

    def _read(self) -> list:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _write(self, events: list) -> None:
        self.path.write_text(json.dumps(events, indent=2, default=str))

    def _log(self, event_type: str, phase: str, **details: Any) -> None:
        events = self._read()
        events.append(
            {
                "timestamp": _now(),
                "run_id": self.run_id,
                "phase": phase,
                "event": event_type,
                **details,
            }
        )
        self._write(events)

    def phase_start(self, phase: str, **details: Any) -> None:
        self._log("phase_start", phase, **details)

    def phase_complete(self, phase: str, output_paths: Optional[list] = None, **details: Any) -> None:
        self._log("phase_complete", phase, output_paths=output_paths or [], **details)

    def phase_fail(self, phase: str, error: str, **details: Any) -> None:
        self._log("phase_fail", phase, error=error, **details)

    def decision(self, phase: str, decision: str, **details: Any) -> None:
        self._log("decision", phase, decision=decision, **details)

    def human_approval(self, phase: str, approved: bool, **details: Any) -> None:
        self._log("human_approval", phase, approved=approved, **details)

    def all_events(self) -> list:
        return self._read()
