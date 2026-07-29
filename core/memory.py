"""
core/memory.py
DocumentStore — a lightweight in-memory + on-disk store that agents use
to stash and retrieve context documents (profiles, STTM summaries, run
metadata) by run_id. This is deliberately simple: a dict keyed by run_id,
persisted as JSON so a resumed run can reload prior context.
"""
import json
from pathlib import Path
from typing import Any, Optional

from core.config import BASE_DIR

_STORE_PATH = BASE_DIR / "memory_store.json"


class DocumentStore:
    def __init__(self):
        self._data: dict = self._load()

    def _load(self) -> dict:
        if _STORE_PATH.exists():
            try:
                return json.loads(_STORE_PATH.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _persist(self) -> None:
        _STORE_PATH.write_text(json.dumps(self._data, indent=2, default=str))

    def put(self, run_id: str, key: str, value: Any) -> None:
        self._data.setdefault(run_id, {})[key] = value
        self._persist()

    def get(self, run_id: str, key: str, default: Optional[Any] = None) -> Any:
        return self._data.get(run_id, {}).get(key, default)

    def get_run(self, run_id: str) -> dict:
        return self._data.get(run_id, {})

    def clear_run(self, run_id: str) -> None:
        self._data.pop(run_id, None)
        self._persist()


# Module-level singleton so agents in the same process share context.
document_store = DocumentStore()
