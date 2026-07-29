"""
tests/test_pipeline_layers.py
Unit tests for the deterministic data-transformation functions used by
each medallion layer. These do NOT call any LLM — they test the pandas
logic that actually reads/writes Parquet, so they run without an API key.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from agents.bronze_agent import _apply_bronze_rules
from agents.silver_agent import _apply_silver_rules
from agents.gold_agent import _apply_gold_rules
from agents.sttm_generator import _fallback_bronze_rows, _fallback_silver_rows, _fallback_gold_rows, _write_sttm_csv
from core.config import STTM_DIR


@pytest.fixture
def run_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "sale_date": ["2024-01-15", "2024-01-16", "2024-01-16", None],
        "product_id": ["P001", "P002", "P002", "P001"],
        "quantity": [2, 5, 1, 3],
        "unit_price": [1299.99, 249.99, 1299.99, None],
    })
    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _fake_profile(csv_path):
    from agents.profiler import _full_profile
    return {"full_profile": _full_profile([csv_path])}


def test_bronze_to_gold_pipeline(sample_csv, run_id):
    # --- Bronze ---
    profile = _fake_profile(sample_csv)
    bronze_rows = _fallback_bronze_rows(profile)
    bronze_sttm_path = STTM_DIR / f"test_bronze_{run_id[:8]}.csv"
    _write_sttm_csv(bronze_rows, bronze_sttm_path)

    bronze_paths = _apply_bronze_rules([sample_csv], str(bronze_sttm_path), run_id)
    assert len(bronze_paths) == 1
    bronze_df = pd.read_parquet(bronze_paths[0])
    assert "_load_timestamp" in bronze_df.columns
    assert "_source_file" in bronze_df.columns
    assert len(bronze_df) == 4  # bronze keeps raw row count, no null handling

    # --- Silver ---
    silver_rows = _fallback_silver_rows(bronze_paths)
    silver_sttm_path = STTM_DIR / f"test_silver_{run_id[:8]}.csv"
    _write_sttm_csv(silver_rows, silver_sttm_path)

    silver_paths = _apply_silver_rules(bronze_paths, str(silver_sttm_path), run_id)
    assert len(silver_paths) == 1
    silver_df = pd.read_parquet(silver_paths[0])
    pk_cols = [c for c in silver_df.columns if c.startswith("pk_")]
    assert len(pk_cols) == 1
    assert pk_cols[0].endswith("_silver_id")
    assert silver_df.columns[0] == pk_cols[0]  # surrogate key is first column

    # --- Gold ---
    gold_rows = _fallback_gold_rows(silver_paths)
    gold_sttm_path = STTM_DIR / f"test_gold_{run_id[:8]}.csv"
    _write_sttm_csv(gold_rows, gold_sttm_path)

    gold_paths = _apply_gold_rules(silver_paths, str(gold_sttm_path), "test intent", run_id)
    assert len(gold_paths) >= 1
    gold_df = pd.read_parquet(gold_paths[0])
    assert gold_df.columns[0] == "pk_gold_id"

    # cleanup
    for p in [bronze_sttm_path, silver_sttm_path, gold_sttm_path, *bronze_paths, *silver_paths, *gold_paths]:
        Path(p).unlink(missing_ok=True)
