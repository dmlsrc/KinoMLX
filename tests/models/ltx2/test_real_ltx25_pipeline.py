"""Explicit opt-in real LTX-2.5 pipeline public-pipeline gate."""

import os
import time
from pathlib import Path

import pytest

from ._receipts import receipt_root
from .ltx25_pipeline_gate import run_ltx25_pipeline


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.requires_avfoundation
def test_real_ltx25_pipeline_public_pipeline(tmp_path: Path) -> None:
    if os.environ.get("KINO_RUN_REAL_LTX25") != "1":
        pytest.skip("set KINO_RUN_REAL_LTX25=1 for the complete LTX-2.5 pipeline gate")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = receipt_root(tmp_path) / f"kinomlx_ltx25_pipeline_{timestamp}_{os.getpid()}"
    receipt = run_ltx25_pipeline(output_dir)
    assert receipt.is_file()
