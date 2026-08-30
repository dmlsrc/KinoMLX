"""Explicit opt-in real LTX-2.5 Phase G public-pipeline gate."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from .phase_g_gate import run_phase_g


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.requires_avfoundation
def test_real_ltx25_phase_g_public_pipeline() -> None:
    if os.environ.get("KINO_RUN_REAL_LTX25") != "1":
        pytest.skip("set KINO_RUN_REAL_LTX25=1 for the complete LTX-2.5 Phase G gate")
    shared_temp = os.environ.get("SHARED_TEMP_DIR")
    if shared_temp is None:
        pytest.skip("SHARED_TEMP_DIR is not configured for the durable Phase G receipt")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(shared_temp) / f"kinomlx_ltx25_phase_g_{timestamp}_{os.getpid()}"
    receipt = run_phase_g(output_dir)
    assert receipt.is_file()
