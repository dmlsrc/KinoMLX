"""Explicit opt-in real LTX-2.5 Phase H public-capability gate."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from .phase_h_gate import run_phase_h


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.requires_metal
def test_real_ltx25_phase_h_public_capabilities() -> None:
    if os.environ.get("KINO_RUN_REAL_LTX25_PHASE_H") != "1":
        pytest.skip("set KINO_RUN_REAL_LTX25_PHASE_H=1 for the complete Phase H gate")
    shared_temp = os.environ.get("SHARED_TEMP_DIR")
    if not shared_temp:
        pytest.skip("SHARED_TEMP_DIR is not configured for the durable Phase H receipt")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(shared_temp) / f"kinomlx_ltx25_phase_h_{timestamp}_{os.getpid()}"
    receipt = run_phase_h(output_dir)
    assert receipt.is_file()
