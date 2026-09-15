"""Explicit opt-in real LTX-2.5 capabilities public-capability gate."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ._receipts import receipt_root
from .ltx25_capabilities_gate import run_ltx25_capabilities


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.requires_metal
def test_real_ltx25_capabilities_public_capabilities(tmp_path: Path) -> None:
    if os.environ.get("KINO_RUN_REAL_LTX25_CAPABILITIES") != "1":
        pytest.skip(
            "set KINO_RUN_REAL_LTX25_CAPABILITIES=1 for the complete LTX-2.5 capabilities gate"
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = receipt_root(tmp_path) / f"kinomlx_ltx25_capabilities_{timestamp}_{os.getpid()}"
    receipt = run_ltx25_capabilities(output_dir)
    assert receipt.is_file()
