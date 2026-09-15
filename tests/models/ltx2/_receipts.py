"""Where the opt-in LTX-2.5 integration gates write their receipts."""

import os
from pathlib import Path


def receipt_root(tmp_path: Path) -> Path:
    """Return ``KINO_LTX25_RECEIPT_DIR`` when set, else the test's temporary directory.

    The gates write a durable receipt bundle. Pointing the variable at a kept
    directory preserves it across runs; without it the bundle lands in pytest's
    per-test temporary directory and follows its retention policy.
    """
    configured = os.environ.get("KINO_LTX25_RECEIPT_DIR")
    if configured:
        return Path(configured).expanduser().absolute()
    return tmp_path
