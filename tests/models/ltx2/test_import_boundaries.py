"""The model stack stays self-contained: mlx-lm is reference-only."""

from __future__ import annotations

import subprocess
import sys

_BLOCKED_PROBE = r"""
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "mlx_lm":
            raise ImportError("mlx_lm blocked: borrowed-from reference, not a dependency")
        return None

sys.meta_path.insert(0, Blocker())

import kinomlx.models.ltx2.runner
import kinomlx.models.ltx2.pipelines.distilled
import kinomlx.models.ltx2.text_encoder.gemma3

assert "mlx_lm" not in sys.modules
print("mlx_lm boundary ok")
"""


def test_model_stack_never_imports_mlx_lm() -> None:
    process = subprocess.run(
        [sys.executable, "-c", _BLOCKED_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "mlx_lm boundary ok" in process.stdout
