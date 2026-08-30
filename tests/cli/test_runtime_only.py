"""CLI discovery and help stay independent from the MLX/model runtime."""

from __future__ import annotations

import subprocess
import sys

_BLOCKED_PROBE = r"""
import sys

BLOCKED = ("mlx", "google.protobuf", "sentencepiece", "Foundation", "AVFoundation", "VideoToolbox")

def is_blocked(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if is_blocked(name):
            raise ImportError(f"{name} blocked: CLI discovery must stay lightweight")
        return None

sys.meta_path.insert(0, Blocker())

import kinomlx
from kinomlx.cli.args import build_parser

parser = build_parser()
text = parser.format_help()
assert "--prompt" in text
assert "--print-config" in text
assert "--save-config" in text
assert "--only-non-defaults" in text
assert "--save-effective-config" in text
assert "kinomlx.models.ltx2.cli" in sys.modules
assert "kinomlx.models.ltx2" in sys.modules
assert not any(name.startswith("_kinomlx_") for name in sys.modules)

from kinomlx.cli._registry import config_registry

registry = config_registry()
assert registry.models() == ("gmnet", "ltx2")
assert "KINO_OUTPUT_DIR" in registry.environment_variables()

from kinomlx.cli.config_init import build_config_parser

config_text = build_config_parser().format_help()
assert "init" in config_text
assert "mlx" not in sys.modules

from kinomlx.cli.config import assemble
from kinomlx.settings import Settings

invocation = assemble(
    parser.parse_args(["--prompt", "test", "--duration", "20", "--print-config"]),
    base_settings=Settings(),
)
assert invocation.request.frames == 481
assert "kinomlx.models.ltx2.runner" not in sys.modules
assert "kinomlx.models.ltx2.pipelines.distilled" not in sys.modules
offenders = sorted(name for name in sys.modules if is_blocked(name))
assert not offenders, f"runtime modules imported by help: {offenders}"
print("lightweight help ok")
"""


def test_import_and_help_without_model_runtime() -> None:
    process = subprocess.run(
        [sys.executable, "-c", _BLOCKED_PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "lightweight help ok" in process.stdout
