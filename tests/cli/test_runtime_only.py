"""CLI discovery and help avoid optional external model runtimes."""

import subprocess
import sys

_BLOCKED_PROBE = r"""
import sys

BLOCKED = ("tokenizers", "transformers", "diffusers", "torch", "PIL", "pydantic", "pydantic_core", "pydantic_settings")

def is_blocked(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if is_blocked(name):
            raise ImportError(f"{name} blocked: CLI discovery must use KinoMLX dependencies")
        return None

sys.meta_path.insert(0, Blocker())

import kinomlx
from kinomlx.cli.arguments import build_root_parser

root_parser = build_root_parser()
assert "--model" in root_parser.format_help()

from kinomlx.models.ltx2.arguments import build_parser

parser = build_parser()
text = parser.format_help()
assert "--prompt" in text
assert "--print-config" in text
assert "--save-config" in text
assert "--only-non-defaults" in text
assert "--save-effective-config" in text
assert "kinomlx.cli.config_arguments" in sys.modules
assert "kinomlx.models.ltx2" in sys.modules
assert not any(name.startswith("_kinomlx_") for name in sys.modules)

from kinomlx.model_registry import config_registry

registry = config_registry()
assert registry.models() == ("flux2", "gmnet", "krea2", "ltx2")
from kinomlx.models.flux2.command import build_parser as klein_parser
assert "--dry-run" in klein_parser().format_help()
assert "KINO_LTX2_OUTPUT_DIR" in registry.environment_variables()

from kinomlx.cli.config_init import build_config_parser

config_text = build_config_parser().format_help()
assert "init" in config_text
assert "inspect" in config_text
from kinomlx.config.inspection import configuration_inventory
assert configuration_inventory(registry.model("krea2"), field="generate.prompt")["fields"]

from kinomlx.models.ltx2.config import assemble
from kinomlx.settings import Settings

invocation = assemble(
    parser.parse_args(["--prompt", "test", "--duration", "20", "--print-config"]),
    base_settings=Settings(),
)
assert invocation.request.frames == 481
offenders = sorted(name for name in sys.modules if is_blocked(name))
assert not offenders, f"optional external runtimes imported by help: {offenders}"
print("help and config assembly ok")
"""


def test_import_and_help_without_optional_external_runtimes() -> None:
    process = subprocess.run(
        [sys.executable, "-c", _BLOCKED_PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "help and config assembly ok" in process.stdout
