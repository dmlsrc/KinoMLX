"""Repo-hygiene checks enforced as tests - conditions ruff cannot express.

ruff's RUF001-003 flag only visually-confusable Unicode (the multiplication
sign, smart quotes, lookalike letters), not em-dashes / arrows / ellipses, and
ruff has no way to state a project rule like "os.environ only in settings".
These tests close that gap, in plain Python so the conditions can grow.
"""

from __future__ import annotations

import ast
import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PACKAGE = _REPO_ROOT / "kinomlx"
_TESTS = _REPO_ROOT / "tests"

# Removed in the dependency-minimization pass; the runtime is MLX +
# sentencepiece + lazy protobuf derivation + pyobjc, native-first for
# image/audio/video I/O. Tests may use numpy, einops, and Tokenizers as oracles,
# so only the package is scanned, not tests/.
_BANNED_RUNTIME_IMPORTS = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "tqdm",
    "av",
    "PIL",
    "pillow",
    "safetensors",
    "scipy",
    "soundfile",
    "cv2",
    "pandas",
    "einops",
)


def _py_files() -> list[pathlib.Path]:
    return sorted(_PACKAGE.rglob("*.py")) + sorted(_TESTS.rglob("*.py"))


def test_python_sources_are_plain_ascii() -> None:
    """Every .py file is plain ASCII (the workspace style rule).

    ruff catches only confusable characters, so em-dashes, arrows, ellipses,
    and the like slip past it; enforce the full rule here. Use ASCII: '-' for
    an em-dash, '->' for an arrow, '...' for an ellipsis, '>=' for the
    greater-or-equal sign.
    """
    offenders: dict[str, list[str]] = {}
    for f in _py_files():
        bad = sorted({c for c in f.read_text() if ord(c) > 127})
        if bad:
            offenders[str(f.relative_to(_REPO_ROOT))] = [f"U+{ord(c):04X} {c!r}" for c in bad]
    assert not offenders, f"Non-ASCII characters in Python sources:\n{offenders}"


def test_environment_reads_only_in_settings() -> None:
    """Only ``kinomlx/settings.py`` reads the process environment directly.

    Infrastructure and model records share the environment bridge in this
    module; every other module takes typed settings instead of reaching into
    the environment itself.
    """
    allowed = {_PACKAGE / "settings.py"}
    offenders: list[str] = []
    for source_path in _PACKAGE.rglob("*.py"):
        if source_path in allowed:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        os_aliases = {
            alias.asname or "os"
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "os"
        }
        relative = source_path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    if alias.name in {"environ", "getenv"}:
                        offenders.append(f"{relative}:{node.lineno}: imports os.{alias.name}")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in os_aliases
                and node.attr in {"environ", "getenv"}
            ):
                offenders.append(f"{relative}:{node.lineno}: reads os.{node.attr}")
    assert not offenders, (
        "Environment read outside settings.py - route config through "
        f"kinomlx.settings.EnvironmentSettings instead:\n{offenders}"
    )


def test_no_banned_runtime_imports() -> None:
    """kinomlx/ stays MLX-native: no numpy/torch/heavy-HF/cv2/etc. imports.

    Catches a regression where a lift or a casual edit reintroduces a
    dependency the runtime deliberately dropped (top-level or lazy).
    """
    pat = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(_BANNED_RUNTIME_IMPORTS) + r")\b")
    offenders = []
    for f in _PACKAGE.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            m = pat.match(line)
            if m:
                offenders.append(f"{f.relative_to(_REPO_ROOT)}:{i}: imports {m.group(1)}")
    assert not offenders, "Banned runtime import(s) - keep kinomlx MLX-native:\n" + "\n".join(
        offenders
    )


def test_mlx_array_parameters_are_not_silently_coerced() -> None:
    """An ``mx.array`` annotation is a runtime boundary, not a coercion hint.

    Direct ``mx.array(parameter)`` calls accept NumPy and other foreign arrays,
    which makes it easy to introduce hidden host copies. Byte/file ingress must
    use the explicit buffer adapter; model APIs keep MLX arrays end to end.
    """

    def is_mlx_array(annotation: ast.expr | None) -> bool:
        return (
            isinstance(annotation, ast.Attribute)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "mx"
            and annotation.attr == "array"
        )

    offenders: list[str] = []
    for source_path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parameters = {
                argument.arg
                for argument in [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                ]
                if is_mlx_array(argument.annotation)
            }
            if not parameters:
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                if not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "mx"
                    and node.func.attr == "array"
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in parameters
                ):
                    continue
                relative = source_path.relative_to(_REPO_ROOT)
                offenders.append(
                    f"{relative}:{node.lineno}: {function.name} coerces {node.args[0].id}"
                )

    assert not offenders, (
        "Typed MLX parameters must reject foreign arrays instead of coercing them:\n"
        + "\n".join(offenders)
    )


def test_models_do_not_cross_import() -> None:
    """Each model package is self-contained: no ``kinomlx.models.<other>`` import.

    A model lift drops a whole subtree under ``models/<name>/``; keeping the
    packages import-isolated means one can be added or removed without
    disturbing the others. Shared code lives in io/ / lora/ / samplers/, not in
    a sibling model. (Vacuous while ltx2 is the only model - it starts doing
    work the moment a second one lands.)
    """
    models_root = _PACKAGE / "models"
    pkgs = [d.name for d in models_root.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    offenders = []
    for pkg in pkgs:
        others = [m for m in pkgs if m != pkg]
        if not others:
            continue
        pat = re.compile(r"(?:from|import)\s+kinomlx\.models\.(" + "|".join(others) + r")\b")
        for f in (models_root / pkg).rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), start=1):
                m = pat.search(line)
                if m:
                    offenders.append(
                        f"{f.relative_to(_REPO_ROOT)}:{i}: imports models.{m.group(1)}"
                    )
    assert not offenders, "Cross-model imports (keep model packages self-contained):\n" + "\n".join(
        offenders
    )
