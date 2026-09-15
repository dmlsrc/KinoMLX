"""Smoke and eager-export contract tests for public package surfaces.

This is the M1 verification gate.  As subpackages gain real code, this
test still passes if every module's top-level imports succeed - it does
not exercise runtime behavior.
"""

import ast
import importlib
from pathlib import Path

import pytest

# Top-level package, the two single-file modules, and each subpackage
# under ``kinomlx/``. Per-model namespaces
# are listed separately so a missing model surfaces with a clear name.
TOP_LEVEL_MODULES = [
    "kinomlx",
    "kinomlx.settings",
    "kinomlx.types",
    "kinomlx.audio",
    "kinomlx.cli",
    "kinomlx.config",
    "kinomlx.debug",
    "kinomlx.errors",
    "kinomlx.io",
    "kinomlx.huggingface",
    "kinomlx.kernels",
    "kinomlx.adapters",
    "kinomlx.models",
    "kinomlx.modeling",
    "kinomlx.profiling",
    "kinomlx.reporting",
    "kinomlx.samplers",
    "kinomlx.ui",
    "kinomlx.videotoolbox",
    "kinomlx.weights",
]

MODEL_PACKAGES = [
    "kinomlx.models.gmnet",
    "kinomlx.models.krea2",
    "kinomlx.models.ltx2",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _initializer_module_name(relative_path: Path) -> str:
    return ".".join(relative_path.parent.parts)


def _declares_public_exports(tree: ast.Module) -> bool:
    for node in tree.body:
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            return True
    return False


def _discover_public_initializers() -> tuple[Path, ...]:
    initializers = []
    for source_path in sorted((REPO_ROOT / "kinomlx").rglob("__init__.py")):
        relative_path = source_path.relative_to(REPO_ROOT)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(relative_path))
        if _declares_public_exports(tree):
            initializers.append(relative_path)
    return tuple(initializers)


PUBLIC_INITIALIZERS = _discover_public_initializers()


@pytest.mark.parametrize("name", TOP_LEVEL_MODULES + MODEL_PACKAGES)
def test_import(name: str) -> None:
    """Every listed module imports cleanly."""
    importlib.import_module(name)


def test_version_exposed() -> None:
    """``kinomlx.__version__`` is a non-empty string."""
    import kinomlx

    assert isinstance(kinomlx.__version__, str)
    assert kinomlx.__version__


def test_ltx2_public_api_is_exported_from_model_package() -> None:
    from kinomlx.models.ltx2 import (
        DistilledRequest,
        DistilledRestart,
        ImageConditioningConfig,
        LTX2Runner,
        LTX2Settings,
        restart_distilled,
    )

    assert DistilledRequest.__name__ == "DistilledRequest"
    assert DistilledRestart.__name__ == "DistilledRestart"
    assert ImageConditioningConfig.__name__ == "ImageConditioningConfig"
    assert LTX2Runner.__name__ == "LTX2Runner"
    assert LTX2Settings.__name__ == "LTX2Settings"
    assert restart_distilled.__name__ == "restart_distilled"


def test_gmnet_public_api_is_exported_from_model_package() -> None:
    from kinomlx.models.gmnet import (
        ExpansionResult,
        GMNetOutputConfig,
        GMNetOutputPlan,
        GMNetOutputReservation,
        GMNetRequest,
        GMNetRunner,
        GMNetSettings,
        expand_gmnet,
        prepare_resources,
        write_gmnet_output,
    )

    assert GMNetOutputConfig.__name__ == "GMNetOutputConfig"
    assert GMNetOutputPlan.__name__ == "GMNetOutputPlan"
    assert GMNetOutputReservation.__name__ == "GMNetOutputReservation"
    assert ExpansionResult.__name__ == "ExpansionResult"
    assert GMNetRequest.__name__ == "GMNetRequest"
    assert GMNetRunner.__name__ == "GMNetRunner"
    assert GMNetSettings.__name__ == "GMNetSettings"
    assert expand_gmnet.__name__ == "expand_gmnet"
    assert prepare_resources.__name__ == "prepare_resources"
    assert write_gmnet_output.__name__ == "write_gmnet_output"


def test_krea2_public_api_is_exported_from_model_package() -> None:
    from kinomlx.models.krea2 import (
        Krea2Capabilities,
        Krea2PromptConditioning,
        Krea2Resources,
        Krea2Settings,
        Krea2TextConditioner,
        Krea2TextConditioning,
        Krea2TextConditioningProvenance,
        Krea2Variant,
        NativeKrea2TextConditioner,
        prepare_resources,
    )

    assert Krea2Capabilities.__name__ == "Krea2Capabilities"
    assert Krea2PromptConditioning.__name__ == "Krea2PromptConditioning"
    assert Krea2Resources.__name__ == "Krea2Resources"
    assert Krea2Settings.__name__ == "Krea2Settings"
    assert Krea2TextConditioner.__name__ == "Krea2TextConditioner"
    assert Krea2TextConditioning.__name__ == "Krea2TextConditioning"
    assert Krea2TextConditioningProvenance.__name__ == "Krea2TextConditioningProvenance"
    assert Krea2Variant.__name__ == "Krea2Variant"
    assert NativeKrea2TextConditioner.__name__ == "NativeKrea2TextConditioner"
    assert prepare_resources.__name__ == "prepare_resources"


@pytest.mark.parametrize("relative_path", PUBLIC_INITIALIZERS, ids=str)
def test_every_public_export_is_bound_and_resolves(relative_path: Path) -> None:
    """Every advertised public name is an ordinary runtime module binding."""
    module = importlib.import_module(_initializer_module_name(relative_path))
    advertised = tuple(module.__all__)

    assert len(advertised) == len(set(advertised))
    assert set(advertised) <= set(vars(module))
    assert set(advertised) <= set(dir(module))
    for name in advertised:
        assert getattr(module, name) is not None
