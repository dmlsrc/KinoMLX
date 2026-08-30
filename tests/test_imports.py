"""Smoke and lazy-export contract tests for public package surfaces.

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
    "kinomlx.kernels",
    "kinomlx.lora",
    "kinomlx.models",
    "kinomlx.profiling",
    "kinomlx.reporting",
    "kinomlx.samplers",
    "kinomlx.ui",
    "kinomlx.videotoolbox",
    "kinomlx.weights",
]

MODEL_PACKAGES = [
    "kinomlx.models.gmnet",
    "kinomlx.models.ltx2",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_MAP_NAMES = {"_EXPORT_MODULES", "_EXPORT_TARGETS"}
type ExportTarget = tuple[str, str]


def _assignment_target(node: ast.Assign | ast.AnnAssign) -> ast.expr:
    if isinstance(node, ast.Assign):
        assert len(node.targets) == 1
        return node.targets[0]
    return node.target


def _module_bindings(tree: ast.Module) -> set[str]:
    bindings: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = _assignment_target(node)
            if isinstance(target, ast.Name):
                bindings.add(target.id)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bindings.add(alias.asname or alias.name.partition(".")[0])
    return bindings


def _is_type_checking_block(node: ast.If) -> bool:
    return (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
        isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
    )


def _initializer_module_name(relative_path: Path) -> str:
    return ".".join(relative_path.parent.parts)


def _export_map_nodes(tree: ast.Module) -> list[tuple[str, ast.Dict]]:
    export_maps: list[tuple[str, ast.Dict]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = _assignment_target(node)
        if (
            isinstance(target, ast.Name)
            and target.id in EXPORT_MAP_NAMES
            and isinstance(node.value, ast.Dict)
        ):
            export_maps.append((target.id, node.value))
    return export_maps


def _resolve_relative_module(package: str, module: str | None, level: int) -> str:
    package_parts = package.split(".")
    assert 0 < level <= len(package_parts)
    parts = package_parts[: len(package_parts) - level + 1]
    if module is not None:
        parts.extend(module.split("."))
    return ".".join(parts)


def _type_checking_imports(block: ast.If, package: str) -> dict[str, str]:
    imports: dict[str, str] = {}
    for statement in block.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                imports[alias.asname or alias.name.partition(".")[0]] = alias.name
        elif isinstance(statement, ast.ImportFrom):
            base = (
                _resolve_relative_module(package, statement.module, statement.level)
                if statement.level
                else statement.module
            )
            assert base is not None
            for alias in statement.names:
                assert alias.name != "*"
                imports[alias.asname or alias.name] = f"{base}.{alias.name}"
    return imports


def _type_checking_mirrors(block: ast.If, package: str) -> dict[str, ExportTarget]:
    imports = _type_checking_imports(block, package)
    mirrors: dict[str, ExportTarget] = {}
    for statement in block.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        target = _assignment_target(statement)
        if not isinstance(target, ast.Name) or target.id.startswith("_"):
            continue
        value = statement.value
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name)
        module = imports.get(value.value.id)
        assert module is not None, f"unknown static-mirror module alias {value.value.id!r}"
        mirrors[target.id] = (module, value.attr)
    return mirrors


def _lazy_export_contract(
    tree: ast.Module,
    package: str,
) -> tuple[dict[str, ExportTarget], set[str], dict[str, ExportTarget]]:
    export_maps = _export_map_nodes(tree)
    type_checking_blocks = [
        node for node in tree.body if isinstance(node, ast.If) and _is_type_checking_block(node)
    ]

    assert len(export_maps) == 1, "expected exactly one recognized lazy-export map"
    assert len(type_checking_blocks) == 1, "expected exactly one TYPE_CHECKING mirror block"
    map_name, export_map = export_maps[0]
    lazy: dict[str, ExportTarget] = {}
    eager: set[str] = set()
    for key, value in zip(export_map.keys, export_map.values, strict=True):
        assert isinstance(key, ast.Constant)
        assert isinstance(key.value, str)
        public_name = key.value
        if isinstance(value, ast.Constant) and value.value is None:
            eager.add(public_name)
            continue
        if map_name == "_EXPORT_MODULES":
            assert isinstance(value, ast.Constant)
            assert isinstance(value.value, str)
            raw_module = value.value
            module = raw_module if raw_module.startswith("kinomlx.") else f"{package}.{raw_module}"
            lazy[public_name] = (module, public_name)
            continue
        assert isinstance(value, ast.Tuple)
        assert len(value.elts) == 2
        module_node, attribute_node = value.elts
        assert isinstance(module_node, ast.Constant)
        assert isinstance(module_node.value, str)
        assert isinstance(attribute_node, ast.Constant)
        assert isinstance(attribute_node.value, str)
        lazy[public_name] = (module_node.value, attribute_node.value)

    mirrors = _type_checking_mirrors(type_checking_blocks[0], package)
    return lazy, eager, mirrors


def _discover_lazy_export_initializers() -> tuple[Path, ...]:
    initializers = []
    for source_path in sorted((REPO_ROOT / "kinomlx").rglob("__init__.py")):
        relative_path = source_path.relative_to(REPO_ROOT)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(relative_path))
        if _export_map_nodes(tree):
            initializers.append(relative_path)
    return tuple(initializers)


LAZY_EXPORT_INITIALIZERS = _discover_lazy_export_initializers()


@pytest.mark.parametrize("name", TOP_LEVEL_MODULES + MODEL_PACKAGES)
def test_import(name: str) -> None:
    """Every listed module imports cleanly."""
    importlib.import_module(name)


def test_version_exposed() -> None:
    """``kinomlx.__version__`` is a non-empty string."""
    import kinomlx

    assert isinstance(kinomlx.__version__, str)
    assert kinomlx.__version__


def test_settings_reexported() -> None:
    """``Settings`` is importable from the top-level package."""
    from kinomlx import Settings

    s = Settings.from_env()
    # Sanity check: ``from_env`` returns a frozen dataclass instance.
    assert s is not None


def test_ltx2_public_api_is_lazily_reexported() -> None:
    from kinomlx import (
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


def test_gmnet_public_api_is_lazily_reexported() -> None:
    from kinomlx import (
        ExpansionResult,
        GMNetOutputConfig,
        GMNetOutputPlan,
        GMNetOutputReservation,
        GMNetRequest,
        GMNetRunner,
        GMNetSettings,
        expand_gmnet,
        prepare_gmnet_resources,
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
    assert prepare_gmnet_resources.__name__ == "prepare_resources"
    assert write_gmnet_output.__name__ == "write_gmnet_output"


@pytest.mark.parametrize("relative_path", LAZY_EXPORT_INITIALIZERS, ids=str)
def test_lazy_public_names_are_visible_to_introspection(relative_path: Path) -> None:
    module_name = _initializer_module_name(relative_path)
    module = importlib.import_module(module_name)
    tree = ast.parse(
        (REPO_ROOT / relative_path).read_text(encoding="utf-8"),
        filename=str(relative_path),
    )
    lazy, eager, _mirrors = _lazy_export_contract(tree, module_name)
    advertised = set(module.__all__)

    assert advertised == set(lazy) | eager
    assert advertised <= set(dir(module))


@pytest.mark.parametrize("relative_path", LAZY_EXPORT_INITIALIZERS, ids=str)
def test_every_public_export_resolves(relative_path: Path) -> None:
    """Every name advertised through ``__all__`` resolves at runtime."""
    module_name = _initializer_module_name(relative_path)
    module = importlib.import_module(module_name)

    for name in module.__all__:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("relative_path", LAZY_EXPORT_INITIALIZERS, ids=str)
def test_lazy_export_maps_match_static_mirrors(relative_path: Path) -> None:
    """Runtime maps and static type mirrors must advertise the same names."""
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(relative_path))
    package = _initializer_module_name(relative_path)
    lazy, eager, mirrors = _lazy_export_contract(tree, package)
    missing_mirrors = lazy.keys() - mirrors.keys()
    extra_mirrors = mirrors.keys() - lazy.keys()
    mismatched_targets = {
        name: (lazy[name], mirrors[name])
        for name in lazy.keys() & mirrors.keys()
        if lazy[name] != mirrors[name]
    }
    missing_eager_bindings = eager - _module_bindings(tree)

    assert not missing_mirrors, (
        f"{relative_path}: lazy exports without static mirrors: {sorted(missing_mirrors)}"
    )
    assert not extra_mirrors, (
        f"{relative_path}: static mirrors without lazy exports: {sorted(extra_mirrors)}"
    )
    assert not mismatched_targets, (
        f"{relative_path}: lazy/static mirror target mismatches: {mismatched_targets}"
    )
    assert not missing_eager_bindings, (
        f"{relative_path}: eager exports without module bindings: {sorted(missing_eager_bindings)}"
    )
