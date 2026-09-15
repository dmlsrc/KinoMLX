"""Explicit heavy-test selection and test-owned application cache state."""

import ast
from pathlib import Path

import pytest

from tests._logging import package_logger_state

HEAVY_IN_WORKER = (
    "heavy tests run in one process: they load real checkpoints and wire MLX memory. "
    "Run them with -n 0 (--run-heavy already disables the workers for you)."
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Include heavy tests that read real model checkpoints or run long GPU work.",
    )


_HEAVY_SERIAL = pytest.StashKey[bool]()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Run heavy tests in one process even though the suite defaults to -n auto.

    Heavy tests load multi-gigabyte checkpoints and wire MLX memory, so several
    workers doing that at once would compete for the same unified memory. xdist
    resolves ``auto`` to a worker count in its ``pytest_cmdline_main`` and only
    builds the distributed session in its trylast ``pytest_configure``, so
    clearing the worker list here keeps a ``--run-heavy`` run serial.
    """
    serial = bool(config.getoption("--run-heavy") and getattr(config.option, "numprocesses", 0))
    if serial:
        config.option.numprocesses = 0
        config.option.dist = "no"
        config.option.tx = []
    config.stash[_HEAVY_SERIAL] = serial


def pytest_report_header(config):
    if config.stash.get(_HEAVY_SERIAL, False):
        return "heavy tests: parallel workers disabled, running in one process"
    return None


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return
    selected, excluded = [], []
    for item in items:
        (excluded if item.get_closest_marker("heavy") else selected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)


def pytest_runtest_setup(item):
    """Refuse a heavy test inside an xdist worker, whatever flags reached it.

    ``pytest_configure`` clears the worker list for a ``--run-heavy`` run, but a
    heavy test can still reach a worker another way (an explicit ``-m heavy``
    selection beside ``-n``, a custom invocation, a future option). Failing at
    setup stops it before any checkpoint is read. ``workerinput`` exists only on
    an xdist worker's config, so this needs no import of xdist.
    """
    if item.get_closest_marker("heavy") and hasattr(item.config, "workerinput"):
        pytest.fail(HEAVY_IN_WORKER, pytrace=False)


@pytest.fixture(autouse=True, scope="session")
def product_memory_policy():
    """Run the session under the product's MLX allocator and residency policy.

    Real-weight tests that load components directly would otherwise run with
    MLX's defaults instead of the limits every product entry point applies.
    Both values follow the same environment overrides as the product.
    """
    from kinomlx.memory_policy import apply_memory_policy
    from kinomlx.settings import Settings

    settings = Settings.from_env_fields("mlx_cache_limit_gb", "mlx_wired_limit_gb")
    apply_memory_policy(settings)


@pytest.fixture(autouse=True)
def isolated_application_cache(tmp_path, monkeypatch):
    """Keep application artifacts in test-owned storage, including real-weight gates."""
    monkeypatch.setenv("KINO_APP_CACHE_DIR", str(tmp_path / "application-cache"))


@pytest.fixture(autouse=True)
def isolated_package_logger():
    """Keep one test's logging configuration out of the next test's assertions."""
    with package_logger_state():
        yield


@pytest.fixture(scope="session")
def parsed_project_sources():
    """Parse immutable runtime/tool source once for independent policy checks."""
    root = Path(__file__).resolve().parents[1]
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for directory in (root / "kinomlx", root / "scripts")
        for path in sorted(directory.rglob("*.py"))
    }
