"""Smoke test that the dev scripts still import.

``scripts/`` isn't a package, so the import gate in ``test_imports.py`` doesn't
reach it - and it quietly rotted once when ``kinomlx.ui``'s API changed
underneath it. Importing each script here re-runs its top-level imports, so an
API drift fails in the suite instead of on someone's terminal. ``main()`` stays
guarded by ``__name__``, so loading the module runs no scenarios.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_demo_ui_imports() -> None:
    spec = importlib.util.spec_from_file_location("demo_ui", _SCRIPTS / "demo_ui.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top-level imports + defs only, not main()
    assert module.SCENARIOS  # catalog populated -> module loaded fully
