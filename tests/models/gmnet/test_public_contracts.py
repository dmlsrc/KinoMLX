"""The documented GMNet public-API example is importable and executable."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).parents[3]


def _load_example(name: str):
    example_path = _REPO / "examples" / "gmnet_expand.py"
    spec = importlib.util.spec_from_file_location(name, example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_gmnet_example_imports_and_exposes_composition() -> None:
    module = _load_example("kinomlx_example_gmnet")

    assert callable(module.expand_still)


def test_external_gmnet_example_executes_through_public_ports(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_example("kinomlx_example_gmnet_synthetic")
    source = tmp_path / "source.png"
    output = tmp_path / "result.exr"
    artifact = SimpleNamespace(exr=output, heic=None, gain_map=None)
    calls: list[tuple[str, object]] = []

    infrastructure = object()
    model_settings = object()
    resources = object()
    request = object()
    plan = SimpleNamespace(reserve=lambda: nullcontext(object()))
    expansion = object()

    monkeypatch.setattr(
        module,
        "Settings",
        SimpleNamespace(from_env=lambda: infrastructure),
    )
    monkeypatch.setattr(
        module,
        "GMNetSettings",
        SimpleNamespace(from_env=lambda: model_settings),
    )
    monkeypatch.setattr(
        module,
        "prepare_gmnet_resources",
        lambda selected, *, infrastructure: (
            calls.append(("resources", (selected, infrastructure))) or resources
        ),
    )
    monkeypatch.setattr(
        module,
        "GMNetRequest",
        lambda *, image: calls.append(("request", image)) or request,
    )
    monkeypatch.setattr(
        module,
        "plan_gmnet_output",
        lambda selected, config: calls.append(("plan", (selected, config))) or plan,
    )

    class FakeRunner:
        def __init__(self, *, resources):
            calls.append(("runner", resources))

        def run(self, recipe, selected):
            calls.append(("run", (recipe, selected)))
            return expansion

    class FakeSink:
        def __init__(self, selected):
            calls.append(("sink", selected))

        def write(self, result, *, reservation):
            calls.append(("write", (result, reservation)))
            return artifact

    monkeypatch.setattr(module, "GMNetRunner", FakeRunner)
    monkeypatch.setattr(module, "GMNetOutputSink", FakeSink)

    produced = module.expand_still(source, output=output)

    assert produced is artifact
    assert calls[0] == ("resources", (model_settings, infrastructure))
    assert calls[1] == ("request", source)
    assert calls[3] == ("runner", resources)
    assert calls[4] == ("run", (module.expand_gmnet, request))
    assert calls[6][0] == "write"
