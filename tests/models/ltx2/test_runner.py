"""Generic LTX2Runner host and public output contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import kinomlx.models.ltx2.runner as runner_module
from kinomlx.models.ltx2.pipelines.restart import DistilledRestart
from kinomlx.models.ltx2.runner import GenerationOutput, LTX2Error, LTX2Runner
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.reporting import RecordingReporter
from kinomlx.settings import Settings


@dataclass
class _Frames:
    spec: object = "signal"
    frame_count: int = 1
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class _FalseyReporter(RecordingReporter):
    def __bool__(self) -> bool:
        return False


def test_runner_supplies_one_prepared_plan_and_the_same_stateless_ports() -> None:
    resources = object()
    components = object()
    text_conditioner = object()
    reporter = _FalseyReporter()
    artifact_sink = object()
    calls = []

    def recipe(request, received_resources, **ports):
        calls.append((request, received_resources, ports))
        return GenerationOutput(frames=_Frames())

    request = DistilledRequest(prompt="first")
    runner = LTX2Runner(
        resources=resources,
        components=components,
        text_conditioner=text_conditioner,
        reporter=reporter,
        artifact_sink=artifact_sink,
    )
    output = runner.run(recipe, request)

    assert output.frame_count == 1
    assert output.signal == "signal"
    assert calls == [
        (
            request,
            resources,
            {
                "components": components,
                "text_conditioner": text_conditioner,
                "reporter": reporter,
                "artifact_sink": artifact_sink,
            },
        )
    ]
    assert runner.resources is resources


def test_runner_core_runs_distinct_recipe_request_types_without_branching() -> None:
    runner = LTX2Runner(resources=object(), components=object())
    events = []

    @dataclass(frozen=True)
    class OtherRequest:
        value: int

    def first(request, _resources, **_ports):
        events.append(("first", request.prompt))
        return GenerationOutput(frames=_Frames())

    def second(request, _resources, **_ports):
        events.append(("second", request.value))
        return GenerationOutput(frames=_Frames())

    runner.run(first, DistilledRequest(prompt="hello"))
    runner.run(second, OtherRequest(7))

    assert events == [("first", "hello"), ("second", 7)]


def test_runner_restart_uses_the_same_public_recipe_ports(monkeypatch, tmp_path) -> None:
    import kinomlx.models.ltx2.pipelines.restart as restart_module

    resources = object()
    components = object()
    text_conditioner = object()
    reporter = RecordingReporter()
    artifact_sink = object()
    calls = []

    def fake_restart(request, received_resources, *, restart, **ports):
        calls.append((request, received_resources, restart, ports))
        return GenerationOutput(frames=_Frames())

    monkeypatch.setattr(restart_module, "restart_distilled", fake_restart)
    request = DistilledRequest(prompt="", width=64, height=64, frames=9)
    restart = DistilledRestart.decode(tmp_path / "final.safetensors")
    runner = LTX2Runner(
        resources=resources,
        components=components,
        text_conditioner=text_conditioner,
        reporter=reporter,
        artifact_sink=artifact_sink,
    )

    output = runner.restart(request, restart)

    assert output.frame_count == 1
    assert calls == [
        (
            request,
            resources,
            restart,
            {
                "components": components,
                "text_conditioner": text_conditioner,
                "reporter": reporter,
                "artifact_sink": artifact_sink,
            },
        )
    ]


def test_runner_prepares_resources_once_at_construction(monkeypatch) -> None:
    expected = object()
    calls = []
    reporter = RecordingReporter()
    model_settings = LTX2Settings()
    infrastructure = Settings()
    monkeypatch.setattr(
        runner_module,
        "prepare_resources",
        lambda received, *, infrastructure, reporter: (
            calls.append((received, infrastructure, reporter)) or expected
        ),
    )

    runner = LTX2Runner(
        model_settings,
        infrastructure=infrastructure,
        reporter=reporter,
        components=object(),
    )

    assert runner.resources is expected
    assert calls == [(model_settings, infrastructure, reporter)]


def test_runner_rejects_ambiguous_resource_ownership() -> None:
    with pytest.raises(ValueError, match="settings or prepared resources"):
        LTX2Runner(LTX2Settings(), resources=object())


def test_runner_wraps_resource_and_recipe_operational_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "prepare_resources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing checkpoint")),
    )
    with pytest.raises(LTX2Error, match="cannot prepare.*missing checkpoint"):
        LTX2Runner(LTX2Settings())

    def failing_recipe(*_args, **_kwargs):
        raise RuntimeError("denoise failed")

    runner = LTX2Runner(resources=object(), components=object())
    with pytest.raises(LTX2Error, match="generation failed.*denoise failed"):
        runner.run(failing_recipe, object())


def test_recipe_must_return_generation_output() -> None:
    runner = LTX2Runner(resources=object(), components=object())

    with pytest.raises(TypeError, match="must return GenerationOutput"):
        runner.run(lambda *_args, **_kwargs: object(), object())


def test_generation_output_context_closes_an_unconsumed_stream() -> None:
    frames = _Frames()
    with GenerationOutput(frames=frames) as output:
        assert output.frames is frames
        assert not frames.closed
    assert frames.closed


def test_generation_output_collects_lazy_runtime_diagnostics() -> None:
    frames = _Frames()
    payload = {"vae_decode": {"tiling": {"total_tiles": 1}}}
    output = GenerationOutput(
        frames=frames,
        diagnostics_provider=lambda: payload,
    )

    assert output.runtime_diagnostics() is payload
    assert GenerationOutput(frames=frames).runtime_diagnostics() == {}
