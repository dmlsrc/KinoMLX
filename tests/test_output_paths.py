"""App-wide output naming and repeat-run behavior without model inference."""

import json
from contextlib import ExitStack
from datetime import datetime

import pytest

from kinomlx.cli.main import main
from kinomlx.io import output_paths
from kinomlx.io.output_paths import (
    OutputCollisionError,
    reserve_output_base,
    timestamped_output_base,
)
from kinomlx.io.reservation import PathReservation, reservation_path

_NOW = datetime(2026, 9, 1, 21, 1, 42)
_STAMP = "20260901_210142"


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return _NOW

    monkeypatch.setattr(output_paths, "datetime", Clock)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (None, "kinomlx"),
        (" ._ ", "kinomlx"),
        ("sample render!", "sample_render"),
        ("draft.v2", "draft.v2"),
        ("../outside/name", "outside_name"),
    ],
)
def test_generated_stem_is_format_independent_and_stays_in_directory(
    tmp_path,
    prefix,
    expected,
) -> None:
    assert timestamped_output_base(tmp_path, prefix, now=_NOW) == (
        tmp_path / f"{expected}_{_STAMP}"
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("generated", [False, True])
def test_only_generated_names_retry_collisions(tmp_path, generated) -> None:
    base = tmp_path / f"draft.v2_{_STAMP}"
    attempts = []

    def reserve(candidate):
        attempts.append(candidate)
        if len(attempts) < 3:
            raise OutputCollisionError("occupied")
        return candidate

    if generated:
        assert reserve_output_base(base, reserve, generated=True) == base.with_name(
            f"{base.name}_3"
        )
        assert len(attempts) == 3
    else:
        with pytest.raises(OutputCollisionError, match="occupied"):
            reserve_output_base(base, reserve, generated=False)
        assert attempts == [base]


def test_permission_failure_does_not_retry_names(tmp_path) -> None:
    attempts = []

    def reserve(candidate):
        attempts.append(candidate)
        raise PermissionError("not writable")

    with pytest.raises(PermissionError, match="not writable"):
        reserve_output_base(tmp_path / "run", reserve, generated=True)
    assert len(attempts) == 1


class _Reporter:
    def memory_checkpoint(self, _name):
        pass

    def memory_to_dict(self):
        return {}

    def to_dict(self):
        return {"total_seconds": 0.1, "phases": []}


def _stub_command(model, tmp_path, monkeypatch):
    plans = []

    def run(_invocation, _reporter, *, plan, reservation, output_sink=None):
        assert reservation.active
        assert reservation.plan == plan
        assert all(reservation_path(path).is_file() for path in plan.targets())
        plans.append(plan)
        for path in plan.artifacts.paths():
            path.write_bytes(b"artifact")
        return {
            "action": "generate" if model == "krea2" else "expand",
            "outputs": plan.artifacts.to_dict(),
            "generation": {},
            "warnings": [],
        }

    monkeypatch.setattr(
        f"kinomlx.models.{model}.command.command_reporter",
        lambda _stack, _settings: _Reporter(),
    )
    action = "_run_generate" if model == "krea2" else "_run_expand"
    monkeypatch.setattr(f"kinomlx.models.{model}.command.{action}", run)
    if model == "krea2":
        arguments = ["--prompt", "test", "--seed", "42"]
    else:
        source = tmp_path / "photo.png"
        source.write_bytes(b"source")
        arguments = ["--image", str(source)]
    return ["--model", model, *arguments, "--save-all-sidecars", "--json"], plans


@pytest.mark.parametrize("model", ["krea2", "gmnet"])
@pytest.mark.parametrize("prefix", [None, "draft.v2"])
def test_still_commands_repeat_safely_with_one_timestamped_bundle_per_run(
    tmp_path,
    monkeypatch,
    capsys,
    fixed_clock,
    model,
    prefix,
) -> None:
    arguments, plans = _stub_command(model, tmp_path, monkeypatch)
    arguments += ["--output-dir", str(tmp_path)]
    if prefix is not None:
        arguments += ["--output-prefix", prefix]
    name = prefix or ("krea2-turbo-seed42" if model == "krea2" else "photo")
    legacy = tmp_path / f"{name}.{'png' if model == 'krea2' else 'exr'}"
    legacy.write_bytes(b"old output")

    for suffix in ("", "_2"):
        assert main(arguments) == 0
        payload = json.loads(capsys.readouterr().out)
        plan = plans[-1]
        assert plan.base == tmp_path / f"{name}_{_STAMP}{suffix}"
        assert all(path.name.startswith(plan.base.name) for path in plan.targets())
        assert all(path.is_file() for path in plan.targets())
        assert not any(reservation_path(path).exists() for path in plan.targets())
        record = json.loads(plan.sidecar_paths()["run_log"].read_text())
        assert record["status"] == "completed"
        assert record["outputs"] == payload["outputs"]
    assert set(plans[0].targets()).isdisjoint(plans[1].targets())
    assert legacy.read_bytes() == b"old output"


@pytest.mark.parametrize("model", ["krea2", "gmnet"])
@pytest.mark.parametrize("prefix", [None, "draft.v2"])
def test_force_without_exact_path_is_rejected_before_output_planning(
    tmp_path, monkeypatch, capsys, model, prefix
) -> None:
    arguments, plans = _stub_command(model, tmp_path, monkeypatch)
    arguments += ["--output-dir", str(tmp_path), "--force"]
    if prefix is not None:
        arguments += ["--output-prefix", prefix]
    assert main(arguments) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "force requires an exact output path" in payload["error"]
    assert plans == []
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))


@pytest.mark.parametrize("model", ["krea2", "gmnet"])
@pytest.mark.parametrize("occupied", ["artifact", "sidecar", "reservation"])
def test_still_commands_skip_an_occupied_bundle_before_running(
    tmp_path,
    monkeypatch,
    capsys,
    fixed_clock,
    model,
    occupied,
) -> None:
    arguments, plans = _stub_command(model, tmp_path, monkeypatch)
    arguments += ["--output-dir", str(tmp_path), "--output-prefix", "run"]
    stem = f"run_{_STAMP}"
    suffix = "_run.json" if occupied == "sidecar" else (".png" if model == "krea2" else ".heic")
    target = tmp_path / f"{stem}{suffix}"
    with ExitStack() as stack:
        if occupied == "reservation":
            marker = PathReservation.acquire(target)
            stack.callback(marker.release)
        else:
            target.write_bytes(b"previous run")
        assert main(arguments) == 0
        capsys.readouterr()
        assert plans[0].base.name == f"{stem}_2"
        if occupied == "reservation":
            assert marker.active
            assert not target.exists()
        else:
            assert target.read_bytes() == b"previous run"
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))


@pytest.mark.parametrize("model", ["krea2", "gmnet"])
def test_still_explicit_path_is_literal_and_existing_output_is_refused(
    tmp_path,
    monkeypatch,
    capsys,
    fixed_clock,
    model,
) -> None:
    arguments, plans = _stub_command(model, tmp_path, monkeypatch)
    target = tmp_path / f"chosen.v2.{('tiff' if model == 'krea2' else 'exr')}"
    arguments += ["--output", str(target)]
    assert main(arguments) == 0
    capsys.readouterr()
    assert target in plans[0].artifacts.paths()
    assert plans[0].base == target.with_suffix("")

    assert main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert len(plans) == 1
    assert target.read_bytes() == b"artifact"
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))
