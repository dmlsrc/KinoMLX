"""GMNet output naming, reservations, and all-or-nothing publication."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.io.reservation import reservation_path
from kinomlx.models.gmnet.catalog import GMNetVariant, variant_spec
from kinomlx.models.gmnet.expand import ExpansionResult
from kinomlx.models.gmnet.output import (
    GMNetOutputConfig,
    GMNetOutputError,
    GMNetOutputSink,
    plan_gmnet_output,
)
from kinomlx.models.gmnet.types import GMNetRequest


def _request(tmp_path: Path, name: str = "source.png") -> GMNetRequest:
    source = tmp_path / name
    source.write_bytes(b"source")
    return GMNetRequest(source)


def _result() -> ExpansionResult:
    return ExpansionResult(
        linear_rgb=mx.ones((2, 3, 3), dtype=mx.float32),
        gain_map=mx.full((2, 3), 0.5, dtype=mx.float32),
        qmax_normalized=0.75,
        spec=variant_spec(GMNetVariant.REALWORLD),
    )


def _writer(payload: bytes):
    def write(_result, path: Path):
        path.write_bytes(payload)
        return path

    return write


def _gain_writer(payload: bytes):
    def write(_result, path: Path, _source: Path):
        path.write_bytes(payload)
        return path

    return write


def _sink(plan, *, exr=b"exr", heic=b"heic", gain=b"gain"):
    return GMNetOutputSink(
        plan,
        exr_writer=_writer(exr),
        heic_writer=_writer(heic),
        gain_writer=_gain_writer(gain),
    )


def test_exact_exr_path_writes_only_that_artifact_by_default(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact))
    artifacts = _sink(plan).write(_result())
    assert artifacts.exr == exact
    assert artifacts.heic is None
    assert exact.read_bytes() == b"exr"
    assert not (tmp_path / "chosen.heic").exists()


def test_exact_path_can_add_the_other_display_format_explicitly(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact, heic=True))
    artifacts = _sink(plan).write(_result())
    assert artifacts.exr == exact
    assert artifacts.heic == tmp_path / "chosen.heic"


def test_derived_output_defaults_to_both_formats_and_input_stem(tmp_path) -> None:
    request = _request(tmp_path, "my.photo.png")
    output = tmp_path / "outputs"
    plan = plan_gmnet_output(request, GMNetOutputConfig(directory=output))
    assert plan.artifacts.exr == output / "my.photo.exr"
    assert plan.artifacts.heic == output / "my.photo.heic"


def test_output_prefix_and_gain_map_are_resolved_together(tmp_path) -> None:
    request = _request(tmp_path)
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(
            directory=tmp_path / "out",
            prefix="expanded",
            exr=False,
            heic=True,
            save_gain_map=True,
        ),
    )
    artifacts = _sink(plan).write(_result())
    assert artifacts.exr is None
    assert artifacts.heic == tmp_path / "out" / "expanded.heic"
    assert artifacts.gain_map == tmp_path / "out" / "expanded.gain_map.safetensors"


def test_execution_sidecars_share_the_resolved_stem_and_are_reserved(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    run_log = tmp_path / "chosen_run.json"
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(
            path=exact,
            save_run_log=True,
            save_console_log=True,
            save_effective_config=True,
        ),
    )
    assert plan.sidecar_paths() == {
        "run_log": run_log,
        "execution_log": tmp_path / "chosen_console.log",
        "effective_config": tmp_path / "chosen_config.toml",
    }

    run_log.write_text("existing", encoding="utf-8")
    with pytest.raises(GMNetOutputError, match="exists"):
        plan.reserve().__enter__()
    assert run_log.read_text(encoding="utf-8") == "existing"


def test_library_save_all_inherits_sidecars_and_preserves_explicit_opt_outs(
    tmp_path,
) -> None:
    request = _request(tmp_path)
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(
            path=tmp_path / "chosen.exr",
            save_all_sidecars=True,
            save_console_log=False,
        ),
    )

    assert plan.artifacts.gain_map == tmp_path / "chosen.gain_map.safetensors"
    assert plan.sidecar_paths() == {
        "run_log": tmp_path / "chosen_run.json",
        "effective_config": tmp_path / "chosen_config.toml",
    }


def test_existing_target_is_refused_before_any_writer_runs(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    exact.write_bytes(b"old")
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact))
    calls = []

    def writer(_result, path):
        calls.append(path)
        path.write_bytes(b"new")

    with pytest.raises(GMNetOutputError, match="exists"):
        GMNetOutputSink(plan, exr_writer=writer).write(_result())
    assert calls == []
    assert exact.read_bytes() == b"old"


def test_writer_failure_leaves_forced_existing_bundle_untouched(tmp_path) -> None:
    request = _request(tmp_path)
    exr = tmp_path / "chosen.exr"
    heic = tmp_path / "chosen.heic"
    exr.write_bytes(b"old-exr")
    heic.write_bytes(b"old-heic")
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(path=exr, heic=True, force=True),
    )

    def fail(_result, _path):
        raise RuntimeError("HEIC encoder stopped")

    sink = GMNetOutputSink(plan, exr_writer=_writer(b"new-exr"), heic_writer=fail)
    with pytest.raises(GMNetOutputError, match="HEIC encoder stopped"):
        sink.write(_result())
    assert exr.read_bytes() == b"old-exr"
    assert heic.read_bytes() == b"old-heic"


def test_publication_failure_rolls_back_every_forced_target(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path)
    exr = tmp_path / "chosen.exr"
    heic = tmp_path / "chosen.heic"
    exr.write_bytes(b"old-exr")
    heic.write_bytes(b"old-heic")
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(path=exr, heic=True, force=True),
    )
    original_rename = Path.rename

    def flaky_rename(self: Path, target: Path):
        if self.name == "chosen.heic" and self.parent.name.startswith(".gmnet-output-"):
            raise OSError("injected publish failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(GMNetOutputError, match="injected publish failure"):
        _sink(plan, exr=b"new-exr", heic=b"new-heic").write(_result())
    assert exr.read_bytes() == b"old-exr"
    assert heic.read_bytes() == b"old-heic"


def test_failed_new_bundle_releases_all_reservations(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact, heic=True))

    def fail(_result, _path):
        raise RuntimeError("writer failed")

    with pytest.raises(GMNetOutputError):
        GMNetOutputSink(plan, exr_writer=_writer(b"new"), heic_writer=fail).write(_result())
    assert not exact.exists()
    assert not (tmp_path / "chosen.heic").exists()
    assert not reservation_path(exact).exists()
    assert not reservation_path(tmp_path / "chosen.heic").exists()


def test_reservation_uses_hidden_markers_not_final_artifact_names(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    heic = tmp_path / "chosen.heic"
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact, heic=True))

    with plan.reserve():
        assert not exact.exists()
        assert not heic.exists()
        assert reservation_path(exact).is_file()
        assert reservation_path(heic).is_file()

    assert not reservation_path(exact).exists()
    assert not reservation_path(heic).exists()


def test_stale_reservation_marker_has_actionable_refusal(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    marker = reservation_path(exact)
    marker.write_text("interrupted run")
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact))

    with pytest.raises(GMNetOutputError, match="remove that marker and retry"):
        plan.reserve().__enter__()

    assert marker.read_text() == "interrupted run"


def test_forced_replacement_preserves_existing_permissions(tmp_path) -> None:
    request = _request(tmp_path)
    exact = tmp_path / "chosen.exr"
    exact.write_bytes(b"old")
    exact.chmod(0o640)
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=exact, force=True))
    _sink(plan, exr=b"new").write(_result())
    assert exact.read_bytes() == b"new"
    assert exact.stat().st_mode & 0o777 == 0o640


def test_source_overwrite_and_unknown_exact_suffix_are_refused(tmp_path) -> None:
    request = _request(tmp_path, "source.heic")
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=request.image))
    with pytest.raises(GMNetOutputError, match="source image"):
        plan.reserve().__enter__()
    with pytest.raises(GMNetOutputError, match="must end in"):
        plan_gmnet_output(request, GMNetOutputConfig(path=tmp_path / "output.png"))


def test_symbolic_link_output_is_refused_explicitly(tmp_path) -> None:
    request = _request(tmp_path)
    existing = tmp_path / "existing.exr"
    existing.write_bytes(b"existing")
    target = tmp_path / "chosen.exr"
    target.symlink_to(existing)
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(path=target, force=True),
    )

    with pytest.raises(GMNetOutputError, match="symbolic links"):
        plan.reserve().__enter__()

    assert target.is_symlink()
    assert existing.read_bytes() == b"existing"


def test_external_reservation_must_be_active(tmp_path) -> None:
    request = _request(tmp_path)
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=tmp_path / "chosen.exr"))
    reservation = plan.reserve()

    with pytest.raises(ValueError, match="not active"):
        _sink(plan).write(_result(), reservation=reservation)


def test_external_reservation_remains_active_for_execution_sidecars(tmp_path) -> None:
    request = _request(tmp_path)
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(
            path=tmp_path / "chosen.exr",
            save_run_log=True,
        ),
    )

    with plan.reserve() as reservation:
        _sink(plan).write(_result(), reservation=reservation)
        assert reservation.active
        assert reservation_path(tmp_path / "chosen_run.json").is_file()


def test_reservation_cannot_be_entered_twice(tmp_path) -> None:
    request = _request(tmp_path)
    plan = plan_gmnet_output(request, GMNetOutputConfig(path=tmp_path / "chosen.exr"))
    with (
        plan.reserve() as reservation,
        pytest.raises(
            GMNetOutputError,
            match="already active",
        ),
    ):
        reservation.__enter__()


def test_rollback_failure_retains_the_prior_artifact_for_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    request = _request(tmp_path)
    exr = tmp_path / "chosen.exr"
    heic = tmp_path / "chosen.heic"
    exr.write_bytes(b"old-exr")
    heic.write_bytes(b"old-heic")
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(path=exr, heic=True, force=True),
    )
    original_rename = Path.rename

    def broken_publish_and_restore(self: Path, target: Path):
        if self.name == "chosen.heic" and self.parent.name.startswith(".gmnet-output-"):
            raise OSError("injected publish failure")
        if self.name == "prior-1":
            raise OSError("injected restore failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", broken_publish_and_restore)
    with pytest.raises(GMNetOutputError, match="Recovery files were retained") as caught:
        _sink(plan, exr=b"new-exr", heic=b"new-heic").write(_result())

    transactions = list(tmp_path.glob(".gmnet-output-*"))
    assert len(transactions) == 1
    assert str(transactions[0]) in str(caught.value)
    assert (transactions[0] / "prior-1").read_bytes() == b"old-heic"
    assert exr.read_bytes() == b"old-exr"
    assert not heic.exists()
