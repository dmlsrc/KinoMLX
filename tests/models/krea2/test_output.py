"""Krea native lossless still publication and rollback tests."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.output_bundle import BundleOutputError
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.io.reservation import reservation_path
from kinomlx.models.krea2.output import (
    Krea2OutputSink,
    Krea2StillOutputError,
    Krea2StillOutputPlan,
    Krea2StillOutputSink,
    plan_krea2_output,
)
from kinomlx.models.krea2.pipelines import Krea2GeneratedImage, Krea2GenerationOutput
from kinomlx.models.krea2.text_conditioning import (
    Krea2PromptConditioning,
    Krea2TextConditioning,
)
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.models.krea2.vae import Krea2DecodedRGB
from kinomlx.restart_selection import fingerprint_key
from kinomlx.videotoolbox.images import load_image_rgb


def _decoded(value: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Krea2DecodedRGB:
    pixels = mx.broadcast_to(mx.array(value, dtype=mx.float32), (1, 2, 3, 3))
    return Krea2DecodedRGB(pixels)


def test_native_png_publication_preserves_dimensions_channel_order_and_byte_conversion(
    tmp_path: Path,
) -> None:
    target = tmp_path / "image.png"
    Krea2StillOutputSink(Krea2StillOutputPlan(target)).write(_decoded((-1.0, 0.0, 1.0)))

    encoded = load_image_rgb(target)
    assert tuple(encoded.shape) == (2, 3, 3)
    assert mx.array_equal(encoded[0, 0], mx.array([0, 127, 255], dtype=mx.uint8)).item()


def test_output_station_applies_the_only_display_transfer_and_clamp(tmp_path: Path) -> None:
    captured: list[mx.array] = []

    def writer(path: Path, image: mx.array) -> None:
        captured.append(image)
        path.write_bytes(b"image")

    Krea2StillOutputSink(
        Krea2StillOutputPlan(tmp_path / "image.png"),
        writer=writer,
    ).write(_decoded((2.0, 0.0, -2.0)))

    assert len(captured) == 1
    assert mx.array_equal(captured[0][0, 0], mx.array([1.0, 0.5, 0.0])).item()


@pytest.mark.parametrize("suffix", [".png", ".tif", ".tiff"])
def test_plan_accepts_only_native_lossless_still_formats(tmp_path: Path, suffix: str) -> None:
    assert Krea2StillOutputPlan(tmp_path / f"image{suffix}").path.suffix == suffix
    with pytest.raises(ValueError, match="must end in"):
        Krea2StillOutputPlan(tmp_path / "image.jpg")


def test_existing_target_is_refused_before_writer_runs(tmp_path: Path) -> None:
    target = tmp_path / "image.png"
    target.write_bytes(b"old")
    calls: list[Path] = []

    def writer(path: Path, _image: mx.array) -> None:
        calls.append(path)

    with pytest.raises(Krea2StillOutputError, match="already exists"):
        Krea2StillOutputSink(Krea2StillOutputPlan(target), writer=writer).write(_decoded())
    assert calls == []
    assert target.read_bytes() == b"old"


def test_encode_failure_preserves_forced_target_and_cleans_partial(tmp_path: Path) -> None:
    target = tmp_path / "image.png"
    target.write_bytes(b"old")

    def writer(path: Path, _image: mx.array) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("encoder stopped")

    plan = Krea2StillOutputPlan(target, force=True)
    with pytest.raises(Krea2StillOutputError, match="encoder stopped"):
        Krea2StillOutputSink(plan, writer=writer).write(_decoded())
    assert target.read_bytes() == b"old"
    assert not reservation_path(target).exists()
    assert not list(tmp_path.glob(".*.tmp.png"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("create_empty", [False, True])
def test_empty_writer_result_is_not_published(tmp_path: Path, existing, create_empty) -> None:
    target = tmp_path / "image.png"
    if existing:
        target.write_bytes(b"old")

    def writer(path: Path, _image: mx.array) -> None:
        if create_empty:
            path.write_bytes(b"")

    with pytest.raises(Krea2StillOutputError, match="nonempty regular file"):
        Krea2StillOutputSink(Krea2StillOutputPlan(target, force=existing), writer=writer).write(
            _decoded()
        )
    if existing:
        assert target.read_bytes() == b"old"
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


def test_cancellation_after_encode_publishes_nothing_and_releases_marker(
    tmp_path: Path,
) -> None:
    target = tmp_path / "image.png"
    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise RuntimeError("canceled")

    def writer(path: Path, _image: mx.array) -> None:
        path.write_bytes(b"complete-but-not-published")

    with pytest.raises(Krea2StillOutputError, match="canceled"):
        Krea2StillOutputSink(Krea2StillOutputPlan(target), writer=writer).write(
            _decoded(),
            checkpoint=checkpoint,
        )
    assert not target.exists()
    assert not reservation_path(target).exists()


def test_process_control_after_encode_is_not_swallowed_and_cleans_partial(
    tmp_path: Path,
) -> None:
    target = tmp_path / "image.png"
    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise KeyboardInterrupt

    def writer(path: Path, _image: mx.array) -> None:
        path.write_bytes(b"complete-but-not-published")

    with pytest.raises(KeyboardInterrupt):
        Krea2StillOutputSink(Krea2StillOutputPlan(target), writer=writer).write(
            _decoded(),
            checkpoint=checkpoint,
        )
    assert not target.exists()
    assert not reservation_path(target).exists()
    assert not list(tmp_path.glob(".*.tmp.png"))


def test_external_reservation_stays_active_until_caller_releases_it(tmp_path: Path) -> None:
    target = tmp_path / "image.png"

    def writer(path: Path, _image: mx.array) -> None:
        path.write_bytes(b"image")

    plan = Krea2StillOutputPlan(target)
    with plan.reserve() as reservation:
        Krea2StillOutputSink(plan, writer=writer).write(
            _decoded(),
            reservation=reservation,
        )
        assert reservation.active
        assert reservation_path(target).is_file()
    assert target.read_bytes() == b"image"
    assert not reservation_path(target).exists()


def test_symbolic_link_target_is_refused_even_with_force(tmp_path: Path) -> None:
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"old")
    target = tmp_path / "image.png"
    target.symlink_to(existing)
    with pytest.raises(Krea2StillOutputError, match="symbolic links"):
        Krea2StillOutputSink(Krea2StillOutputPlan(target, force=True)).write(_decoded())
    assert existing.read_bytes() == b"old"


def test_nonfinite_injected_decoder_is_rejected_before_reservation(tmp_path: Path) -> None:
    target = tmp_path / "image.png"
    decoded = Krea2DecodedRGB(mx.full((1, 2, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="nonfinite RGB"):
        Krea2StillOutputSink(Krea2StillOutputPlan(target)).write(decoded)
    assert not target.exists()
    assert not reservation_path(target).exists()


def _one_image(index: int) -> Krea2GenerationOutput:
    """One seed's result, shaped the way the bundle sink receives each image."""
    return Krea2GenerationOutput(
        images=(
            Krea2GeneratedImage(
                pixels=mx.full((2, 3, 3), float(index) / 10.0, dtype=mx.float32),
                prompt=f"prompt {index}",
                seed=42 + index,
                index=index,
            ),
        ),
        initial_noise=mx.zeros((1, 16, 1, 1), dtype=mx.bfloat16),
        final_latents=mx.ones((1, 16, 1, 1), dtype=mx.bfloat16),
    )


def _generation(count: int = 1, *, include_text: bool = False) -> Krea2GenerationOutput:
    images = tuple(
        Krea2GeneratedImage(
            pixels=mx.full((2, 3, 3), float(index) / 10.0, dtype=mx.float32),
            prompt=f"prompt {index}",
            seed=42 + index,
            index=index,
        )
        for index in range(count)
    )
    text = None
    if include_text:
        text = Krea2TextConditioning(
            positive=Krea2PromptConditioning(
                hidden_states=mx.zeros((count, 4, 1, 2), dtype=mx.bfloat16),
                attention_mask=mx.ones((count, 4), dtype=mx.bool_),
                position_ids=mx.zeros((3, count, 4), dtype=mx.int32),
                prompts=tuple(f"prompt {index}" for index in range(count)),
                token_identity="fixture-tokens",
            ),
            negative=None,
            provenance=SimpleNamespace(to_metadata=lambda: {"identity": "fixture"}),
        )
    return Krea2GenerationOutput(
        images=images,
        initial_noise=mx.zeros((count, 16, 1, 1), dtype=mx.bfloat16),
        final_latents=mx.ones((count, 16, 1, 1), dtype=mx.bfloat16),
        text_conditioning=text,
    )


def test_multi_image_plan_uses_one_stem_and_reserves_every_sidecar(tmp_path: Path) -> None:
    request = Krea2Request(("one", "two"), images_per_prompt=2)
    plan = plan_krea2_output(
        request,
        "turbo",
        ImageOutputConfig(
            directory=tmp_path,
            prefix="draft",
            save_all_sidecars=True,
        ),
    )

    stem = plan.base.name
    assert stem.startswith("draft_")
    # Consecutive seeds 0 and 1 repeat for each of the two prompts.
    names = tuple(f"{stem}-{index:03d}-seed{(index - 1) % 2}" for index in range(1, 5))
    assert plan.artifacts.images == tuple(tmp_path / f"{name}.png" for name in names)
    assert plan.artifacts.initial_noise == tuple(
        tmp_path / f"{name}_initial_noise.safetensors" for name in names
    )
    assert plan.artifacts.final_latents == tuple(
        tmp_path / f"{name}_final_latents.safetensors" for name in names
    )
    assert plan.artifacts.text_conditioning == tuple(
        tmp_path / f"{name}_text.safetensors" for name in names
    )
    assert plan.sidecar_paths() == {
        "run_log": tmp_path / f"{stem}_run.json",
        "execution_log": tmp_path / f"{stem}_console.log",
        "effective_config": tmp_path / f"{stem}_config.toml",
    }
    with plan.reserve():
        assert all(reservation_path(path).is_file() for path in plan.targets())
    assert all(not reservation_path(path).exists() for path in plan.targets())


def test_exact_multi_image_path_is_a_numbered_stem(tmp_path: Path) -> None:
    request = Krea2Request("test", images_per_prompt=2)
    plan = plan_krea2_output(
        request,
        "raw",
        ImageOutputConfig(path=tmp_path / "chosen.tiff"),
    )
    assert plan.artifacts.images == (
        tmp_path / "chosen-001-seed0.tiff",
        tmp_path / "chosen-002-seed1.tiff",
    )


def test_generation_writes_only_separate_image_and_tensor_files_per_seed(
    tmp_path: Path,
) -> None:
    request = Krea2Request("test", images_per_prompt=2)
    plan = plan_krea2_output(
        request,
        "turbo",
        ImageOutputConfig(
            directory=tmp_path,
            prefix="bundle",
            save_initial_noise=True,
            save_final_latents=True,
            save_text_conditioning=True,
        ),
    )

    def writer(path: Path, _image: mx.array) -> None:
        path.write_bytes(b"image")

    result = _generation(2, include_text=True)
    sink = Krea2OutputSink(plan, image_writer=writer)
    artifacts = sink.write(result)

    assert all(path.read_bytes() == b"image" for path in artifacts.images)
    expected_fingerprints = {}
    for index in range(2):
        selected = plan.seed_artifacts(index)
        assert mx.load(str(selected.initial_noise[0]))["initial_noise"].shape[0] == 1
        assert mx.load(str(selected.final_latents[0]))["final_latents"].shape[0] == 1
        text = mx.load(str(selected.text_conditioning[0]))
        assert text["positive_hidden_states"].shape[0] == 1
        assert text["positive_position_ids"].shape[1] == 1
        for name in ("initial_noise", "final_latents", "text_conditioning"):
            expected_fingerprints[f"{name}.{index}"] = file_sha256(selected.single_path(name))
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))
    assert sink.fingerprints == expected_fingerprints
    assert set(tmp_path.iterdir()) == set(plan.artifacts.paths())
    assert len(plan.artifacts.paths()) == 8
    assert all(
        isinstance(artifacts.to_dict()[name], list)
        for name in ("initial_noise", "final_latents", "text_conditioning")
    )
    assert sink.fingerprint_errors == {}


def test_output_fingerprint_failure_is_advisory(tmp_path: Path, monkeypatch) -> None:
    plan = plan_krea2_output(
        Krea2Request("test"),
        "turbo",
        ImageOutputConfig(directory=tmp_path, save_final_latents=True),
    )

    def unavailable(_path):
        raise PermissionError("receipt unavailable")

    monkeypatch.setattr("kinomlx.models.krea2.output.file_sha256", unavailable)
    sink = Krea2OutputSink(plan, image_writer=lambda path, _pixels: path.write_bytes(b"image"))
    artifacts = sink.write(_generation(1))

    assert all(path.is_file() for path in artifacts.paths())
    assert sink.fingerprints == {}
    assert sink.fingerprint_errors == {"final_latents": "PermissionError: receipt unavailable"}


def test_decode_restart_bundle_carries_the_latent_it_re_emits(tmp_path: Path) -> None:
    """A decode restart writes the final latent a further decode restart needs."""
    plan = plan_krea2_output(
        Krea2Request("test"),
        "turbo",
        ImageOutputConfig(directory=tmp_path, prefix="replay", save_all_sidecars=True),
        station="decode",
    )

    assert plan.artifacts.initial_noise == ()
    assert plan.artifacts.text_conditioning == ()
    sink = Krea2OutputSink(plan, image_writer=lambda path, _pixels: path.write_bytes(b"image"))
    sink.write(_generation(1))

    latents = plan.artifacts.single_path("final_latents")
    assert latents is not None
    assert latents.is_file()
    assert mx.load(str(latents))["final_latents"].shape[0] == 1


def test_failed_publication_does_not_expose_staged_fingerprints(
    tmp_path: Path, monkeypatch
) -> None:
    plan = plan_krea2_output(
        Krea2Request("test"),
        "turbo",
        ImageOutputConfig(directory=tmp_path, save_final_latents=True),
    )
    sink = Krea2OutputSink(plan, image_writer=lambda path, _pixels: path.write_bytes(b"image"))

    def fail_publication(*_args):
        raise OSError("publication failed")

    monkeypatch.setattr("kinomlx.models.krea2.output.publish_staged", fail_publication)
    with pytest.raises(match="publication failed"):
        sink.write(_generation(1))
    assert sink.fingerprints == {}
    assert not any(path.exists() for path in plan.artifacts.paths())


def test_later_seed_failure_keeps_completed_replacement_and_unfinished_old_target(
    tmp_path: Path,
) -> None:
    request = Krea2Request("test", images_per_prompt=2)
    plan = plan_krea2_output(
        request,
        "turbo",
        ImageOutputConfig(path=tmp_path / "chosen.png", force=True),
    )
    for path in plan.artifacts.images:
        path.write_bytes(b"old")
    calls = 0

    def writer(path: Path, _image: mx.array) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(b"new")
        if calls == 2:
            raise RuntimeError("encoder stopped")

    with pytest.raises(match="encoder stopped"):
        Krea2OutputSink(plan, image_writer=writer).write(_generation(2))
    assert plan.artifacts.images[0].read_bytes() == b"new"
    assert plan.artifacts.images[1].read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))
    assert not list(tmp_path.glob(".krea2-output-*"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("create_empty", [False, True])
def test_empty_later_image_preserves_finished_seed(tmp_path: Path, existing, create_empty) -> None:
    plan = plan_krea2_output(
        Krea2Request("test", images_per_prompt=2),
        "turbo",
        ImageOutputConfig(path=tmp_path / "chosen.png", force=existing, save_final_latents=True),
    )
    if existing:
        for target in plan.artifacts.paths():
            target.write_bytes(b"old")
    calls = 0

    def writer(path: Path, _image: mx.array) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_bytes(b"first image")
        elif create_empty:
            path.write_bytes(b"")

    with pytest.raises(match="nonempty regular file"):
        Krea2OutputSink(plan, image_writer=writer).write(_generation(2))
    first = plan.seed_artifacts(0)
    expected_files = set(plan.artifacts.paths()) if existing else set(first.paths())
    assert set(tmp_path.iterdir()) == expected_files
    assert first.images[0].read_bytes() == b"first image"
    assert mx.load(str(first.final_latents[0]))["final_latents"].shape[0] == 1
    if existing:
        assert all(path.read_bytes() == b"old" for path in plan.seed_artifacts(1).paths())


def test_bundle_collision_is_refused_before_any_writer_runs(tmp_path: Path) -> None:
    request = Krea2Request("test")
    plan = plan_krea2_output(
        request,
        "turbo",
        ImageOutputConfig(path=tmp_path / "chosen.png"),
    )
    plan.artifacts.images[0].write_bytes(b"old")
    called = False

    def writer(_path: Path, _image: mx.array) -> None:
        nonlocal called
        called = True

    with pytest.raises(match="exists"):
        Krea2OutputSink(plan, image_writer=writer).write(_generation())
    assert called is False
    assert plan.artifacts.images[0].read_bytes() == b"old"


@pytest.mark.parametrize("count", [1, 3])
def test_per_image_digest_keys_are_the_shared_restart_key(tmp_path: Path, count: int) -> None:
    plan = plan_krea2_output(
        Krea2Request("test", images_per_prompt=count),
        "turbo",
        ImageOutputConfig(
            directory=tmp_path,
            prefix="bundle",
            save_initial_noise=True,
            save_final_latents=True,
        ),
    )
    sink = Krea2OutputSink(plan)
    with plan.reserve() as reservation:
        for index in range(count):
            sink.write_image(index, _one_image(index), reservation=reservation)

    # A restart reads these digests back through fingerprint_key, so the writer
    # must spell them with that one function, not a second inline convention.
    assert set(sink.fingerprints) == {
        fingerprint_key(name, index=index, count=count)
        for name in ("initial_noise", "final_latents")
        for index in range(count)
    }


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_completed_seed_and_sidecars_survive_next_seed_failure(tmp_path, failure):
    plan = plan_krea2_output(
        Krea2Request("test", seeds=(9, 2)),
        "turbo",
        ImageOutputConfig(path=tmp_path / "result.png", save_all_sidecars=True),
    )
    sink = Krea2OutputSink(plan)
    digests = {}

    def run():
        with plan.reserve() as reservation:
            sink.write_image(0, _generation(include_text=True), reservation=reservation)
            artifacts = plan.seed_artifacts(0)
            assert load_image_rgb(artifacts.images[0]).shape == (2, 3, 3)
            for path in (
                *artifacts.initial_noise,
                *artifacts.final_latents,
                *artifacts.text_conditioning,
            ):
                assert mx.load(str(path))
            digests.update({path: file_sha256(path) for path in artifacts.paths()})
            raise failure("stop remaining seeds")

    with pytest.raises(failure):
        run()
    assert all(file_sha256(path) == digest for path, digest in digests.items())
    assert not plan.artifacts.images[1].exists()
    assert all(not reservation_path(path).exists() for path in plan.targets())


def test_completed_sidecar_survives_later_sidecar_write_failure(tmp_path, monkeypatch):
    plan = plan_krea2_output(
        Krea2Request("test"),
        "turbo",
        ImageOutputConfig(path=tmp_path / "result.png", save_all_sidecars=True),
    )
    sink = Krea2OutputSink(plan)
    original = mx.save_safetensors

    def write(path, tensors, *, metadata):
        if "final_latents" in tensors:
            raise OSError("disk full")
        original(path, tensors, metadata=metadata)

    monkeypatch.setattr(mx, "save_safetensors", write)
    with plan.reserve() as reservation, pytest.raises(BundleOutputError, match="disk full"):
        sink.write_image(0, _generation(include_text=True), reservation=reservation)
    noise = plan.artifacts.initial_noise[0]
    assert "initial_noise" in mx.load(str(noise))
    assert str(noise) in sink.completed_outputs()["published_artifacts"]
