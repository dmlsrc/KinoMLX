"""Shared public/CLI recipe gates for distilled stage restart."""

import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.conditioning.preparation as condition_preparation
import kinomlx.models.ltx2.pipelines.distilled as distilled
import kinomlx.models.ltx2.pipelines.restart as restart_pipeline
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import save_weights
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.artifacts import (
    FINAL_LATENTS,
    STAGE_1_LATENTS,
    STAGE_2_CONDITIONING,
    TEXT_CONDITIONING,
)
from kinomlx.models.ltx2.restart import LTX2Restart
from kinomlx.models.ltx2.text_conditioning import LTX2TextConditioner
from kinomlx.models.ltx2.types import (
    AudioLatentShape,
    LTX2ImageConditioningConfig,
    LTX2Request,
    video_latent_shape_from_pixel,
)
from kinomlx.types import VideoPixelShape
from tests._reporting import RecordingReporter
from tests.models.ltx2.test_distilled_pipeline import (
    _X0,
    _patch_operations,
    _RecordingComponents,
    _RecordingTextConditioner,
    _Resources,
)


class _RestartTextConditioner(_RecordingTextConditioner):
    def __call__(self, request, resources, **kwargs):
        text = super().__call__(request, resources, **kwargs)
        return replace(
            text,
            replay_receipt={"policy": "observe", "identity_match": False},
        )


def _stage_latents(
    path: Path,
    geometry: VideoPixelShape,
    *,
    fps: float = 24.0,
    audio: bool = False,
    metadata: dict[str, str] | None = None,
) -> tuple[mx.array, mx.array | None]:
    video_shape = video_latent_shape_from_pixel(geometry).to_tuple()
    video = mx.arange(math.prod(video_shape), dtype=mx.float32).reshape(video_shape) / 100.0
    audio_latent = None
    arrays = {"video_latent": video}
    if audio:
        audio_shape = AudioLatentShape.from_video(geometry, fps=fps).to_tuple()
        audio_latent = (
            mx.arange(math.prod(audio_shape), dtype=mx.float32).reshape(audio_shape) / 50.0
        )
        arrays["audio_latent"] = audio_latent
    save_weights(path, arrays, metadata or {})
    return video, audio_latent


def _seeded_decode(
    _latent,
    _decoder_provider,
    *,
    spec,
    frame_count,
    decoder_seed,
    **_kwargs,
):
    value = float((int(decoder_seed) * 17) % 251) / 255.0

    def frames():
        for _index in range(frame_count):
            yield mx.full((spec.height, spec.width, 3), value, dtype=mx.float16)

    return VideoFrameStream(frames, spec=spec, frame_count=frame_count)


def _consume(output) -> tuple[mx.array, mx.array | None]:
    frames = mx.stack(list(output.frames))
    return frames, output.audio_waveform


class _RecordingArtifacts:
    """Accept every offered artifact so a stage's own output can be read back."""

    def __init__(self) -> None:
        self.tensors: dict[str, dict[str, mx.array]] = {}

    def save(self, artifact) -> None:
        self.tensors[artifact.name] = dict(artifact.tensors)


def test_decode_restart_re_emits_the_latent_product_it_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "final.safetensors"
    stage_1_path = tmp_path / "stage1.safetensors"
    final_video, _audio = _stage_latents(
        final_path, VideoPixelShape(batch=1, frames=9, height=64, width=64)
    )
    stage_1_video, _stage_1_audio = _stage_latents(
        stage_1_path, VideoPixelShape(batch=1, frames=9, height=32, width=32)
    )
    monkeypatch.setattr(distilled, "decode_ltx23_sdr_frames", _seeded_decode)
    monkeypatch.setattr(distilled, "release_stage_temporaries", lambda: None)
    request = LTX2Request(prompt="", width=64, height=64, frames=9, seed=42)

    final_saved = _RecordingArtifacts()
    restart_pipeline.restart_ltx2(
        request,
        _Resources(),
        restart=LTX2Restart.decode(final_path),
        components=_RecordingComponents(),
        artifact_sink=final_saved,
    ).close()

    stage_1_saved = _RecordingArtifacts()
    restart_pipeline.restart_ltx2(
        request,
        _Resources(),
        restart=LTX2Restart.decode(stage_1_path, latent_stage="stage-1"),
        components=_RecordingComponents(),
        artifact_sink=stage_1_saved,
    ).close()

    assert set(final_saved.tensors) == {FINAL_LATENTS}
    assert mx.array_equal(final_saved.tensors[FINAL_LATENTS]["video_latent"], final_video).item()
    assert set(stage_1_saved.tensors) == {STAGE_1_LATENTS}
    assert mx.array_equal(
        stage_1_saved.tensors[STAGE_1_LATENTS]["video_latent"], stage_1_video
    ).item()


def test_stage_2_restart_re_emits_the_text_and_stage_1_latent_it_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / "stage1.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    stage_1_video, _audio = _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    saved = _RecordingArtifacts()

    output = restart_pipeline.restart_ltx2(
        LTX2Request(prompt="ignored", width=64, height=64, frames=9, seed=42),
        _Resources(),
        restart=LTX2Restart.stage_2(latent_path, text_conditioning=text_path),
        components=_RecordingComponents(),
        text_conditioner=_RestartTextConditioner(),
        artifact_sink=saved,
    )
    output.close()

    assert {STAGE_1_LATENTS, FINAL_LATENTS, TEXT_CONDITIONING} <= set(saved.tensors)
    assert mx.array_equal(saved.tensors[STAGE_1_LATENTS]["video_latent"], stage_1_video).item()


def test_public_restart_constructors_keep_the_call_site_small(tmp_path: Path) -> None:
    final = LTX2Restart.decode(tmp_path / "final.safetensors")
    stage_1 = LTX2Restart.decode(
        tmp_path / "stage1.safetensors",
        latent_stage="stage-1",
    )
    stage_2 = LTX2Restart.stage_2(
        tmp_path / "stage1.safetensors",
        text_conditioning=tmp_path / "text.safetensors",
    )

    assert (final.stage, final.latent_stage) == ("decode", "stage-2")
    assert (stage_1.stage, stage_1.latent_stage) == ("decode", "stage-1")
    assert (stage_2.stage, stage_2.selected_latent_stage) == ("stage-2", "stage-1")


def test_stage_2_restart_selects_observational_identity_policy_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / "stage1.safetensors"
    text_path = tmp_path / "text.safetensors"
    latent_path.touch()
    text_path.touch()
    captured = {}
    expected = object()

    def fake_stage_2(*_args, text_conditioner, **_kwargs):
        captured["text_conditioner"] = text_conditioner
        return expected

    monkeypatch.setattr(restart_pipeline, "_restart_stage_2", fake_stage_2)

    output = restart_pipeline.restart_ltx2(
        LTX2Request(prompt="", width=64, height=64, frames=9),
        _Resources(),
        restart=LTX2Restart.stage_2(
            latent_path,
            text_conditioning=text_path,
        ),
        components=_RecordingComponents(),
    )

    text_conditioner = captured["text_conditioner"]
    assert output is expected
    assert isinstance(text_conditioner, LTX2TextConditioner)
    assert text_conditioner.replay_identity_policy == "observe"


def test_restart_metadata_labels_are_advisory_when_tensor_structure_fits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "renamed-community-latents.safetensors"
    geometry = VideoPixelShape(batch=1, frames=9, height=64, width=64)
    expected_video, _audio = _stage_latents(
        path,
        geometry,
        metadata={
            "pipeline": "community_pipeline",
            "stage": "99",
            "final": "false",
        },
    )

    loaded = restart_pipeline.load_stage_latents(
        path,
        stage="stage-2",
        geometry=geometry,
        fps=24.0,
        generate_audio=False,
        reference_aligned_audio=False,
        reporter=RecordingReporter(),
        source_model_generation="2.5",
    )

    assert mx.array_equal(loaded.video, expected_video).item()


def test_restart_wrong_latent_shape_fails_before_any_component_opens(
    tmp_path: Path,
) -> None:
    latent_path = tmp_path / "wrong-shape.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    save_weights(latent_path, {"video_latent": mx.zeros((1, 128, 2, 9, 9))})
    components = _RecordingComponents()

    with pytest.raises(ValueError, match=r"LTX-2\.5 stage-1 video latent shape.*does not match"):
        restart_pipeline.restart_ltx2(
            LTX2Request(prompt="", width=64, height=64, frames=9),
            _Resources(generation="2.5"),
            restart=LTX2Restart.stage_2(
                latent_path,
                text_conditioning=text_path,
                source_model_generation="2.5",
            ),
            components=components,
            text_conditioner=lambda *_args, **_kwargs: pytest.fail(
                "shape failure must precede text loading"
            ),
        )

    assert components.events == []


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [("integer", "floating dtype"), ("nonfinite", "finite values")],
)
def test_restart_latent_value_checks_are_shared_by_both_generations(
    generation: str,
    invalid_kind: str,
    message: str,
    tmp_path: Path,
) -> None:
    latent_path = tmp_path / f"ltx-{generation}-{invalid_kind}.safetensors"
    shape = (1, 128, 2, 2, 2)
    latent = (
        mx.zeros(shape, dtype=mx.int32)
        if invalid_kind == "integer"
        else mx.full(shape, float("nan"), dtype=mx.float32)
    )
    save_weights(latent_path, {"video_latent": latent})
    components = _RecordingComponents()

    with pytest.raises(ValueError, match=message):
        restart_pipeline.restart_ltx2(
            LTX2Request(prompt="", width=64, height=64, frames=9),
            _Resources(generation=generation),
            restart=LTX2Restart.decode(
                latent_path,
                source_model_generation=generation,
            ),
            components=components,
        )

    assert components.events == []


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_stage_2_restart_runs_only_the_shared_downstream_stages(
    generation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / f"ltx-{generation}-stage1.safetensors"
    text_path = tmp_path / f"ltx-{generation}-text.safetensors"
    text_path.touch()
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    components = _RecordingComponents()
    text_conditioner = _RestartTextConditioner()
    captured = {}

    class _Artifacts:
        def save(self, artifact) -> None:
            if artifact.name == FINAL_LATENTS:
                captured.update(dict(artifact.tensors))

    output = restart_pipeline.restart_ltx2(
        LTX2Request(prompt="ignored", width=64, height=64, frames=9, seed=42),
        _Resources(generation=generation),
        restart=LTX2Restart.stage_2(
            latent_path,
            text_conditioning=text_path,
            source_model_generation=generation,
        ),
        components=components,
        text_conditioner=text_conditioner,
        artifact_sink=_Artifacts(),
    )

    assert len(text_conditioner.calls) == 1
    assert text_conditioner.calls[0]["request"].text_conditioning == text_path
    assert len(components.transformers) == 1
    assert components.transformers[0].video_token_counts == [8, 8, 8]
    assert "video_latent" in captured
    assert tuple(captured["video_latent"].shape) == (1, 128, 2, 2, 2)
    assert output.metadata["execution_mode"] == "restart-stage-2"
    assert output.metadata["source_model_generation"] == generation
    assert output.metadata["text_conditioning_replay"] == {
        "policy": "observe",
        "identity_match": False,
    }
    output.close()
    assert components.active == set()


def test_stage_2_restart_re_prepares_every_saved_image_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / "two-condition-stage1.safetensors"
    text_path = tmp_path / "two-condition-text.safetensors"
    text_path.touch()
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)

    def encode_image(_path, _encoder, *, width, height, compute_dtype, **_kwargs):
        return mx.ones((1, 128, 1, height // 32, width // 32), dtype=compute_dtype)

    monkeypatch.setattr(condition_preparation, "encode_image", encode_image)
    saved: dict[str, dict[str, str]] = {}

    class _Artifacts:
        def save(self, artifact) -> None:
            if artifact.name == STAGE_2_CONDITIONING:
                saved[artifact.name] = dict(artifact.metadata)

    output = restart_pipeline.restart_ltx2(
        LTX2Request(
            prompt="ignored",
            width=64,
            height=64,
            frames=9,
            seed=42,
            images=(
                LTX2ImageConditioningConfig(tmp_path / "first.png"),
                LTX2ImageConditioningConfig(tmp_path / "last.png", frame_index=8),
            ),
        ),
        _Resources(),
        restart=LTX2Restart.stage_2(latent_path, text_conditioning=text_path),
        components=_RecordingComponents(),
        text_conditioner=_RestartTextConditioner(),
        artifact_sink=_Artifacts(),
    )
    output.close()

    metadata = saved[STAGE_2_CONDITIONING]
    assert metadata["condition_count"] == "2"
    assert metadata["condition_0_family"] == "image"
    assert metadata["condition_0_frame_index"] == "0"
    assert metadata["condition_1_family"] == "keyframe"
    assert metadata["condition_1_frame_index"] == "8"


def test_stage_2_restart_re_prepares_conditions_at_their_stage_2_strengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restarted stage applies its own strengths, not the common intent."""
    latent_path = tmp_path / "staged-stage1.safetensors"
    text_path = tmp_path / "staged-text.safetensors"
    text_path.touch()
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)

    def encode_image(_path, _encoder, *, width, height, compute_dtype, **_kwargs):
        return mx.ones((1, 128, 1, height // 32, width // 32), dtype=compute_dtype)

    monkeypatch.setattr(condition_preparation, "encode_image", encode_image)
    saved: dict[str, dict[str, str]] = {}

    class _Artifacts:
        def save(self, artifact) -> None:
            if artifact.name == STAGE_2_CONDITIONING:
                saved[artifact.name] = dict(artifact.metadata)

    output = restart_pipeline.restart_ltx2(
        LTX2Request(
            prompt="ignored",
            width=64,
            height=64,
            frames=9,
            seed=42,
            images=(
                LTX2ImageConditioningConfig(
                    tmp_path / "first.png", strength=1.0, stage2_strength=0.6
                ),
                LTX2ImageConditioningConfig(tmp_path / "last.png", frame_index=8, strength=0.9),
            ),
        ),
        _Resources(),
        restart=LTX2Restart.stage_2(latent_path, text_conditioning=text_path),
        components=_RecordingComponents(),
        text_conditioner=_RestartTextConditioner(),
        artifact_sink=_Artifacts(),
    )
    output.close()

    metadata = saved[STAGE_2_CONDITIONING]
    assert metadata["condition_0_strength"] == "0.6"
    assert metadata["condition_1_strength"] == "0.9"


def test_stage_2_restart_re_prepares_conditions_in_the_requested_frame_zero_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restarted stage applies frame 0 the way the request asks."""
    latent_path = tmp_path / "staged-stage1.safetensors"
    text_path = tmp_path / "staged-text.safetensors"
    text_path.touch()
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)

    def encode_image(_path, _encoder, *, width, height, compute_dtype, **_kwargs):
        return mx.ones((1, 128, 1, height // 32, width // 32), dtype=compute_dtype)

    monkeypatch.setattr(condition_preparation, "encode_image", encode_image)
    saved: dict[str, dict[str, str]] = {}

    class _Artifacts:
        def save(self, artifact) -> None:
            if artifact.name == STAGE_2_CONDITIONING:
                saved[artifact.name] = dict(artifact.metadata)

    output = restart_pipeline.restart_ltx2(
        LTX2Request(
            prompt="ignored",
            width=64,
            height=64,
            frames=9,
            seed=42,
            first_frame_mode="guide",
            images=(LTX2ImageConditioningConfig(tmp_path / "first.png"),),
        ),
        _Resources(),
        restart=LTX2Restart.stage_2(latent_path, text_conditioning=text_path),
        components=_RecordingComponents(),
        text_conditioner=_RestartTextConditioner(),
        artifact_sink=_Artifacts(),
    )
    output.close()

    assert saved[STAGE_2_CONDITIONING]["first_frame_mode"] == "guide"


def test_legacy_stage_1_artifact_requires_legacy_mlx_noise_for_stage_2(
    tmp_path: Path,
) -> None:
    latent_path = tmp_path / "legacy-stage1.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
        metadata={"pipeline": "distilled_two_stage", "stage": "1", "final": "false"},
    )

    with pytest.raises(ValueError, match="legacy stage-1 latents have no Torch-MPS"):
        restart_pipeline.restart_ltx2(
            LTX2Request(
                prompt="ignored",
                width=64,
                height=64,
                frames=9,
                noise_backend="torch-mps",
            ),
            _Resources(),
            restart=LTX2Restart.stage_2(
                latent_path,
                text_conditioning=text_path,
            ),
            components=_RecordingComponents(),
            text_conditioner=_RestartTextConditioner(),
        )


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_stage_2_restart_reproduces_the_uninterrupted_final_latent_exactly(
    generation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    request = LTX2Request(prompt="test", width=64, height=64, frames=9, seed=42)
    uninterrupted_artifacts = {}

    class _UninterruptedArtifacts:
        def save(self, artifact) -> None:
            if artifact.name in {STAGE_1_LATENTS, FINAL_LATENTS}:
                uninterrupted_artifacts[artifact.name] = dict(artifact.tensors)

    uninterrupted = distilled.generate_ltx2(
        request,
        _Resources(generation=generation),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
        artifact_sink=_UninterruptedArtifacts(),
    )
    uninterrupted.close()

    latent_path = tmp_path / "stage1.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    save_weights(latent_path, uninterrupted_artifacts[STAGE_1_LATENTS])
    restarted_artifacts = {}

    class _RestartedArtifacts:
        def save(self, artifact) -> None:
            if artifact.name == FINAL_LATENTS:
                restarted_artifacts.update(dict(artifact.tensors))

    restarted = restart_pipeline.restart_ltx2(
        request,
        _Resources(generation=generation),
        restart=LTX2Restart.stage_2(
            latent_path,
            text_conditioning=text_path,
            source_model_generation=generation,
        ),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
        artifact_sink=_RestartedArtifacts(),
    )
    restarted.close()

    assert mx.array_equal(
        restarted_artifacts["video_latent"],
        uninterrupted_artifacts[FINAL_LATENTS]["video_latent"],
    ).item()


def test_stage_2_restart_reseeds_generated_keyframes_from_the_saved_stage_1_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2 refines the stage-1 slots, so its restart needs them to reproduce the run."""
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    record = _X0.__call__

    def halving(self, video, audio=None):
        # Predictions that follow the input give the slots content worth seeding.
        record(self, video, audio)
        return video.latent * 0.5, None if audio is None else audio.latent * 0.5

    monkeypatch.setattr(_X0, "__call__", halving)
    request = LTX2Request(
        prompt="test", width=64, height=64, frames=17, seed=42, generated_keyframes=2
    )
    uninterrupted = _RecordingArtifacts()
    output = distilled.generate_ltx2(
        request,
        _Resources(generation="2.5"),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
        artifact_sink=uninterrupted,
    )
    indices = output.metadata["generated_keyframe_indices"]
    output.close()
    assert len(indices) == 2
    stage_1 = uninterrupted.tensors[STAGE_1_LATENTS]
    assert mx.any(stage_1["generated_keyframes"]).item()
    text_path = tmp_path / "text.safetensors"
    text_path.touch()

    def restarted_final(tensors: dict[str, mx.array]) -> dict[str, mx.array]:
        latent_path = tmp_path / "stage1.safetensors"
        save_weights(latent_path, tensors)
        artifacts = _RecordingArtifacts()
        restarted = restart_pipeline.restart_ltx2(
            request,
            _Resources(generation="2.5"),
            restart=LTX2Restart.stage_2(
                latent_path,
                text_conditioning=text_path,
                source_model_generation="2.5",
            ),
            components=_RecordingComponents(),
            text_conditioner=_RecordingTextConditioner(),
            artifact_sink=artifacts,
        )
        assert restarted.metadata["generated_keyframe_indices"] == indices
        restarted.close()
        return artifacts.tensors[FINAL_LATENTS]

    final = uninterrupted.tensors[FINAL_LATENTS]
    restarted = restarted_final(stage_1)
    for name in ("video_latent", "generated_keyframes"):
        assert mx.array_equal(restarted[name], final[name]).item()
    # The fake predicts each token from itself alone, so the seed shows in stage 2's slots.
    zeroed = {**stage_1, "generated_keyframes": mx.zeros_like(stage_1["generated_keyframes"])}
    reseeded = restarted_final(zeroed)["generated_keyframes"]
    assert not mx.array_equal(reseeded, final["generated_keyframes"]).item()
    without = {name: value for name, value in stage_1.items() if name != "generated_keyframes"}
    with pytest.raises(ValueError, match="has no generated_keyframes tensor"):
        restarted_final(without)

    final_path = tmp_path / "final.safetensors"
    save_weights(final_path, final)
    decoded = restart_pipeline.restart_ltx2(
        request,
        _Resources(generation="2.5"),
        restart=LTX2Restart.decode(final_path, source_model_generation="2.5"),
        components=_RecordingComponents(),
    )
    assert decoded.metadata["generated_keyframe_indices"] == indices
    decoded.close()


def test_torch_mps_stage_2_restart_resumes_recorded_noise_position_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    request = LTX2Request(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        seed=42,
        noise_backend="torch-mps",
    )
    uninterrupted_artifacts = {}

    class _UninterruptedArtifacts:
        def save(self, artifact) -> None:
            if artifact.name in {STAGE_1_LATENTS, FINAL_LATENTS}:
                uninterrupted_artifacts[artifact.name] = artifact

    uninterrupted = distilled.generate_ltx2(
        request,
        _Resources(generation="2.3"),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
        artifact_sink=_UninterruptedArtifacts(),
    )
    assert uninterrupted.metadata["noise_backend"] == "torch-mps"
    assert uninterrupted.metadata["noise_compatibility_profile"] == "pytorch-2.13.0-mps"
    assert uninterrupted.metadata["initial_noise_state"]["backend"] == "torch-mps"
    uninterrupted.close()

    stage_1 = uninterrupted_artifacts[STAGE_1_LATENTS]
    latent_path = tmp_path / "stage1.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    save_weights(latent_path, dict(stage_1.tensors), dict(stage_1.metadata))
    restarted_artifacts = {}

    class _RestartedArtifacts:
        def save(self, artifact) -> None:
            if artifact.name == FINAL_LATENTS:
                restarted_artifacts.update(dict(artifact.tensors))

    restarted = restart_pipeline.restart_ltx2(
        request,
        _Resources(generation="2.3"),
        restart=LTX2Restart.stage_2(
            latent_path,
            text_conditioning=text_path,
            source_model_generation="2.3",
        ),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
        artifact_sink=_RestartedArtifacts(),
    )
    restarted.close()

    final = uninterrupted_artifacts[FINAL_LATENTS]
    assert mx.array_equal(
        restarted_artifacts["video_latent"],
        dict(final.tensors)["video_latent"],
    ).item()


def test_stage_1_decode_reports_half_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / "stage1.safetensors"
    _stage_latents(
        latent_path,
        VideoPixelShape(batch=1, frames=9, height=32, width=32),
    )
    monkeypatch.setattr(distilled, "decode_ltx23_sdr_frames", _seeded_decode)
    monkeypatch.setattr(distilled, "release_stage_temporaries", lambda: None)
    output = restart_pipeline.restart_ltx2(
        LTX2Request(prompt="", width=64, height=64, frames=9),
        _Resources(),
        restart=LTX2Restart.decode(latent_path, latent_stage="stage-1"),
        components=_RecordingComponents(),
    )

    assert (output.signal.width, output.signal.height, output.frame_count) == (32, 32, 9)
    assert output.metadata["video_shape"] == (1, 3, 9, 32, 32)
    output.close()


def test_decode_restart_is_bit_exact_before_encoding_and_seed_can_reroll_diffvae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_path = tmp_path / "final.safetensors"
    geometry = VideoPixelShape(batch=1, frames=9, height=64, width=64)
    video, audio = _stage_latents(latent_path, geometry, audio=True)
    source_hash = file_sha256(latent_path)
    monkeypatch.setattr(distilled, "decode_ltx23_sdr_frames", _seeded_decode)
    monkeypatch.setattr(distilled, "release_stage_temporaries", lambda: None)
    components = _RecordingComponents()
    request = LTX2Request(
        prompt="",
        width=64,
        height=64,
        frames=9,
        seed=42,
        generate_audio=True,
    )

    normal = distilled.decode_stage_latents(
        request,
        _Resources(),
        distilled.StageLatents(video=video, audio=audio),
        geometry=geometry,
        components=components,
        reporter=RecordingReporter(),
    )
    normal_frames, normal_audio = _consume(normal)

    restarted = restart_pipeline.restart_ltx2(
        request,
        _Resources(),
        restart=LTX2Restart.decode(latent_path),
        components=_RecordingComponents(),
    )
    restart_frames, restart_audio = _consume(restarted)
    assert mx.array_equal(restart_frames, normal_frames).item()
    assert mx.array_equal(restart_audio, normal_audio).item()

    rerolled = restart_pipeline.restart_ltx2(
        LTX2Request(
            prompt="",
            width=64,
            height=64,
            frames=9,
            seed=43,
            generate_audio=True,
        ),
        _Resources(),
        restart=LTX2Restart.decode(latent_path),
        components=_RecordingComponents(),
    )
    rerolled_frames, rerolled_audio = _consume(rerolled)

    assert not mx.array_equal(rerolled_frames, normal_frames).item()
    assert mx.array_equal(rerolled_audio, normal_audio).item()
    assert file_sha256(latent_path) == source_hash
