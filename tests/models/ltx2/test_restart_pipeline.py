"""Shared public/CLI recipe gates for distilled station restart."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import kinomlx.models.ltx2.pipelines.distilled as distilled
import kinomlx.models.ltx2.pipelines.restart as restart_pipeline
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import save_weights
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.artifacts import FINAL_LATENTS, STAGE_1_LATENTS
from kinomlx.models.ltx2.pipelines.restart import DistilledRestart
from kinomlx.models.ltx2.text_conditioning import NativeTextConditioner
from kinomlx.models.ltx2.types import (
    AudioLatentShape,
    DistilledRequest,
    video_latent_shape_from_pixel,
)
from kinomlx.reporting import RecordingReporter
from kinomlx.types import VideoPixelShape
from tests.models.ltx2.test_distilled_pipeline import (
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
    video = mx.arange(int(np.prod(video_shape)), dtype=mx.float32).reshape(video_shape) / 100.0
    audio_latent = None
    arrays = {"video_latent": video}
    if audio:
        audio_shape = AudioLatentShape.from_video(geometry, fps=fps).to_tuple()
        audio_latent = (
            mx.arange(int(np.prod(audio_shape)), dtype=mx.float32).reshape(audio_shape) / 50.0
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


def _consume(output) -> tuple[np.ndarray, np.ndarray | None]:
    frames = np.stack([np.asarray(frame) for frame in output.frames])
    waveform = None if output.audio_waveform is None else np.asarray(output.audio_waveform)
    return frames, waveform


def test_public_restart_constructors_keep_the_call_site_small(tmp_path: Path) -> None:
    final = DistilledRestart.decode(tmp_path / "final.safetensors")
    stage_1 = DistilledRestart.decode(
        tmp_path / "stage1.safetensors",
        latent_stage="stage-1",
    )
    stage_2 = DistilledRestart.stage_2(
        tmp_path / "stage1.safetensors",
        text_conditioning=tmp_path / "text.safetensors",
    )

    assert (final.phase, final.latent_stage) == ("decode", "final")
    assert (stage_1.phase, stage_1.latent_stage) == ("decode", "stage-1")
    assert (stage_2.phase, stage_2.latent_stage) == ("stage-2", "stage-1")


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

    output = restart_pipeline.restart_distilled(
        DistilledRequest(prompt="", width=64, height=64, frames=9),
        _Resources(),
        restart=DistilledRestart.stage_2(
            latent_path,
            text_conditioning=text_path,
        ),
        components=_RecordingComponents(),
    )

    text_conditioner = captured["text_conditioner"]
    assert output is expected
    assert isinstance(text_conditioner, NativeTextConditioner)
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
        stage="final",
        geometry=geometry,
        fps=24.0,
        generate_audio=False,
        reference_aligned_audio=False,
        reporter=RecordingReporter(),
        source_model_generation="2.5",
    )

    np.testing.assert_array_equal(np.asarray(loaded.video), np.asarray(expected_video))


def test_restart_wrong_latent_shape_fails_before_any_component_opens(
    tmp_path: Path,
) -> None:
    latent_path = tmp_path / "wrong-shape.safetensors"
    text_path = tmp_path / "text.safetensors"
    text_path.touch()
    save_weights(latent_path, {"video_latent": mx.zeros((1, 128, 2, 9, 9))})
    components = _RecordingComponents()

    with pytest.raises(ValueError, match=r"LTX-2\.5 stage-1 video latent shape.*does not match"):
        restart_pipeline.restart_distilled(
            DistilledRequest(prompt="", width=64, height=64, frames=9),
            _Resources(generation="2.5"),
            restart=DistilledRestart.stage_2(
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
        restart_pipeline.restart_distilled(
            DistilledRequest(prompt="", width=64, height=64, frames=9),
            _Resources(generation=generation),
            restart=DistilledRestart.decode(
                latent_path,
                source_model_generation=generation,
            ),
            components=components,
        )

    assert components.events == []


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_stage_2_restart_runs_only_the_shared_downstream_stations(
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

    output = restart_pipeline.restart_distilled(
        DistilledRequest(prompt="ignored", width=64, height=64, frames=9, seed=42),
        _Resources(generation=generation),
        restart=DistilledRestart.stage_2(
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
        restart_pipeline.restart_distilled(
            DistilledRequest(
                prompt="ignored",
                width=64,
                height=64,
                frames=9,
                noise_backend="torch-mps",
            ),
            _Resources(),
            restart=DistilledRestart.stage_2(
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
    request = DistilledRequest(prompt="test", width=64, height=64, frames=9, seed=42)
    uninterrupted_artifacts = {}

    class _UninterruptedArtifacts:
        def save(self, artifact) -> None:
            if artifact.name in {STAGE_1_LATENTS, FINAL_LATENTS}:
                uninterrupted_artifacts[artifact.name] = dict(artifact.tensors)

    uninterrupted = distilled.generate_distilled(
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

    restarted = restart_pipeline.restart_distilled(
        request,
        _Resources(generation=generation),
        restart=DistilledRestart.stage_2(
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


def test_torch_mps_stage_2_restart_resumes_recorded_noise_position_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_operations(monkeypatch)
    monkeypatch.setattr(restart_pipeline, "release_stage_temporaries", lambda: None)
    request = DistilledRequest(
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

    uninterrupted = distilled.generate_distilled(
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

    restarted = restart_pipeline.restart_distilled(
        request,
        _Resources(generation="2.3"),
        restart=DistilledRestart.stage_2(
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


def test_stage_1_direct_decode_reports_half_resolution(
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
    output = restart_pipeline.restart_distilled(
        DistilledRequest(prompt="", width=64, height=64, frames=9),
        _Resources(),
        restart=DistilledRestart.decode(latent_path, latent_stage="stage-1"),
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
    request = DistilledRequest(
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

    restarted = restart_pipeline.restart_distilled(
        request,
        _Resources(),
        restart=DistilledRestart.decode(latent_path),
        components=_RecordingComponents(),
    )
    restart_frames, restart_audio = _consume(restarted)
    np.testing.assert_array_equal(restart_frames, normal_frames)
    np.testing.assert_array_equal(restart_audio, normal_audio)

    rerolled = restart_pipeline.restart_distilled(
        DistilledRequest(
            prompt="",
            width=64,
            height=64,
            frames=9,
            seed=43,
            generate_audio=True,
        ),
        _Resources(),
        restart=DistilledRestart.decode(latent_path),
        components=_RecordingComponents(),
    )
    rerolled_frames, rerolled_audio = _consume(rerolled)

    assert not np.array_equal(rerolled_frames, normal_frames)
    np.testing.assert_array_equal(rerolled_audio, normal_audio)
    assert file_sha256(latent_path) == source_hash
