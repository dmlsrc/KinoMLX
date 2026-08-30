"""Reproducibility-sidecar contracts at the CLI orchestration boundary."""

from __future__ import annotations

import gc
import json
import time
import tomllib
import weakref
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.artifacts import TensorArtifact
from kinomlx.cli._registry import config_registry
from kinomlx.debug import (
    RunRecord,
    SidecarArtifactSink,
    SidecarError,
    SidecarPaths,
    initialize_execution_log,
    sidecar_selected,
    write_effective_config,
)
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import load_weights_with_metadata
from kinomlx.models.ltx2.artifacts import (
    FINAL_LATENTS,
    STAGE_1_CONDITIONING,
    STAGE_1_LATENTS,
    STAGE_2_CONDITIONING,
    TEXT_CONDITIONING,
    distilled_stage_latents_artifact,
    media_conditioning_artifact,
    restart_artifacts,
    sidecar_paths,
    text_conditioning_artifact,
)
from kinomlx.models.ltx2.conditioning import (
    HDRReferenceConditionSource,
    ImageConditionSource,
    VideoConditionByLatentIndex,
    VideoConditionByReferenceLatent,
)
from kinomlx.reporting import TimingReporter
from kinomlx.samplers.noise import NoiseStreamState
from kinomlx.types import VideoPixelShape


def _ltx2_paths(path: Path) -> SidecarPaths:
    paths = SidecarPaths.for_output(path)
    return paths.with_model_artifacts(sidecar_paths(paths.video))


@pytest.mark.parametrize(
    ("selected", "save_all", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_sidecar_selection_has_one_shared_inheritance_predicate(
    selected: bool | None,
    save_all: bool,
    expected: bool,
) -> None:
    assert sidecar_selected(selected, save_all=save_all) is expected


def test_text_conditioning_artifact_preserves_complete_legacy_schema() -> None:
    artifact = text_conditioning_artifact(
        prompt="prompt",
        video_encoding=mx.zeros((1, 1, 1)),
        audio_encoding=mx.zeros((1, 1, 1)),
        attention_mask=mx.ones((1, 1)),
        provenance={
            "model_generation": "ltx-2.3",
            "text_encoder_identity": "gemma-3-12b-it",
            "projection_identity": "connector:test",
        },
    )

    assert dict(artifact.metadata)["schema_version"] == "2"


def test_text_conditioning_artifact_rejects_partial_schema3_provenance() -> None:
    with pytest.raises(ValueError, match="schema-3 provenance is incomplete"):
        text_conditioning_artifact(
            prompt="prompt",
            video_encoding=mx.zeros((1, 1, 1)),
            audio_encoding=mx.zeros((1, 1, 1)),
            attention_mask=mx.ones((1, 1)),
            provenance={
                "model_generation": "ltx-2.5",
                "text_encoder_identity": "gemma4-12b-ltx",
                "projection_identity": "connector:test",
                "tokenizer_source_sha256": "source-json",
            },
        )


def test_latent_artifact_records_resumable_noise_position() -> None:
    state = NoiseStreamState(
        backend="torch-mps",
        compatibility_profile="pytorch-2.13.0-mps",
        seed=42,
        draws=2,
        elements=188_160,
        philox_blocks=47_040,
    )
    artifact = distilled_stage_latents_artifact(
        1,
        video_latent=mx.zeros((1, 128, 2, 2, 2), dtype=mx.bfloat16),
        audio_latent=None,
        final=False,
        noise_state=state,
    )
    metadata = dict(artifact.metadata)

    assert metadata["schema_version"] == "2"
    assert NoiseStreamState.from_artifact_metadata(metadata) == state


def test_sidecar_paths_follow_the_materialized_mp4_stem(tmp_path: Path) -> None:
    paths = _ltx2_paths(tmp_path / "sample.mov")
    artifacts = paths.artifact_paths()
    assert paths.video == tmp_path / "sample.mp4"
    assert artifacts[STAGE_1_LATENTS] == tmp_path / "sample_stage1.safetensors"
    assert artifacts[FINAL_LATENTS] == tmp_path / "sample.safetensors"
    assert artifacts[TEXT_CONDITIONING] == tmp_path / "sample_text.safetensors"
    assert artifacts[STAGE_1_CONDITIONING] == (tmp_path / "sample_stage1_conditioning.safetensors")
    assert artifacts[STAGE_2_CONDITIONING] == (tmp_path / "sample_stage2_conditioning.safetensors")
    assert paths.run_log == tmp_path / "sample_run.json"
    assert paths.execution_log == tmp_path / "sample_console.log"
    assert paths.effective_config == tmp_path / "sample_config.toml"
    assert paths.audio_waveform == tmp_path / "sample.wav"
    assert paths.original_video == tmp_path / "sample_orig.mp4"


def test_effective_config_sidecar_is_round_trippable_toml(tmp_path: Path) -> None:
    path = tmp_path / "sample_config.toml"
    write_effective_config(
        path,
        config_registry()
        .model("ltx2")
        .dump_config(
            {
                "model": "ltx2",
                "generate": {"prompt": "test", "seed": 7},
            }
        ),
    )

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed == {
        "model": "ltx2",
        "generate": {"prompt": "test\n", "seed": 7},
    }
    assert config_registry().model("ltx2").normalize_config(parsed) == {
        "model": "ltx2",
        "generate": {"prompt": "test", "seed": 7},
    }


def test_effective_config_sidecar_refuses_to_replace_an_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample_config.toml"
    path.write_text("source config\n", encoding="utf-8")

    with pytest.raises(SidecarError, match="cannot write effective config"):
        write_effective_config(path, 'model = "ltx2"\n')

    assert path.read_text(encoding="utf-8") == "source config\n"


def test_media_conditioning_artifact_preserves_every_ordered_encoded_source(
    tmp_path: Path,
) -> None:
    image_latent = mx.ones((1, 128, 1, 2, 2), dtype=mx.bfloat16)
    reference_latent = mx.full((1, 128, 2, 2, 2), 2.0, dtype=mx.float32)
    artifact = media_conditioning_artifact(
        2,
        sources=(
            ImageConditionSource(tmp_path / "source.exr", strength=0.75, hdr_authoring="ACESCG"),
            HDRReferenceConditionSource(tmp_path / "reference.mov", strength=0.9),
        ),
        conditions=(
            VideoConditionByLatentIndex(image_latent, strength=0.75, latent_idx=0),
            VideoConditionByReferenceLatent(reference_latent, strength=0.9),
        ),
        geometry=VideoPixelShape(batch=1, frames=9, height=64, width=64),
        fps=24.0,
    )

    tensors = dict(artifact.tensors)
    metadata = dict(artifact.metadata)
    assert artifact.name == STAGE_2_CONDITIONING
    assert set(tensors) == {"condition_0_latent", "condition_1_latent"}
    assert tensors["condition_0_latent"] is image_latent
    assert tensors["condition_1_latent"] is reference_latent
    assert metadata["artifact"] == "ltx2_media_conditioning"
    assert metadata["condition_count"] == "2"
    assert metadata["condition_0_family"] == "image"
    assert metadata["condition_0_frame_index"] == "0"
    assert metadata["condition_0_hdr_authoring"] == "ACESCG"
    assert metadata["condition_1_family"] == "hdr-reference"
    assert metadata["condition_1_strength"] == "0.9"
    assert metadata["stage"] == "2"
    assert metadata["width"] == "64"
    assert artifact.reporting_phase == "save stage 2 media conditioning"


def test_restart_artifacts_keep_only_new_stage_conditioning() -> None:
    requested = frozenset(
        {
            TEXT_CONDITIONING,
            STAGE_1_CONDITIONING,
            STAGE_2_CONDITIONING,
            STAGE_1_LATENTS,
            FINAL_LATENTS,
        }
    )

    assert restart_artifacts(requested, phase="stage-2") == {
        STAGE_2_CONDITIONING,
        FINAL_LATENTS,
    }
    assert restart_artifacts(requested, phase="decode") == frozenset()


def test_artifact_sink_writes_text_and_each_latent_stage(tmp_path: Path) -> None:
    paths = _ltx2_paths(tmp_path / "sample.mp4")
    artifact_paths = paths.artifact_paths()
    sink = SidecarArtifactSink(
        artifact_paths,
        enabled={TEXT_CONDITIONING, STAGE_1_LATENTS, FINAL_LATENTS},
    )
    sink.save(
        text_conditioning_artifact(
            prompt="neutral prompt",
            video_encoding=mx.ones((1, 2, 4), dtype=mx.bfloat16),
            audio_encoding=mx.ones((1, 2, 2), dtype=mx.bfloat16),
            attention_mask=mx.ones((1, 2), dtype=mx.bool_),
            provenance={
                "model_generation": "ltx-2.3",
                "text_encoder_identity": "gemma-3-12b-it",
                "projection_identity": "connector:test",
                "tokenizer_source_sha256": "source-json",
                "tokenizer_model_sha256": "derived-model",
                "tokenizer_metadata_sha256": "derived-metadata",
                "tokenization_policy": "sentencepiece:left-padding-v1",
                "text_artifact_identity": "gemma-3:text",
                "projection_source_identity": "projection:source",
                "connector_source_identity": "connector:source",
            },
        )
    )
    sink.save(
        distilled_stage_latents_artifact(
            1,
            video_latent=mx.zeros((1, 2, 2, 1, 1), dtype=mx.bfloat16),
            audio_latent=mx.zeros((1, 2, 3, 1), dtype=mx.float32),
            final=False,
        )
    )
    sink.save(
        distilled_stage_latents_artifact(
            2,
            video_latent=mx.ones((1, 2, 2, 2, 2), dtype=mx.bfloat16),
            audio_latent=None,
            final=True,
        )
    )

    text, text_metadata = load_weights_with_metadata(artifact_paths[TEXT_CONDITIONING])
    stage_1, stage_1_metadata = load_weights_with_metadata(artifact_paths[STAGE_1_LATENTS])
    final, final_metadata = load_weights_with_metadata(artifact_paths[FINAL_LATENTS])
    assert set(text) == {"video_encoding", "audio_encoding", "attention_mask"}
    assert text["video_encoding"].dtype == mx.bfloat16
    assert text_metadata["prompt"] == "neutral prompt"
    assert text_metadata["schema_version"] == "3"
    assert text_metadata["artifact"] == "ltx2_text_conditioning"
    assert text_metadata["model_generation"] == "ltx-2.3"
    assert text_metadata["text_encoder_identity"] == "gemma-3-12b-it"
    assert text_metadata["projection_identity"] == "connector:test"
    assert set(stage_1) == {"video_latent", "audio_latent"}
    assert stage_1_metadata["stage"] == "1"
    assert set(final) == {"video_latent"}
    assert final_metadata == {
        "schema_version": "1",
        "pipeline": "distilled_two_stage",
        "stage": "2",
        "final": "true",
    }
    assert sink.manifest == {
        TEXT_CONDITIONING: str(artifact_paths[TEXT_CONDITIONING]),
        STAGE_1_LATENTS: str(artifact_paths[STAGE_1_LATENTS]),
        FINAL_LATENTS: str(artifact_paths[FINAL_LATENTS]),
    }
    assert sink.fingerprints == {
        name: file_sha256(artifact_paths[name])
        for name in {TEXT_CONDITIONING, STAGE_1_LATENTS, FINAL_LATENTS}
    }
    assert sink.fingerprint_errors == {}


def test_artifact_sink_persists_an_unfamiliar_model_envelope(tmp_path: Path) -> None:
    path = tmp_path / "other-model.safetensors"
    sink = SidecarArtifactSink(
        {"other_model_state": path},
        enabled={"other_model_state"},
    )
    sink.save(
        TensorArtifact(
            name="other_model_state",
            tensors=(("state", mx.ones((2,), dtype=mx.float16)),),
            metadata=(("family", "other"),),
            reporting_phase="save other state",
        )
    )

    tensors, metadata = load_weights_with_metadata(path)
    assert set(tensors) == {"state"}
    assert metadata == {"family": "other"}
    assert sink.manifest == {"other_model_state": str(path)}
    assert sink.fingerprints == {"other_model_state": file_sha256(path)}


def test_artifact_sink_records_writer_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module

    paths = _ltx2_paths(tmp_path / "sample.mp4")
    artifact_paths = paths.artifact_paths()
    sink = SidecarArtifactSink(
        artifact_paths,
        enabled={STAGE_1_LATENTS},
    )
    monkeypatch.setattr(
        sidecars_module,
        "save_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    sink.save(
        TensorArtifact(
            name=STAGE_1_LATENTS,
            tensors=(("video_latent", object()),),
        )
    )

    assert sink.manifest == {}
    assert sink.errors == [
        {
            "artifact": "stage_1_latents",
            "path": str(artifact_paths[STAGE_1_LATENTS]),
            "error": "OSError: disk full",
        }
    ]


def test_artifact_sink_records_hash_failure_without_vetoing_saved_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module

    paths = _ltx2_paths(tmp_path / "sample.mp4")
    artifact_path = paths.artifact_paths()[STAGE_1_LATENTS]
    sink = SidecarArtifactSink(
        paths.artifact_paths(),
        enabled={STAGE_1_LATENTS},
    )
    monkeypatch.setattr(
        sidecars_module,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("receipt read failed")),
    )

    sink.save(
        distilled_stage_latents_artifact(
            1,
            video_latent=mx.zeros((1,), dtype=mx.float32),
            audio_latent=None,
            final=False,
        )
    )

    assert artifact_path.is_file()
    assert sink.manifest == {STAGE_1_LATENTS: str(artifact_path)}
    assert sink.fingerprints == {}
    assert sink.fingerprint_errors == {STAGE_1_LATENTS: "OSError: receipt read failed"}
    assert sink.errors == []


def test_artifact_sink_ignores_reporter_failures(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    class _ThrowingReporter:
        def __init__(self) -> None:
            self.events: list[str] = []

        def phase_start(self, phase: str, **_kwargs) -> None:
            self.events.append(f"start:{phase}")
            raise RuntimeError("closed progress host")

        def phase_end(self, phase: str) -> None:
            self.events.append(f"end:{phase}")
            raise RuntimeError("closed progress host")

    paths = _ltx2_paths(tmp_path / "sample.mp4")
    artifact_paths = paths.artifact_paths()
    reporter = _ThrowingReporter()
    sink = SidecarArtifactSink(
        artifact_paths,
        enabled={STAGE_1_LATENTS},
        reporter=reporter,
    )

    with caplog.at_level("WARNING"):
        sink.save(
            distilled_stage_latents_artifact(
                1,
                video_latent=mx.zeros((1,), dtype=mx.float32),
                audio_latent=None,
                final=False,
            )
        )

    assert artifact_paths[STAGE_1_LATENTS].is_file()
    assert sink.manifest == {STAGE_1_LATENTS: str(artifact_paths[STAGE_1_LATENTS])}
    assert sink.errors == []
    assert reporter.events == [
        "start:save stage 1 latents",
        "end:save stage 1 latents",
    ]
    assert "Could not start sidecar reporting phase" in caplog.text
    assert "Could not end sidecar reporting phase" in caplog.text


def test_artifact_sink_does_not_retain_offered_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module

    class _Payload:
        pass

    monkeypatch.setattr(sidecars_module, "save_weights", lambda *_args, **_kwargs: None)
    paths = _ltx2_paths(tmp_path / "sample.mp4")
    sink = SidecarArtifactSink(
        paths.artifact_paths(),
        enabled={STAGE_1_LATENTS},
    )
    payload = _Payload()
    reference = weakref.ref(payload)
    sink.save(TensorArtifact(STAGE_1_LATENTS, (("video_latent", payload),)))

    del payload
    gc.collect()
    assert reference() is None


def test_artifact_sink_with_both_categories_disabled_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("disabled artifact sink attempted a write")

    monkeypatch.setattr(sidecars_module, "save_weights", fail_if_called)
    sink = SidecarArtifactSink(
        {},
        enabled=(),
    )
    sink.save(TensorArtifact("arbitrary_model_artifact", (("tensor", object()),)))
    assert sink.manifest == {}
    assert sink.errors == []


def test_run_record_and_execution_log_capture_status_and_timing(tmp_path: Path) -> None:
    paths = _ltx2_paths(tmp_path / "sample.mp4")
    now = [4.0]
    timings = TimingReporter(clock=lambda: now[0])
    initialize_execution_log(paths.execution_log, ["kinomlx", "--seed", "42"])
    record = RunRecord(
        paths.run_log,
        model="ltx2",
        invocation={"output": {"path": paths.video}},
        argv=["kinomlx", "--seed", "42"],
        timings=timings,
        planned_outputs={"video": str(paths.video)},
    )
    timings.phase_start("load models")
    now[0] = 7.5
    timings.phase_end("load models")
    record.write(
        status="completed",
        outputs={"video": str(paths.video)},
        output_fingerprints={"final_latents": "sha256:abc"},
        output_fingerprint_errors={"stage_1_latents": "OSError: receipt read failed"},
        generation={"model_version": "2.3"},
    )

    payload = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["timings"]["total_seconds"] == 3.5
    assert payload["timings"]["phases"][0]["duration_seconds"] == 3.5
    assert payload["generation"] == {"model_version": "2.3"}
    assert payload["output_fingerprints"] == {"final_latents": "sha256:abc"}
    assert payload["output_fingerprint_errors"] == {
        "stage_1_latents": "OSError: receipt read failed"
    }
    assert payload["sidecar_errors"] == []
    assert paths.execution_log.read_text(encoding="utf-8") == ("Command: kinomlx --seed 42\n\n")


def test_run_record_serializes_shared_sidecar_errors(tmp_path: Path) -> None:
    paths = _ltx2_paths(tmp_path / "sample.mp4")
    errors: list[dict[str, str]] = []
    record = RunRecord(
        paths.run_log,
        model="ltx2",
        invocation={},
        argv=["kinomlx"],
        timings=TimingReporter(),
        planned_outputs={"video": str(paths.video)},
        sidecar_errors=errors,
    )
    errors.append(
        {
            "artifact": "stage_1_latents",
            "path": str(paths.artifact_paths()[STAGE_1_LATENTS]),
            "error": "OSError: disk full",
        }
    )
    record.write(status="completed", outputs={"video": str(paths.video)})
    payload = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert payload["sidecar_errors"] == errors


def test_timing_reporter_defaults_to_monotonic_performance_clock() -> None:
    reporter = TimingReporter()
    assert reporter._clock is time.perf_counter
