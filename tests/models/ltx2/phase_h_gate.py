"""Standalone installed-wheel-capable LTX-2.5 Phase H capability gate."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import kinomlx
from kinomlx.cli.args import build_parser
from kinomlx.cli.config import assemble
from kinomlx.debug.sidecars import SidecarArtifactSink
from kinomlx.io.safetensors import load_weights
from kinomlx.models.ltx2.artifacts import FINAL_LATENTS, TEXT_CONDITIONING
from kinomlx.models.ltx2.components import NativeLTX2Components
from kinomlx.models.ltx2.pipelines.distilled import (
    generate_distilled,
    prepare_text_conditioning,
)
from kinomlx.models.ltx2.resources import ComponentKind, LTX2Resources, prepare_resources
from kinomlx.models.ltx2.runner import LTX2Runner
from kinomlx.models.ltx2.text_conditioning import EncodedTextConditioning
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.models.ltx2.upscaler.temporal import temporal_upsample_video
from kinomlx.reporting import RecordingReporter, TimingReporter

_WIDTH = 64
_HEIGHT = 64
_FRAMES = 9
_FPS = 24.0
_SEED = 26_082_200
_PROMPT = "A red sailboat crossing a calm blue lake, simple composition."
_OFFICIAL_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
_DEV_TRANSFORMER = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
_MODEL_SOURCE_FIELDS = (
    "weights_path",
    "gemma_path",
    "spatial_upscaler_path",
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "temporal_latent_upscaler_path",
    "duration_head_path",
)
_COMPONENT_ORACLE = Path(__file__).parent / "fixtures/ltx25_phase_h_component_oracles.json"


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else value


def _sha256(path: Path) -> str:
    resolved = path.resolve()
    if len(resolved.name) == 64 and all(char in "0123456789abcdef" for char in resolved.name):
        return resolved.name
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    for label, getter_name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        getter = getattr(mx, getter_name, None)
        try:
            result[label] = 0 if getter is None else max(0, int(getter()))
        except RuntimeError, TypeError, ValueError:
            result[label] = 0
    return result


def _reset_peak_memory() -> bool:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        return False
    try:
        reset()
    except RuntimeError, TypeError, ValueError:
        return False
    return True


def _release_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_json_value(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_ltx25_invocation():
    options = build_parser().parse_args(["--ltx-generation", "2.5", "--print-config"])
    invocation = assemble(options)
    remaining = {
        field: str(value)
        for field in _MODEL_SOURCE_FIELDS
        if (value := getattr(invocation.model_settings, field)) is not None
    }
    if remaining:
        raise AssertionError(
            f"Phase H requires generation-based discovery without path overrides: {remaining}"
        )
    return invocation


def _split_dev_resources(invocation: Any, base: LTX2Resources) -> LTX2Resources:
    root = base.transformer_path.parent.parent
    settings = dataclasses.replace(
        invocation.model_settings,
        transformer_path=root / "diffusion_models" / _DEV_TRANSFORMER,
        text_encoder_path=base.require(ComponentKind.TEXT_ENCODER).source_path,
        video_vae_path=base.require(ComponentKind.VIDEO_VAE).source_path,
        audio_vae_path=base.require(ComponentKind.AUDIO_VAE).source_path,
        spatial_upscaler_path=base.require(ComponentKind.SPATIAL_UPSCALER).source_path,
        temporal_latent_upscaler_path=base.require(
            ComponentKind.LATENT_TEMPORAL_UPSCALER
        ).source_path,
        duration_head_path=base.require(ComponentKind.DURATION_HEAD).source_path,
    )
    return prepare_resources(settings, infrastructure=invocation.settings)


@dataclass(frozen=True)
class _InjectedTextConditioner:
    text: EncodedTextConditioning

    def __call__(self, request, resources, *, reporter=None):
        del request, resources, reporter
        return self.text


def _prepare_injected_text(
    resources: LTX2Resources,
    path: Path,
) -> tuple[EncodedTextConditioning, dict[str, Any]]:
    events = RecordingReporter()
    timing = TimingReporter(events)
    sink = SidecarArtifactSink(
        {TEXT_CONDITIONING: path},
        enabled={TEXT_CONDITIONING},
        reporter=timing,
    )
    started = time.perf_counter()
    text = prepare_text_conditioning(
        DistilledRequest(
            prompt=_PROMPT,
            width=_WIDTH,
            height=_HEIGHT,
            frames=_FRAMES,
            fps=_FPS,
            seed=_SEED,
            generate_audio=False,
        ),
        resources,
        text_conditioner=None,
        reporter=timing,
        artifacts=sink,
    )
    mx.eval(text.video_encoding, text.audio_encoding, text.attention_mask)
    return text, {
        "elapsed_seconds": time.perf_counter() - started,
        "path": str(path),
        "sha256": _sha256(path),
        "video_shape": list(text.video_encoding.shape),
        "audio_shape": list(text.audio_encoding.shape),
        "attention_mask_shape": list(text.attention_mask.shape),
        "provenance": text.provenance.to_metadata(),
        "timing": timing.to_dict(),
        "events": events.events,
    }


def _component_canaries(resources: LTX2Resources) -> dict[str, Any]:
    fixture = json.loads(_COMPONENT_ORACLE.read_text(encoding="utf-8"))
    components = NativeLTX2Components()
    duration_fixture = fixture["duration"]
    if _sha256(resources.duration_head_path) != duration_fixture["artifact_sha256"]:
        raise AssertionError("duration-head artifact differs from the pinned oracle")
    video = np.sin(
        np.arange(math.prod(duration_fixture["video_shape"]), dtype=np.float32) * np.float32(0.007)
    ).reshape(duration_fixture["video_shape"])
    audio = np.cos(
        np.arange(math.prod(duration_fixture["audio_shape"]), dtype=np.float32) * np.float32(0.011)
    ).reshape(duration_fixture["audio_shape"])
    with components.duration_predictor(resources) as predictor:
        callable_predictor = predictor
        seconds = callable_predictor(
            mx.array(video).astype(mx.bfloat16),
            mx.array(audio).astype(mx.bfloat16),
        )
        mx.eval(seconds)
        frame_count = predictor.predict_num_frames(
            mx.array(video).astype(mx.bfloat16),
            mx.array(audio).astype(mx.bfloat16),
            frame_rate=_FPS,
            temporal_compression_ratio=8,
        )
    if not math.isclose(
        float(seconds.item()),
        duration_fixture["torch_bfloat16_seconds"],
        abs_tol=1e-6,
    ):
        raise AssertionError("duration-head output differs from the pinned Torch oracle")
    if frame_count != duration_fixture["frames_at_24_fps"]:
        raise AssertionError("duration-head frame snapping differs from the pinned oracle")
    duration_receipt = {
        "artifact_sha256": duration_fixture["artifact_sha256"],
        "seconds": float(seconds.item()),
        "frames_at_24_fps": frame_count,
    }
    del seconds
    _release_mlx()

    temporal_fixture = fixture["temporal_upscaler"]
    if _sha256(resources.temporal_upscaler_path) != temporal_fixture["artifact_sha256"]:
        raise AssertionError("temporal-upscaler artifact differs from the pinned oracle")
    latent = np.sin(
        np.arange(math.prod(temporal_fixture["input_shape"]), dtype=np.float32) * np.float32(0.017)
    ).reshape(temporal_fixture["input_shape"])
    with components.temporal_upscaler(resources) as upscaler:
        raw = upscaler(mx.array(latent).astype(mx.bfloat16)).astype(mx.float32)
        station = temporal_upsample_video(
            mx.array(latent).astype(mx.bfloat16),
            upscaler,
        ).astype(mx.float32)
        mx.eval(raw, station)
    if list(raw.shape) != temporal_fixture["output_shape"]:
        raise AssertionError(f"unexpected temporal output shape {tuple(raw.shape)}")
    raw_values = np.asarray(raw).reshape(-1)
    actual_anchors = raw_values[temporal_fixture["anchor_indices"]]
    np.testing.assert_allclose(
        actual_anchors,
        temporal_fixture["torch_bfloat16_anchors"],
        rtol=0.0,
        atol=temporal_fixture["anchor_atol"],
    )
    if not bool(mx.all(mx.isfinite(station)).item()):
        raise AssertionError("temporal public station returned nonfinite values")
    temporal_receipt = {
        "artifact_sha256": temporal_fixture["artifact_sha256"],
        "raw_shape": list(raw.shape),
        "raw_mean": float(np.mean(raw_values)),
        "raw_peak": float(np.max(np.abs(raw_values))),
        "raw_rms": float(np.sqrt(np.mean(np.square(raw_values)))),
        "station_shape": list(station.shape),
        "station_finite": True,
    }
    del raw, station
    _release_mlx()
    return {"duration": duration_receipt, "temporal_upscaler": temporal_receipt}


def _write_cross_generation_adapter(path: Path) -> None:
    a = mx.sin(mx.arange(4096, dtype=mx.float32) * 0.01)[None, :] * 0.02
    b = mx.cos(mx.arange(128, dtype=mx.float32) * 0.03)[:, None] * 0.02
    mx.save_safetensors(
        str(path),
        {
            "diffusion_model.proj_out.lora_A.weight": a.astype(mx.bfloat16),
            "diffusion_model.proj_out.lora_B.weight": b.astype(mx.bfloat16),
        },
        metadata={
            "model_version": "2.3.0-community",
            "description": "Phase H structurally valid cross-generation canary",
        },
    )


def _latent_summary(path: Path, *, frame_count: int) -> dict[str, Any]:
    weights = load_weights(path)
    latent = weights["video_latent"].astype(mx.float32)
    mx.eval(latent)
    expected_shape = (1, 128, (frame_count - 1) // 8 + 1, 2, 2)
    if tuple(latent.shape) != expected_shape:
        raise AssertionError(f"final latent shape {tuple(latent.shape)} != {expected_shape}")
    if not bool(mx.all(mx.isfinite(latent)).item()) or not bool(mx.any(latent != 0).item()):
        raise AssertionError("final latent is zero or nonfinite")
    receipt = {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(latent.shape),
        "mean": float(mx.mean(latent).item()),
        "rms": float(mx.sqrt(mx.mean(mx.square(latent))).item()),
        "peak": float(mx.max(mx.abs(latent)).item()),
    }
    weights.clear()
    del latent
    _release_mlx()
    return receipt


def _run_case(
    name: str,
    resources: LTX2Resources,
    text: EncodedTextConditioning,
    output_dir: Path,
    *,
    frames: int | None,
    generated_keyframes: int = 0,
    loras: tuple[tuple[Path, float], ...] = (),
) -> dict[str, Any]:
    latent_path = output_dir / f"{name}.safetensors"
    events = RecordingReporter()
    timing = TimingReporter(events)
    artifact_sink = SidecarArtifactSink(
        {FINAL_LATENTS: latent_path},
        enabled={FINAL_LATENTS},
        reporter=timing,
    )
    request = DistilledRequest(
        prompt=_PROMPT,
        width=_WIDTH,
        height=_HEIGHT,
        frames=frames,
        fps=_FPS,
        seed=_SEED,
        generate_audio=False,
        generated_keyframes=generated_keyframes,
        lora_paths=tuple(path for path, _strength in loras),
        lora_strengths=tuple(strength for _path, strength in loras),
    )
    runner = LTX2Runner(
        resources=resources,
        text_conditioner=_InjectedTextConditioner(text),
        reporter=timing,
        artifact_sink=artifact_sink,
    )
    peak_reset = _reset_peak_memory()
    memory_before = _memory_snapshot()
    started = time.perf_counter()
    output = runner.run(generate_distilled, request)
    elapsed = time.perf_counter() - started
    frame_count = output.frame_count
    metadata = _json_value(output.metadata)
    output.close()
    if output.audio_waveform is not None or output.audio_sample_rate is not None:
        raise AssertionError(f"{name}: audio-off run returned audio")
    if frames is None:
        if frame_count % 8 != 1 or not 25 <= frame_count <= 473:
            raise AssertionError(f"{name}: auto-duration returned invalid {frame_count} frames")
    elif frame_count != frames:
        raise AssertionError(f"{name}: returned {frame_count} frames, expected {frames}")
    generated = metadata["generated_keyframe_indices"]
    if len(generated) != generated_keyframes:
        raise AssertionError(f"{name}: generated-keyframe receipt is {generated!r}")
    lora_receipts = metadata["lora_receipts"]
    if len(lora_receipts) != len(loras):
        raise AssertionError(f"{name}: LoRA receipt count differs from the profile")
    latent = _latent_summary(latent_path, frame_count=frame_count)
    return {
        "name": name,
        "transformer_source": str(resources.transformer_path),
        "transformer_fingerprint": resources.checkpoint.source_fingerprint,
        "request": {
            "seed": request.seed,
            "frames": frames,
            "resolved_frames": frame_count,
            "generated_keyframes": generated_keyframes,
            "loras": [
                {"path": str(path), "sha256": _sha256(path), "strength": strength}
                for path, strength in loras
            ],
        },
        "elapsed_seconds": elapsed,
        "peak_memory_reset": peak_reset,
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "timing": timing.to_dict(),
        "events": events.events,
        "metadata": metadata,
        "final_latent": latent,
    }


def _delta_summary(left_path: Path, right_path: Path) -> dict[str, float]:
    left_weights = load_weights(left_path)
    right_weights = load_weights(right_path)
    left = left_weights["video_latent"].astype(mx.float32)
    right = right_weights["video_latent"].astype(mx.float32)
    delta = left - right
    left_norm = mx.sqrt(mx.sum(mx.square(left)))
    right_norm = mx.sqrt(mx.sum(mx.square(right)))
    cosine = mx.sum(left * right) / (left_norm * right_norm)
    values = (
        mx.sqrt(mx.mean(mx.square(delta))),
        mx.max(mx.abs(delta)),
        mx.mean(mx.abs(delta)),
        cosine,
    )
    mx.eval(*values)
    result = {
        "rms": float(values[0].item()),
        "peak": float(values[1].item()),
        "mean_abs": float(values[2].item()),
        "cosine": float(values[3].item()),
    }
    if not all(math.isfinite(value) for value in result.values()) or result["rms"] <= 0:
        raise AssertionError(f"A/B latent delta is invalid: {result}")
    left_weights.clear()
    right_weights.clear()
    del left, right, delta
    _release_mlx()
    return result


def _assert_official_receipt(case: dict[str, Any]) -> None:
    receipts = case["metadata"]["lora_receipts"]
    if len(receipts) != 1:
        raise AssertionError(f"{case['name']}: expected one official LoRA receipt")
    receipt = receipts[0]
    expected = {
        "complete_targets": 1660,
        "placed_targets": 1660,
        "structural_coverage": 1.0,
        "warning": False,
        "generation_mismatch": False,
    }
    for key, value in expected.items():
        if receipt[key] != value:
            raise AssertionError(f"{case['name']}: receipt {key}={receipt[key]!r}")
    if receipt["stages"] != [1, 2]:
        raise AssertionError(f"{case['name']}: official LoRA did not cover both stages")


def run_phase_h(
    output_dir: Path,
    *,
    expected_package_root: Path | None = None,
) -> Path:
    """Run real Phase H canaries and return the durable JSON receipt path."""
    package_file = Path(kinomlx.__file__).resolve()
    if expected_package_root is not None:
        expected = expected_package_root.expanduser().resolve()
        if not package_file.is_relative_to(expected):
            raise AssertionError(f"loaded {package_file}, expected a package below {expected}")
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "phase-h-receipt.json"
    invocation = _resolve_ltx25_invocation()
    base = prepare_resources(
        invocation.model_settings,
        infrastructure=invocation.settings,
    )
    dev = _split_dev_resources(invocation, base)
    root = base.transformer_path.parent.parent
    official_lora = root / "loras" / _OFFICIAL_LORA
    if _sha256(official_lora) != "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5":
        raise AssertionError("official distilled LoRA differs from the pinned Phase H artifact")
    cross_lora = output_dir / "cross-generation-2.3-proj-out-rank1.safetensors"
    _write_cross_generation_adapter(cross_lora)
    text, text_receipt = _prepare_injected_text(
        base,
        output_dir / "injected-text-conditioning.safetensors",
    )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "package": {
            "version": importlib.metadata.version("kinomlx"),
            "module_path": str(package_file),
        },
        "selection": {
            "surface": "--ltx-generation 2.5",
            "distilled_transformer": str(base.transformer_path),
            "dev_transformer": str(dev.transformer_path),
            "official_lora": str(official_lora),
            "official_lora_sha256": _sha256(official_lora),
            "cross_generation_lora": str(cross_lora),
            "cross_generation_lora_sha256": _sha256(cross_lora),
        },
        "component_canaries": _component_canaries(base),
        "injected_text": text_receipt,
        "cases": [],
        "comparisons": {},
    }
    _write_receipt(receipt_path, receipt)

    cases = (
        _run_case(
            "distilled-baseline",
            base,
            text,
            output_dir,
            frames=_FRAMES,
        ),
        _run_case(
            "dev-plus-official-distilled-lora",
            dev,
            text,
            output_dir,
            frames=_FRAMES,
            loras=((official_lora, 1.0),),
        ),
        _run_case(
            "distilled-plus-official-distilled-lora",
            base,
            text,
            output_dir,
            frames=_FRAMES,
            loras=((official_lora, 1.0),),
        ),
        _run_case(
            "auto-duration-generated-keyframe-cross-generation-lora",
            base,
            text,
            output_dir,
            frames=None,
            generated_keyframes=1,
            loras=((cross_lora, 0.5),),
        ),
    )
    receipt["cases"] = list(cases)
    for case in cases[1:3]:
        _assert_official_receipt(case)
    cross_receipt = cases[3]["metadata"]["lora_receipts"][0]
    cross_expected = {
        "base_model_generation": "2.5",
        "declared_model_generation": "2.3",
        "generation_mismatch": True,
        "complete_targets": 1,
        "placed_targets": 1,
        "structural_coverage": 1.0,
        "warning": False,
        "strength": 0.5,
        "stages": [1, 2],
    }
    for key, value in cross_expected.items():
        if cross_receipt[key] != value:
            raise AssertionError(f"cross-generation receipt {key}={cross_receipt[key]!r}")

    baseline_path = Path(cases[0]["final_latent"]["path"])
    receipt["comparisons"] = {
        "dev_plus_official_vs_fused_distilled": _delta_summary(
            Path(cases[1]["final_latent"]["path"]),
            baseline_path,
        ),
        "official_on_distilled_vs_distilled_baseline": _delta_summary(
            Path(cases[2]["final_latent"]["path"]),
            baseline_path,
        ),
    }
    receipt["status"] = "passed"
    receipt["total_case_elapsed_seconds"] = sum(case["elapsed_seconds"] for case in cases)
    receipt["final_memory"] = _memory_snapshot()
    _write_receipt(receipt_path, receipt)
    del text
    _release_mlx()
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-package-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    receipt = run_phase_h(
        options.output_dir,
        expected_package_root=options.expected_package_root,
    )
    sys.stdout.write(
        json.dumps({"receipt": str(receipt), "sha256": _sha256(receipt)}, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
