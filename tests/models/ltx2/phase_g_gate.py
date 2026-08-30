"""Standalone installed-wheel-capable LTX-2.5 Phase G SDR gate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import mlx.core as mx

import kinomlx
from kinomlx.cli.args import build_parser
from kinomlx.cli.config import OutputConfig, assemble
from kinomlx.cli.output import write_generation
from kinomlx.components import ComponentLease
from kinomlx.debug.sidecars import SidecarArtifactSink
from kinomlx.io.image import save_image
from kinomlx.models.ltx2.artifacts import (
    FINAL_LATENTS,
    TEXT_CONDITIONING,
    sidecar_paths,
)
from kinomlx.models.ltx2.components import NativeLTX2Components
from kinomlx.models.ltx2.pipelines.distilled import generate_distilled
from kinomlx.models.ltx2.resources import LTX2Resources, prepare_resources
from kinomlx.models.ltx2.runner import GenerationOutput, LTX2Runner
from kinomlx.models.ltx2.types import (
    DistilledRequest,
    ImageConditioningConfig,
)
from kinomlx.reporting import RecordingReporter, TimingReporter

_WIDTH = 64
_HEIGHT = 64
_FRAMES = 9
_FPS = 24.0
_PROMPT = "A red sailboat crossing a calm blue lake, simple composition."
_CASE_NAMES = (
    "text-to-video",
    "first-frame-image-to-video",
    "keyframe-image-to-video",
    "audio-off",
    "joint-audio-video",
)
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
_MAX_UNACCOUNTED_VAE_ENTRY_BYTES = 1 << 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(enum_value, str):
        return enum_value
    return value


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


def _assert_vae_runtime_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    case_name: str,
) -> None:
    """Reject heavyweight predecessor residency at the LTX-2.5 VAE station."""
    vae_decode = diagnostics.get("vae_decode")
    if not isinstance(vae_decode, Mapping):
        raise AssertionError(f"LTX-2.5 {case_name}: missing VAE runtime diagnostics")
    entry = vae_decode.get("entry_memory")
    tiling = vae_decode.get("tiling")
    if not isinstance(entry, Mapping) or not isinstance(tiling, Mapping):
        raise AssertionError(f"LTX-2.5 {case_name}: incomplete VAE runtime diagnostics")
    unaccounted = int(entry.get("unaccounted_active_bytes", -1))
    if not 0 <= unaccounted <= _MAX_UNACCOUNTED_VAE_ENTRY_BYTES:
        raise AssertionError(
            f"LTX-2.5 {case_name}: VAE entry retains {unaccounted} unaccounted bytes; "
            f"limit is {_MAX_UNACCOUNTED_VAE_ENTRY_BYTES}"
        )
    if int(tiling.get("total_tiles", 0)) != 1:
        raise AssertionError(
            f"LTX-2.5 {case_name}: tiny gate should resolve one VAE tile, got "
            f"{tiling.get('total_tiles')!r}"
        )


class _RecordingComponents:
    """Decorate native leases with balanced station and memory evidence."""

    def __init__(self, inner: NativeLTX2Components) -> None:
        self._inner = inner
        self._started = time.perf_counter()
        self._serials: Counter[str] = Counter()
        self._active: set[str] = set()
        self.events: list[dict[str, Any]] = []

    @property
    def active(self) -> frozenset[str]:
        return frozenset(self._active)

    def _record(self, action: str, component: str, instance: str) -> None:
        self.events.append(
            {
                "action": action,
                "component": component,
                "instance": instance,
                "time_seconds": time.perf_counter() - self._started,
                "memory": _memory_snapshot(),
            }
        )

    def _load(self, name: str, loader: Callable[[], ComponentLease[Any]]) -> ComponentLease[Any]:
        self._serials[name] += 1
        instance = f"{name}#{self._serials[name]}"
        self._record("load_start", name, instance)
        try:
            inner_lease = loader()
        except BaseException:
            self._record("load_error", name, instance)
            raise
        self._active.add(instance)
        self._record("load_complete", name, instance)

        def close_component(_component: Any) -> None:
            try:
                inner_lease.close()
            finally:
                self._active.discard(instance)
                self._record("close", name, instance)

        return ComponentLease(inner_lease.value, close_component=close_component)

    def transformer(self, resources, profile=()):
        return self._load(
            "transformer",
            lambda: self._inner.transformer(
                resources,
                profile,
            ),
        )

    def video_encoder(self, resources):
        return self._load(
            "video_encoder",
            lambda: self._inner.video_encoder(resources),
        )

    def spatial_upscaler(self, resources):
        return self._load(
            "spatial_upscaler",
            lambda: self._inner.spatial_upscaler(resources),
        )

    def duration_predictor(self, resources):
        return self._load(
            "duration_predictor",
            lambda: self._inner.duration_predictor(resources),
        )

    def audio_decoder(self, resources):
        return self._load(
            "audio_decoder",
            lambda: self._inner.audio_decoder(resources),
        )

    def vocoder(self, resources):
        return self._load("vocoder", lambda: self._inner.vocoder(resources))

    def video_decoder(self, resources):
        return self._load(
            "video_decoder",
            lambda: self._inner.video_decoder(resources),
        )


class _InspectingFrameStream:
    """Collect numeric facts without retaining frames or breaking one-pass flow."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.spec = source.spec
        self.frame_count = source.frame_count
        self.consumed = 0
        self.nonzero = False
        self.minimum = math.inf
        self.maximum = -math.inf
        self._mean_square_sum = 0.0

    @property
    def receipts(self):
        return self._source.receipts

    @property
    def rms(self) -> float:
        if self.consumed == 0:
            return 0.0
        return math.sqrt(self._mean_square_sum / self.consumed)

    def __iter__(self):
        for frame in self._source:
            if tuple(frame.shape) != (_HEIGHT, _WIDTH, 3):
                raise AssertionError(f"unexpected decoded frame shape {tuple(frame.shape)}")
            if frame.dtype != mx.float16:
                raise AssertionError(f"unexpected decoded frame dtype {frame.dtype}")
            if not bool(mx.all(mx.isfinite(frame)).item()):
                raise AssertionError("decoded frame contains nonfinite values")
            if not bool(mx.all((frame >= 0.0) & (frame <= 1.0)).item()):
                raise AssertionError("decoded SDR frame falls outside [0, 1]")
            frame32 = frame.astype(mx.float32)
            self.minimum = min(self.minimum, float(mx.min(frame32).item()))
            self.maximum = max(self.maximum, float(mx.max(frame32).item()))
            self._mean_square_sum += float(mx.mean(mx.square(frame32)).item())
            self.nonzero = self.nonzero or bool(mx.any(frame != 0).item())
            self.consumed += 1
            yield frame

    def close(self) -> None:
        self._source.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed,
            "nonzero": self.nonzero,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "rms": self.rms,
        }


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Phase G requires {name} on PATH")
    return path


def _run_tool(command: Sequence[str]) -> bytes:
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{' '.join(command[:2])} failed: {stderr}")
    return result.stdout


def _probe_media(path: Path, *, expect_audio: bool, expected_audio_samples: int | None) -> dict:
    ffprobe = _require_tool("ffprobe")
    ffmpeg = _require_tool("ffmpeg")
    raw_probe = _run_tool(
        (
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        )
    )
    probe = json.loads(raw_probe.decode("utf-8"))
    streams = probe.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise AssertionError(f"expected one video stream, got {len(video_streams)}")
    if len(audio_streams) != int(expect_audio):
        raise AssertionError(
            f"expected {int(expect_audio)} audio streams, got {len(audio_streams)}"
        )

    video = video_streams[0]
    expected_video = {
        "codec_name": "hevc",
        "profile": "Main 10",
        "pix_fmt": "yuv420p10le",
        "width": _WIDTH,
        "height": _HEIGHT,
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }
    for field, expected in expected_video.items():
        if video.get(field) != expected:
            raise AssertionError(f"video {field}={video.get(field)!r}, expected {expected!r}")
    if int(video.get("nb_frames", 0)) != _FRAMES:
        raise AssertionError(f"video frame count is {video.get('nb_frames')!r}")
    video_duration = float(video["duration"])
    if not math.isclose(video_duration, _FRAMES / _FPS, abs_tol=1 / 48_000):
        raise AssertionError(f"video duration is {video_duration}")

    decoded_video = _run_tool(
        (
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        )
    )
    expected_video_bytes = _FRAMES * _WIDTH * _HEIGHT * 3
    if len(decoded_video) != expected_video_bytes:
        raise AssertionError(
            f"complete video decode returned {len(decoded_video)} bytes, "
            f"expected {expected_video_bytes}"
        )
    if len(set(decoded_video)) <= 1:
        raise AssertionError("muxed video decodes to one uniform byte value")

    result: dict[str, Any] = {
        "ffprobe": probe,
        "video_duration_seconds": video_duration,
        "decoded_video_bytes": len(decoded_video),
        "decoded_video_sha256": hashlib.sha256(decoded_video).hexdigest(),
    }
    if not expect_audio:
        return result

    audio = audio_streams[0]
    if audio.get("codec_name") != "alac":
        raise AssertionError(f"audio codec is {audio.get('codec_name')!r}")
    if int(audio.get("sample_rate", 0)) != 48_000:
        raise AssertionError(f"audio sample rate is {audio.get('sample_rate')!r}")
    if int(audio.get("channels", 0)) != 2:
        raise AssertionError(f"audio channel count is {audio.get('channels')!r}")
    audio_duration = float(audio["duration"])
    if expected_audio_samples is None:
        raise AssertionError("joint output has no source audio sample count")
    expected_audio_duration = expected_audio_samples / 48_000
    if not math.isclose(audio_duration, expected_audio_duration, abs_tol=1 / 48_000):
        raise AssertionError(
            f"audio duration {audio_duration} does not match {expected_audio_duration}"
        )
    if not 0.0 <= video_duration - audio_duration <= 1 / _FPS:
        raise AssertionError(
            f"noncausal media durations: video={video_duration}, audio={audio_duration}"
        )

    decoded_audio = _run_tool(
        (
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "pipe:1",
        )
    )
    if len(decoded_audio) % (2 * 4):
        raise AssertionError("decoded stereo float32 audio has a partial sample")
    samples = array("f")
    samples.frombytes(decoded_audio)
    if sys.byteorder != "little":
        samples.byteswap()
    decoded_audio_frames = len(samples) // 2
    if decoded_audio_frames != expected_audio_samples:
        raise AssertionError(
            f"complete audio decode returned {decoded_audio_frames} samples per channel, "
            f"expected {expected_audio_samples}"
        )
    audio_peak = max(abs(sample) for sample in samples)
    audio_rms = math.sqrt(math.fsum(sample * sample for sample in samples) / len(samples))
    if not math.isfinite(audio_rms) or audio_rms <= 1e-8 or audio_peak <= 1e-8:
        raise AssertionError(
            f"muxed audio is silent or nonfinite: rms={audio_rms}, peak={audio_peak}"
        )
    result.update(
        {
            "audio_duration_seconds": audio_duration,
            "decoded_audio_samples_per_channel": decoded_audio_frames,
            "decoded_audio_rms": audio_rms,
            "decoded_audio_peak": audio_peak,
            "decoded_audio_sha256": hashlib.sha256(decoded_audio).hexdigest(),
        }
    )
    return result


def _write_condition_image(path: Path) -> None:
    x = mx.linspace(0.0, 1.0, _WIDTH, dtype=mx.float32)[None, :]
    y = mx.linspace(0.0, 1.0, _HEIGHT, dtype=mx.float32)[:, None]
    red = mx.broadcast_to(x, (_HEIGHT, _WIDTH))
    green = mx.broadcast_to(y, (_HEIGHT, _WIDTH))
    blue = (red + green) * 0.5
    image = mx.stack((red, green, blue), axis=-1)
    mx.eval(image)
    save_image(path, image)


@dataclass(frozen=True)
class _Case:
    name: str
    seed: int
    generate_audio: bool = False
    image_frame_index: int | None = None


def _cases(names: Sequence[str]) -> tuple[_Case, ...]:
    known = {
        "text-to-video": _Case("text-to-video", 70),
        "first-frame-image-to-video": _Case(
            "first-frame-image-to-video",
            71,
            image_frame_index=0,
        ),
        "keyframe-image-to-video": _Case(
            "keyframe-image-to-video",
            72,
            image_frame_index=_FRAMES - 1,
        ),
        "audio-off": _Case("audio-off", 73),
        "joint-audio-video": _Case("joint-audio-video", 74, generate_audio=True),
    }
    unknown = sorted(set(names) - known.keys())
    if unknown:
        raise ValueError(f"unknown Phase G case: {', '.join(unknown)}")
    return tuple(known[name] for name in names)


def _resource_receipt(resources: LTX2Resources) -> dict[str, Any]:
    transformer = resources.transformer_config
    return {
        "checkpoint": _json_value(resources.checkpoint),
        "capabilities": _json_value(resources.capabilities),
        "dtype_policy": resources.dtype_policy.to_metadata(),
        "components": [
            {
                "kind": component.kind.value,
                "source_path": str(component.source_path),
                "source_fingerprint": component.source_fingerprint,
                "cache_path": (None if component.cache_path is None else str(component.cache_path)),
            }
            for component in resources.components
        ],
        "video_vae": {
            "kind": resources.capabilities.video_vae_kind,
            "encoder_scale": _json_value(resources.video_vae_config.encoder_scale),
            "decoder_scale": _json_value(resources.video_vae_config.decoder_scale),
            "signal_domain": resources.video_vae_config.signal_domain,
            "causal_decoder": resources.video_vae_config.causal_decoder,
        },
        "transformer": (
            None
            if transformer is None
            else {
                "num_layers": transformer.num_layers,
                "video_heads": transformer.video_heads,
                "video_head_dim": transformer.video_head_dim,
                "video_max_pos": list(transformer.video_max_pos),
                "config_digest": transformer.config_digest,
                "inferred_fields": list(transformer.inferred_fields),
            }
        ),
    }


def _assert_geometry_is_metadata_derived(resources: LTX2Resources) -> dict[str, Any]:
    scale = resources.video_vae_config.encoder_scale
    transformer = resources.transformer_config
    if resources.capabilities.model_generation != "2.5":
        raise AssertionError(
            f"Phase G selected LTX-{resources.capabilities.model_generation}, expected 2.5"
        )
    if resources.capabilities.video_vae_kind != "native-conv3d":
        raise AssertionError(
            f"Phase G requires the convolutional VAE, got {resources.capabilities.video_vae_kind}"
        )
    derived_width = scale.width * 2
    derived_height = scale.height * 2
    derived_frames = scale.time + 1
    if (derived_width, derived_height, derived_frames) != (_WIDTH, _HEIGHT, _FRAMES):
        raise AssertionError(
            "Phase G canary constants no longer match the prepared VAE metadata: "
            f"{(derived_width, derived_height, derived_frames)}"
        )
    if transformer is None:
        raise AssertionError("prepared LTX-2.5 resources have no transformer constructor facts")
    latent_positions = (
        (derived_frames - 1) // scale.time + 1,
        derived_height // scale.height,
        derived_width // scale.width,
    )
    if any(
        actual > maximum
        for actual, maximum in zip(
            latent_positions,
            transformer.video_max_pos,
            strict=True,
        )
    ):
        raise AssertionError(
            f"canary latent positions {latent_positions} exceed {transformer.video_max_pos}"
        )
    return {
        "derivation": "width=2*vae_scale.width; height=2*vae_scale.height; frames=vae_scale.time+1",
        "pixel_shape": [1, 3, derived_frames, derived_height, derived_width],
        "latent_positions": list(latent_positions),
        "transformer_video_max_pos": list(transformer.video_max_pos),
    }


def _assert_balanced_stations(events: Sequence[dict[str, Any]], case: _Case) -> None:
    loaded = Counter(event["component"] for event in events if event["action"] == "load_complete")
    closed = Counter(event["component"] for event in events if event["action"] == "close")
    expected = Counter(
        {
            "transformer": 1,
            "spatial_upscaler": 1,
            "video_decoder": 1,
        }
    )
    if case.image_frame_index is not None:
        expected["video_encoder"] = 2
    if case.generate_audio:
        expected["audio_decoder"] = 1
        expected["vocoder"] = 1
    if loaded != expected:
        raise AssertionError(f"{case.name}: component loads {loaded} != {expected}")
    if closed != loaded:
        raise AssertionError(f"{case.name}: component closes {closed} != loads {loaded}")
    loaded_instances = {event["instance"] for event in events if event["action"] == "load_complete"}
    closed_instances = {event["instance"] for event in events if event["action"] == "close"}
    if loaded_instances != closed_instances:
        raise AssertionError(f"{case.name}: unbalanced component instances")


def _resolve_ltx25_settings():
    options = build_parser().parse_args(["--ltx-generation", "2.5", "--print-config"])
    invocation = assemble(options)
    if invocation.model_settings.model_generation != "2.5":
        raise AssertionError("CLI generation selector did not resolve LTX-2.5")
    remaining = {
        field: str(value)
        for field in _MODEL_SOURCE_FIELDS
        if (value := getattr(invocation.model_settings, field)) is not None
    }
    if remaining:
        raise AssertionError(
            f"CLI generation selector retained lower-precedence checkpoint paths: {remaining}"
        )
    return invocation


def _case_request(
    case: _Case,
    *,
    image_path: Path,
    text_conditioning: Path | None,
) -> DistilledRequest:
    image = (
        None
        if case.image_frame_index is None
        else ImageConditioningConfig(
            path=image_path,
            frame_index=case.image_frame_index,
            strength=0.95,
        )
    )
    return DistilledRequest(
        prompt=_PROMPT if text_conditioning is None else "",
        text_conditioning=text_conditioning,
        width=_WIDTH,
        height=_HEIGHT,
        frames=_FRAMES,
        fps=_FPS,
        seed=case.seed,
        generate_audio=case.generate_audio,
        image=image,
    )


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_json_value(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_phase_g(
    output_dir: Path,
    *,
    case_names: Sequence[str] = _CASE_NAMES,
    expected_package_root: Path | None = None,
) -> Path:
    """Run selected real cases and return the durable JSON receipt path."""
    package_file = Path(kinomlx.__file__).resolve()
    if expected_package_root is not None:
        expected = expected_package_root.expanduser().resolve()
        if not package_file.is_relative_to(expected):
            raise AssertionError(f"loaded {package_file}, expected a package below {expected}")
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "phase-g-receipt.json"
    image_path = output_dir / "condition.png"
    _write_condition_image(image_path)

    preparation_events = RecordingReporter()
    preparation_timing = TimingReporter(preparation_events)
    prepare_started = time.perf_counter()
    invocation = _resolve_ltx25_settings()
    resources = prepare_resources(
        invocation.model_settings,
        infrastructure=invocation.settings,
        reporter=preparation_timing,
    )
    preparation_elapsed = time.perf_counter() - prepare_started
    geometry = _assert_geometry_is_metadata_derived(resources)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "package": {
            "version": importlib.metadata.version("kinomlx"),
            "module_path": str(package_file),
            "expected_package_root": (
                None if expected_package_root is None else str(expected_package_root)
            ),
        },
        "selection": {
            "surface": "--ltx-generation 2.5",
            "explicit_component_paths": False,
            "environment_weights_path_present": "KINO_WEIGHTS_PATH" in os.environ,
        },
        "geometry": geometry,
        "resources": _resource_receipt(resources),
        "preparation": {
            "elapsed_seconds": preparation_elapsed,
            "timing": preparation_timing.to_dict(),
            "reporter_events": preparation_events.events,
            "memory": _memory_snapshot(),
        },
        "inputs": {
            "condition_image": str(image_path),
            "condition_image_sha256": _sha256(image_path),
            "prompt": _PROMPT,
        },
        "requested_cases": list(case_names),
        "cases": [],
    }
    _write_receipt(receipt_path, receipt)

    conditioning_path = output_dir / "shared-text-conditioning.safetensors"
    for case in _cases(case_names):
        text_path = conditioning_path if conditioning_path.is_file() else None
        request = _case_request(
            case,
            image_path=image_path,
            text_conditioning=text_path,
        )
        video_path = output_dir / f"{case.name}.mp4"
        latent_path = sidecar_paths(video_path)[FINAL_LATENTS]
        artifact_paths = {FINAL_LATENTS: latent_path}
        enabled = {FINAL_LATENTS}
        if text_path is None:
            artifact_paths[TEXT_CONDITIONING] = conditioning_path
            enabled.add(TEXT_CONDITIONING)

        reporter_events = RecordingReporter()
        timing = TimingReporter(reporter_events)
        artifact_sink = SidecarArtifactSink(
            artifact_paths,
            enabled=enabled,
            reporter=timing,
        )
        components = _RecordingComponents(NativeLTX2Components(reporter=timing))
        runner = LTX2Runner(
            resources=resources,
            components=components,
            reporter=timing,
            artifact_sink=artifact_sink,
        )
        peak_reset = _reset_peak_memory()
        memory_before = _memory_snapshot()
        started = time.perf_counter()
        output = runner.run(generate_distilled, request)
        source_frames = output.frames
        inspected = _InspectingFrameStream(source_frames)
        audio_samples = None
        source_audio: dict[str, Any] | None = None
        if case.generate_audio:
            waveform = output.audio_waveform
            if waveform is None or output.audio_sample_rate != 48_000:
                output.close()
                raise AssertionError(f"{case.name}: missing 48 kHz joint audio")
            if tuple(waveform.shape[:2]) != (1, 2):
                output.close()
                raise AssertionError(f"{case.name}: audio shape is {tuple(waveform.shape)}")
            if not bool(mx.all(mx.isfinite(waveform)).item()):
                output.close()
                raise AssertionError(f"{case.name}: source audio contains nonfinite values")
            waveform32 = waveform.astype(mx.float32)
            audio_samples = int(waveform.shape[-1])
            source_audio = {
                "shape": list(waveform.shape),
                "sample_rate": output.audio_sample_rate,
                "rms": float(mx.sqrt(mx.mean(mx.square(waveform32))).item()),
                "peak": float(mx.max(mx.abs(waveform32)).item()),
            }
            if source_audio["rms"] <= 1e-8 or source_audio["peak"] <= 1e-8:
                output.close()
                raise AssertionError(f"{case.name}: source audio is silent")
        elif output.audio_waveform is not None or output.audio_sample_rate is not None:
            output.close()
            raise AssertionError(f"{case.name}: audio-off request returned audio")

        wrapped = GenerationOutput(
            frames=inspected,
            audio_waveform=output.audio_waveform,
            audio_sample_rate=output.audio_sample_rate,
            metadata=output.metadata,
            diagnostics_provider=output.runtime_diagnostics,
        )
        try:
            muxed = write_generation(
                wrapped,
                OutputConfig(path=video_path),
                fps=_FPS,
                reporter=timing,
            )
        finally:
            output.close()
        elapsed = time.perf_counter() - started
        runtime_diagnostics = output.runtime_diagnostics()
        _assert_vae_runtime_diagnostics(runtime_diagnostics, case_name=case.name)
        if muxed != video_path:
            raise AssertionError(f"{case.name}: sink returned unexpected path {muxed}")
        if inspected.consumed != _FRAMES or not inspected.nonzero:
            raise AssertionError(f"{case.name}: incomplete or zero source frame stream")
        if components.active:
            raise AssertionError(f"{case.name}: live component leases remain: {components.active}")
        _assert_balanced_stations(components.events, case)
        phase_snapshot = timing.to_dict()
        active_phases = [
            phase["phase"] for phase in phase_snapshot["phases"] if phase["status"] != "completed"
        ]
        if active_phases:
            raise AssertionError(f"{case.name}: active reporter phases remain: {active_phases}")
        media = _probe_media(
            video_path,
            expect_audio=case.generate_audio,
            expected_audio_samples=audio_samples,
        )
        case_receipt = {
            "name": case.name,
            "request": {
                "seed": case.seed,
                "generate_audio": case.generate_audio,
                "image_frame_index": case.image_frame_index,
                "text_source": "saved-conditioning" if text_path is not None else "prompt",
            },
            "elapsed_seconds": elapsed,
            "peak_memory_reset": peak_reset,
            "memory_before": memory_before,
            "memory_after": _memory_snapshot(),
            "station_events": components.events,
            "timing": phase_snapshot,
            "reporter_events": reporter_events.events,
            "frame_stream": inspected.to_dict(),
            "vae_receipts": _json_value(inspected.receipts),
            "runtime_diagnostics": _json_value(runtime_diagnostics),
            "source_audio": source_audio,
            "media": media,
            "metadata": _json_value(output.metadata),
            "artifacts": {
                "video": str(video_path),
                "video_sha256": _sha256(video_path),
                "final_latent": str(latent_path),
                "final_latent_sha256": _sha256(latent_path),
                "text_conditioning": (
                    str(conditioning_path) if conditioning_path.is_file() else None
                ),
                "text_conditioning_sha256": (
                    _sha256(conditioning_path) if conditioning_path.is_file() else None
                ),
            },
        }
        receipt["cases"].append(case_receipt)
        _write_receipt(receipt_path, receipt)

    receipt["status"] = "passed"
    receipt["total_case_elapsed_seconds"] = sum(
        case["elapsed_seconds"] for case in receipt["cases"]
    )
    receipt["final_memory"] = _memory_snapshot()
    _write_receipt(receipt_path, receipt)
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--case",
        action="append",
        choices=_CASE_NAMES,
        dest="cases",
        help="Run only the selected case; repeat for more than one",
    )
    parser.add_argument("--expected-package-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    receipt = run_phase_g(
        options.output_dir,
        case_names=_CASE_NAMES if options.cases is None else options.cases,
        expected_package_root=options.expected_package_root,
    )
    sys.stdout.write(
        json.dumps({"receipt": str(receipt), "sha256": _sha256(receipt)}, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
