"""Standalone source- or installed-wheel Krea 2 VAE gate."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import runpy
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
from vae._fixtures import (
    ALTERNATE_VAE_NORMALIZED_RMSE_LIMIT,
    ALTERNATE_VAE_PIXEL_MAX_ABS_LIMIT,
    ALTERNATE_VAE_PIXEL_MEAN_ABS_LIMIT,
    RAW_REVISION,
    TILED_NORMALIZED_RMSE_LIMIT,
    TILED_PIXEL_MAX_ABS_LIMIT,
    TILED_PIXEL_MEAN_ABS_LIMIT,
    TILED_SEAM_MEAN_ABS_LIMIT,
    TILED_SEAM_TO_GLOBAL_MEAN_LIMIT,
    TILED_TO_FULL_PEAK_MEMORY_LIMIT,
    TURBO_REVISION,
    VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS,
    VAE_FIXTURE_SHA256,
    VAE_PIXEL_MAX_ABS_LIMIT,
    VAE_PIXEL_MEAN_ABS_LIMIT,
)

_BLOCKED = (
    "diffusers",
    "huggingface_hub",
    "numpy",
    "PIL",
    "tokenizers",
    "torch",
    "transformers",
)
_REVISION = re.compile(r"[0-9a-f]{40,64}")
_BOUNDARY_NAMES = tuple(
    name for name in VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS if name != "decoded_float"
)
_RESOURCE_SAMPLE_SIZE = 1024
_SEAM_HALF_WIDTH = 8


class _ImportBlocker:
    def __init__(self) -> None:
        self.attempted: set[str] = set()

    def find_spec(
        self,
        name: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _BLOCKED):
            self.attempted.add(name)
            raise ImportError(f"blocked development-only dependency: {name}")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_fixture_set(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected_digest in VAE_FIXTURE_SHA256.items():
        path = root / name
        digest = _sha256(path)
        if digest != expected_digest:
            raise AssertionError(f"Krea vae fixture digest changed: {name}")
        observed[name] = digest
    return observed


def _memory_snapshot(mx: object) -> tuple[dict[str, int | None], bool]:
    result: dict[str, int | None] = {}
    available = True
    for label, getter_name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        getter = getattr(mx, getter_name, None)
        if getter is None:
            result[label] = None
            available = False
            continue
        try:
            result[label] = max(0, int(getter()))
        except RuntimeError, TypeError, ValueError:
            result[label] = None
            available = False
    return result, available


def _reset_peak(mx: object) -> bool:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        return False
    try:
        reset()
    except RuntimeError, TypeError, ValueError:
        return False
    return True


def _metrics(mx: Any, candidate: Any, reference: Any) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise AssertionError(
            f"candidate shape/dtype {candidate.shape}/{candidate.dtype} differs from "
            f"reference {reference.shape}/{reference.dtype}"
        )
    mx.eval(candidate, reference)
    widened_candidate = candidate.astype(mx.float32)
    widened_reference = reference.astype(mx.float32)
    difference = widened_candidate - widened_reference
    rmse = math.sqrt(mx.mean(difference * difference).item())
    scale = math.sqrt(mx.mean(widened_reference * widened_reference).item())
    return {
        "shape": [int(dimension) for dimension in candidate.shape],
        "candidate_dtype": str(candidate.dtype).rsplit(".", maxsplit=1)[-1],
        "reference_dtype": str(reference.dtype).rsplit(".", maxsplit=1)[-1],
        "finite": bool(mx.all(mx.isfinite(candidate)).item()),
        "exact": bool(mx.array_equal(candidate, reference).item()),
        "max_abs": float(mx.max(mx.abs(difference)).item()),
        "mean_abs": float(mx.mean(mx.abs(difference)).item()),
        "normalized_rmse": rmse / max(scale, 1e-12),
    }


def _cross_dtype_metrics(mx: Any, candidate: Any, reference: Any) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"candidate shape {candidate.shape} differs from reference {reference.shape}"
        )
    mx.eval(candidate, reference)
    widened_candidate = candidate.astype(mx.float32)
    widened_reference = reference.astype(mx.float32)
    difference = widened_candidate - widened_reference
    rmse = math.sqrt(mx.mean(difference * difference).item())
    scale = math.sqrt(mx.mean(widened_reference * widened_reference).item())
    return {
        "shape": [int(dimension) for dimension in candidate.shape],
        "candidate_dtype": str(candidate.dtype).rsplit(".", maxsplit=1)[-1],
        "reference_dtype": str(reference.dtype).rsplit(".", maxsplit=1)[-1],
        "finite": bool(mx.all(mx.isfinite(candidate)).item()),
        "max_abs": float(mx.max(mx.abs(difference)).item()),
        "mean_abs": float(mx.mean(mx.abs(difference)).item()),
        "normalized_rmse": rmse / max(scale, 1e-12),
    }


def _require_metric(label: str, metric: dict[str, Any], limit: float) -> None:
    if not metric["finite"]:
        raise AssertionError(f"{label} contains nonfinite values")
    normalized_rmse = float(metric["normalized_rmse"])
    if normalized_rmse > limit:
        raise AssertionError(f"{label} normalized RMSE {normalized_rmse} exceeds {limit}")


def _pixel_metrics(mx: Any, candidate: Any, reference: Any) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise AssertionError(
            f"candidate pixels {candidate.shape}/{candidate.dtype} differ from "
            f"reference {reference.shape}/{reference.dtype}"
        )
    difference = candidate.astype(mx.int32) - reference.astype(mx.int32)
    absolute = mx.abs(difference)
    mx.eval(absolute)
    return {
        "shape": [int(dimension) for dimension in candidate.shape],
        "exact": bool(mx.array_equal(candidate, reference).item()),
        "max_abs": int(mx.max(absolute).item()),
        "mean_abs": float(mx.mean(absolute.astype(mx.float32)).item()),
        "nonzero": int(mx.sum(difference != 0).item()),
        "total": int(candidate.size),
    }


def _require_pixels(
    label: str,
    metric: dict[str, Any],
    *,
    max_abs_limit: int,
    mean_abs_limit: float,
) -> None:
    if int(metric["max_abs"]) > max_abs_limit:
        raise AssertionError(f"{label} maximum byte delta exceeds {max_abs_limit}")
    if float(metric["mean_abs"]) > mean_abs_limit:
        raise AssertionError(f"{label} mean byte delta exceeds {mean_abs_limit}")


def _seam_metrics(mx: Any, full: Any, tiled: Any, *, stride: int) -> dict[str, float]:
    error = mx.abs(full.astype(mx.float32) - tiled.astype(mx.float32))
    height, width = int(error.shape[1]), int(error.shape[2])
    bands = []
    for seam in range(stride, height, stride):
        bands.append(
            error[
                :,
                max(0, seam - _SEAM_HALF_WIDTH) : min(height, seam + _SEAM_HALF_WIDTH),
                :,
                :,
            ].reshape(-1)
        )
    for seam in range(stride, width, stride):
        bands.append(
            error[
                :,
                :,
                max(0, seam - _SEAM_HALF_WIDTH) : min(width, seam + _SEAM_HALF_WIDTH),
                :,
            ].reshape(-1)
        )
    if not bands:
        raise AssertionError("Krea tiled resource probe produced no seam bands")
    seam_error = mx.concatenate(bands)
    seam_mean = float(mx.mean(seam_error).item())
    global_mean = float(mx.mean(error).item())
    return {
        "band_half_width": float(_SEAM_HALF_WIDTH),
        "seam_mean_abs": seam_mean,
        "seam_max_abs": float(mx.max(seam_error).item()),
        "global_mean_abs": global_mean,
        "seam_to_global_mean": seam_mean / max(global_mean, 1e-12),
    }


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_vae(
    output_dir: Path,
    *,
    fixture_dir: Path,
    expected_package_root: Path,
    product_revision: str,
) -> Path:
    """Run the complete real saved-latent and installed-package VAE gate."""
    if _REVISION.fullmatch(product_revision) is None:
        raise ValueError("product revision must be a full hexadecimal Git revision")
    fixture_hashes = _validate_fixture_set(fixture_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)

    RecordingReporter = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_reporting.py"))[
        "RecordingReporter"
    ]

    import kinomlx.models.krea2 as krea2_package
    from kinomlx.models.krea2.components import Krea2Components
    from kinomlx.models.krea2.output import Krea2StillOutputPlan, Krea2StillOutputSink
    from kinomlx.models.krea2.resources import prepare_resources
    from kinomlx.models.krea2.settings import Krea2Settings
    from kinomlx.models.krea2.vae import (
        KREA2_VAE_CONSUMED_TENSORS,
        Krea2VAETilingConfig,
    )
    from kinomlx.reporting import TimingReporter
    from kinomlx.settings import Settings
    from kinomlx.videotoolbox.images import load_image_rgb

    package_file = Path(krea2_package.__file__).resolve()
    expected_package = expected_package_root.resolve()
    if not package_file.is_relative_to(expected_package):
        raise AssertionError(
            f"Krea vae imported {package_file}, expected package root {expected_package}"
        )

    infrastructure = Settings.from_env()
    turbo = prepare_resources(
        Krea2Settings(variant="turbo", transformer_dtype="bfloat16", vae_dtype="bfloat16"),
        infrastructure=infrastructure,
    )
    raw = prepare_resources(
        Krea2Settings(variant="raw", transformer_dtype="bfloat16", vae_dtype="bfloat16"),
        infrastructure=infrastructure,
    )
    if turbo.revision != TURBO_REVISION or raw.revision != RAW_REVISION:
        raise AssertionError("Krea vae resolved an unpinned Raw or Turbo revision")
    if turbo.vae_config != raw.vae_config:
        raise AssertionError("Raw and Turbo no longer declare the same VAE graph")
    if turbo.vae.source_fingerprint != raw.vae.source_fingerprint:
        raise AssertionError("Raw and Turbo no longer resolve the same VAE source")
    if turbo.vae.graph_fingerprint != raw.vae.graph_fingerprint:
        raise AssertionError("Raw and Turbo VAE graph identities differ")

    events = RecordingReporter()
    timing = TimingReporter(events)
    components = Krea2Components(reporter=timing)
    memory_before, before_available = _memory_snapshot(mx)
    load_started = time.perf_counter()
    lease = components.image_decoder(turbo)
    load_elapsed = time.perf_counter() - load_started
    memory_loaded, loaded_available = _memory_snapshot(mx)
    load_receipt = lease.value.load_receipt
    if load_receipt is None or load_receipt.target_count != KREA2_VAE_CONSUMED_TENSORS:
        raise AssertionError("Krea VAE lease has no complete 104-target load receipt")

    inputs = mx.load(str(fixture_dir / "denoise-trajectories.safetensors"), format="safetensors")
    references = mx.load(
        str(fixture_dir / "official-vae-boundaries.safetensors"),
        format="safetensors",
    )
    boundary_metrics: dict[str, dict[str, Any]] = {}

    def observer(name: str, value: Any) -> None:
        if name not in _BOUNDARY_NAMES:
            raise AssertionError(f"unexpected Krea VAE boundary: {name}")
        metric = _metrics(mx, value, references[name])
        _require_metric(name, metric, VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS[name])
        boundary_metrics[name] = metric

    output_path = output_dir / "saved-latent.png"
    output_plan = Krea2StillOutputPlan(output_path)
    decode_started = time.perf_counter()
    decoded = lease.value(inputs["official_final_unpacked"], observer=observer)
    mx.eval(decoded.pixels)
    decode_elapsed = time.perf_counter() - decode_started
    if set(boundary_metrics) != set(_BOUNDARY_NAMES):
        missing = sorted(set(_BOUNDARY_NAMES) - boundary_metrics.keys())
        extra = sorted(boundary_metrics.keys() - set(_BOUNDARY_NAMES))
        raise AssertionError(
            f"incomplete Krea VAE boundary capture: missing={missing}, extra={extra}"
        )
    decoder_product = decoded.pixels.transpose(0, 3, 1, 2)
    oracle_decoded_float = mx.clip(decoder_product, -1.0, 1.0)
    decoded_metric = _metrics(mx, oracle_decoded_float, references["decoded_float"])
    _require_metric(
        "decoded_float",
        decoded_metric,
        VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS["decoded_float"],
    )
    boundary_metrics["decoded_float"] = decoded_metric

    with output_plan.reserve() as reservation:
        encode_started = time.perf_counter()
        Krea2StillOutputSink(output_plan).write(decoded, reservation=reservation)
        encode_elapsed = time.perf_counter() - encode_started
    candidate_pixels = load_image_rgb(output_path)
    official_pixels = references["encoded_pixels"]
    official_png_pixels = load_image_rgb(fixture_dir / "official.png")
    if not mx.array_equal(official_png_pixels, official_pixels).item():
        raise AssertionError("official Krea PNG disagrees with its encoded-pixel sidecar")
    encoded_metric = _pixel_metrics(mx, candidate_pixels, official_pixels)
    _require_pixels(
        "saved-latent PNG",
        encoded_metric,
        max_abs_limit=VAE_PIXEL_MAX_ABS_LIMIT,
        mean_abs_limit=VAE_PIXEL_MEAN_ABS_LIMIT,
    )

    base_latent = mx.array(inputs["official_final_unpacked"])
    mx.eval(base_latent)
    inputs.clear()
    references.clear()
    del (
        candidate_pixels,
        decoded,
        decoder_product,
        official_pixels,
        official_png_pixels,
        oracle_decoded_float,
    )
    gc.collect()
    mx.clear_cache()

    latent_extent = _RESOURCE_SAMPLE_SIZE // turbo.vae_config.spatial_compression
    repeats = math.ceil(latent_extent / int(base_latent.shape[-1]))
    resource_latent = mx.tile(base_latent, (1, 1, repeats, repeats))[
        :, :, :latent_extent, :latent_extent
    ]
    mx.eval(resource_latent)

    full_peak_reset = _reset_peak(mx)
    full_memory_before, full_before_available = _memory_snapshot(mx)
    full_started = time.perf_counter()
    full_decoded = lease.value(resource_latent).pixels
    mx.eval(full_decoded)
    full_elapsed = time.perf_counter() - full_started
    full_memory_after, full_after_available = _memory_snapshot(mx)
    full_copy = mx.array(full_decoded)
    mx.eval(full_copy)
    del full_decoded
    gc.collect()
    mx.clear_cache()

    tiling = Krea2VAETilingConfig()
    tiled_peak_reset = _reset_peak(mx)
    tiled_memory_before, tiled_before_available = _memory_snapshot(mx)
    tiled_started = time.perf_counter()
    tiled_decoded = lease.value(resource_latent, tiling=tiling).pixels
    mx.eval(tiled_decoded)
    tiled_elapsed = time.perf_counter() - tiled_started
    tiled_memory_after, tiled_after_available = _memory_snapshot(mx)

    tiled_metric = _metrics(mx, tiled_decoded, full_copy)
    _require_metric("1024 tiled decode", tiled_metric, TILED_NORMALIZED_RMSE_LIMIT)
    full_pixels = (mx.clip(full_copy * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(mx.uint8)
    tiled_pixels = (mx.clip(tiled_decoded * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(mx.uint8)
    tiled_pixel_metric = _pixel_metrics(mx, tiled_pixels, full_pixels)
    _require_pixels(
        "1024 tiled decode",
        tiled_pixel_metric,
        max_abs_limit=TILED_PIXEL_MAX_ABS_LIMIT,
        mean_abs_limit=TILED_PIXEL_MEAN_ABS_LIMIT,
    )
    seam_metric = _seam_metrics(
        mx,
        full_copy,
        tiled_decoded,
        stride=tiling.tile_sample_stride,
    )
    if seam_metric["seam_mean_abs"] > TILED_SEAM_MEAN_ABS_LIMIT:
        raise AssertionError("Krea tiled seam-band mean exceeds its declared contract")
    if seam_metric["seam_to_global_mean"] > TILED_SEAM_TO_GLOBAL_MEAN_LIMIT:
        raise AssertionError("Krea tiled seam bands amplify error beyond their declared contract")

    memory_available = all(
        (
            before_available,
            loaded_available,
            full_before_available,
            full_after_available,
            tiled_before_available,
            tiled_after_available,
        )
    )
    if not memory_available or not full_peak_reset or not tiled_peak_reset:
        raise AssertionError("MLX memory probes are unavailable; refusing an unmeasured VAE gate")
    full_peak = full_memory_after["peak_bytes"]
    tiled_peak = tiled_memory_after["peak_bytes"]
    if not isinstance(full_peak, int) or not isinstance(tiled_peak, int) or full_peak <= 0:
        raise AssertionError("MLX VAE memory probes returned invalid peak values")
    peak_ratio = tiled_peak / full_peak
    if peak_ratio >= TILED_TO_FULL_PEAK_MEMORY_LIMIT:
        raise AssertionError(
            f"Krea tiled peak ratio {peak_ratio} must be below {TILED_TO_FULL_PEAK_MEMORY_LIMIT}"
        )

    del base_latent, full_copy, full_pixels, resource_latent, tiled_decoded, tiled_pixels
    gc.collect()
    mx.clear_cache()
    lease.close()
    del lease
    gc.collect()
    mx.clear_cache()
    memory_after, after_available = _memory_snapshot(mx)
    if not after_available:
        raise AssertionError("MLX memory probes disappeared after Krea VAE release")
    after_active = memory_after["active_bytes"]
    if not isinstance(after_active, int) or after_active > 1024**2:
        raise AssertionError("Krea VAE lease retained more than 1 MiB after release")
    if memory_after["cache_bytes"] != 0:
        raise AssertionError("Krea VAE lease left MLX cache memory resident")

    dtype_inputs = mx.load(
        str(fixture_dir / "denoise-trajectories.safetensors"),
        format="safetensors",
    )
    dtype_references = mx.load(
        str(fixture_dir / "official-vae-boundaries.safetensors"),
        format="safetensors",
    )
    alternate_dtypes: dict[str, dict[str, object]] = {}
    for dtype in ("float16", "float32"):
        prepare_started = time.perf_counter()
        dtype_resources = prepare_resources(
            Krea2Settings(
                variant="turbo",
                transformer_dtype="bfloat16",
                vae_dtype=dtype,
            ),
            infrastructure=infrastructure,
        )
        prepare_elapsed = time.perf_counter() - prepare_started
        dtype_load_started = time.perf_counter()
        dtype_lease = components.image_decoder(dtype_resources)
        dtype_load_elapsed = time.perf_counter() - dtype_load_started
        dtype_receipt = dtype_lease.value.load_receipt
        if dtype_receipt is None or dtype_receipt.target_count != KREA2_VAE_CONSUMED_TENSORS:
            raise AssertionError(f"Krea {dtype} VAE lease has no complete load receipt")
        try:
            dtype_decode_started = time.perf_counter()
            dtype_decoded = dtype_lease.value(dtype_inputs["official_final_unpacked"])
            mx.eval(dtype_decoded.pixels)
            dtype_decode_elapsed = time.perf_counter() - dtype_decode_started
            observed_dtype = str(dtype_decoded.pixels.dtype).rsplit(".", maxsplit=1)[-1]
            if observed_dtype != dtype:
                raise AssertionError(f"Krea {dtype} VAE produced {observed_dtype}")
            dtype_float = mx.clip(
                dtype_decoded.pixels.transpose(0, 3, 1, 2),
                -1.0,
                1.0,
            )
            dtype_metric = _cross_dtype_metrics(
                mx,
                dtype_float,
                dtype_references["decoded_float"],
            )
            _require_metric(
                f"{dtype} decoded_float",
                dtype_metric,
                ALTERNATE_VAE_NORMALIZED_RMSE_LIMIT,
            )
            dtype_output = output_dir / f"saved-latent-{dtype}.png"
            dtype_plan = Krea2StillOutputPlan(dtype_output)
            dtype_encode_started = time.perf_counter()
            Krea2StillOutputSink(dtype_plan).write(dtype_decoded)
            dtype_encode_elapsed = time.perf_counter() - dtype_encode_started
            dtype_pixels = load_image_rgb(dtype_output)
            dtype_pixel_metric = _pixel_metrics(
                mx,
                dtype_pixels,
                dtype_references["encoded_pixels"],
            )
            _require_pixels(
                f"{dtype} saved-latent PNG",
                dtype_pixel_metric,
                max_abs_limit=ALTERNATE_VAE_PIXEL_MAX_ABS_LIMIT,
                mean_abs_limit=ALTERNATE_VAE_PIXEL_MEAN_ABS_LIMIT,
            )
            alternate_dtypes[dtype] = {
                "prepare_seconds": prepare_elapsed,
                "load_seconds": dtype_load_elapsed,
                "decode_seconds": dtype_decode_elapsed,
                "encode_seconds": dtype_encode_elapsed,
                "source": str(dtype_resources.vae.source_path.resolve()),
                "source_files": list(dtype_receipt.source_files),
                "floating": dtype_metric,
                "encoded_pixels": dtype_pixel_metric,
                "image": {
                    "path": str(dtype_output),
                    "sha256": _sha256(dtype_output),
                    "bytes": dtype_output.stat().st_size,
                },
            }
            del dtype_decoded, dtype_float, dtype_pixels
        finally:
            dtype_lease.close()
            del dtype_lease
            gc.collect()
            mx.clear_cache()
        dtype_memory, dtype_memory_available = _memory_snapshot(mx)
        if not dtype_memory_available:
            raise AssertionError(f"MLX memory probes disappeared after Krea {dtype} release")
        dtype_active = dtype_memory["active_bytes"]
        if not isinstance(dtype_active, int) or dtype_active > 1024**2:
            raise AssertionError(f"Krea {dtype} VAE retained more than 1 MiB after release")
        if dtype_memory["cache_bytes"] != 0:
            raise AssertionError(f"Krea {dtype} VAE left MLX cache memory resident")
        alternate_dtypes[dtype]["memory_after_release"] = dtype_memory
    dtype_inputs.clear()
    dtype_references.clear()
    gc.collect()
    mx.clear_cache()
    memory_after_alternates, alternates_after_available = _memory_snapshot(mx)
    if not alternates_after_available:
        raise AssertionError("MLX memory probes disappeared after alternate Krea dtype gates")
    if memory_after_alternates["active_bytes"] != 0:
        raise AssertionError("alternate Krea VAE dtype gates retained active MLX memory")
    if memory_after_alternates["cache_bytes"] != 0:
        raise AssertionError("alternate Krea VAE dtype gates retained MLX cache memory")

    offenders = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _BLOCKED)
    )
    if offenders:
        raise AssertionError(f"development-only modules were imported: {offenders}")
    if blocker.attempted:
        raise AssertionError(
            f"development-only imports were attempted and caught: {sorted(blocker.attempted)}"
        )

    receipt = {
        "schema_version": 1,
        "status": "pass",
        "package": {
            "version": importlib.metadata.version("kinomlx"),
            "revision": product_revision,
            "module_path": str(package_file),
            "expected_package_root": str(expected_package),
        },
        "fixtures": fixture_hashes,
        "resources": {
            "target_count": load_receipt.target_count,
            "target_digest": load_receipt.target_digest,
            "source_files": list(load_receipt.source_files),
            "shared_source": True,
            "shared_graph": True,
            "shared_object": True,
            "graph_fingerprint": turbo.vae.graph_fingerprint,
            "dtype": turbo.vae.dtype_policy,
            "turbo": {
                "revision": turbo.revision,
                "source_fingerprint": turbo.vae.source_fingerprint,
                "source": str(turbo.vae.source_path),
            },
            "raw": {
                "revision": raw.revision,
                "source_fingerprint": raw.vae.source_fingerprint,
                "source": str(raw.vae.source_path),
            },
        },
        "saved_latent": {
            "decode_seconds": decode_elapsed,
            "encode_seconds": encode_elapsed,
            "boundaries": boundary_metrics,
            "encoded_pixels": encoded_metric,
            "image": {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
            },
        },
        "alternate_dtypes": alternate_dtypes,
        "tiling": {
            "sample_size": _RESOURCE_SAMPLE_SIZE,
            "tile_sample_size": tiling.tile_sample_size,
            "tile_sample_stride": tiling.tile_sample_stride,
            "full_seconds": full_elapsed,
            "tiled_seconds": tiled_elapsed,
            "floating": tiled_metric,
            "encoded_pixels": tiled_pixel_metric,
            "seams": seam_metric,
            "full_memory_before": full_memory_before,
            "full_memory_after": full_memory_after,
            "tiled_memory_before": tiled_memory_before,
            "tiled_memory_after": tiled_memory_after,
            "tiled_to_full_peak_ratio": peak_ratio,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "mlx_version": importlib.metadata.version("mlx"),
            "load_seconds": load_elapsed,
            "memory_before": memory_before,
            "memory_loaded": memory_loaded,
            "memory_after_release": memory_after,
            "memory_after_alternate_dtypes": memory_after_alternates,
            "timing": timing.to_dict(),
            "blocked_import_policy": list(_BLOCKED),
            "blocked_import_attempts": sorted(blocker.attempted),
            "loaded_blocked_modules": offenders,
        },
    }
    receipt_path = output_dir / "vae-vae-receipt.json"
    _write_receipt(receipt_path, receipt)
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--product-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    receipt = run_vae(
        options.output_dir,
        fixture_dir=options.fixture_dir,
        expected_package_root=options.expected_package_root,
        product_revision=options.product_revision,
    )
    sys.stdout.write(str(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
