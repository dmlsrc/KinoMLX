"""Standalone source- or installed-wheel Krea 2 generation gate."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
from _generation_fixtures import (
    DECODED_PIXEL_MAX_ABS_LIMIT,
    DECODED_PIXEL_MEAN_ABS_LIMIT,
    FIXTURE_SHA256,
    MODEL_OUTPUT_NORMALIZED_RMSE_LIMIT,
    ONE_STEP_STATE_NORMALIZED_RMSE_LIMIT,
    RAW_REVISION,
    SCHEDULE_MAX_ABS_LIMIT,
    TURBO_REVISION,
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


def _fixture_receipt(roots: dict[str, Path]) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for variant, expected in FIXTURE_SHA256.items():
        root = roots[variant]
        variant_receipt: dict[str, str] = {}
        for name, digest in expected.items():
            actual = _sha256(root / name)
            if actual != digest:
                raise AssertionError(f"Krea generation fixture digest changed: {variant}/{name}")
            variant_receipt[name] = actual
        observed[variant] = variant_receipt
    return observed


def _metrics(mx: Any, candidate: Any, reference: Any) -> dict[str, object]:
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"candidate shape {candidate.shape} differs from reference {reference.shape}"
        )
    mx.eval(candidate, reference)
    left = candidate.astype(mx.float32)
    right = reference.astype(mx.float32)
    difference = left - right
    rmse = math.sqrt(mx.mean(difference * difference).item())
    scale = math.sqrt(mx.mean(right * right).item())
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


def _require_metric(label: str, metric: dict[str, object], limit: float) -> None:
    if metric["finite"] is not True:
        raise AssertionError(f"{label} contains nonfinite values")
    observed = float(metric["normalized_rmse"])
    if observed > limit:
        raise AssertionError(f"{label} normalized RMSE {observed} exceeds {limit}")


def _pixel_metrics(mx: Any, candidate: Any, reference: Any) -> dict[str, object]:
    difference = candidate.astype(mx.int32) - reference.astype(mx.int32)
    absolute = mx.abs(difference)
    mx.eval(absolute)
    return {
        "shape": [int(dimension) for dimension in candidate.shape],
        "exact": bool(mx.array_equal(candidate, reference).item()),
        "max_abs": int(mx.max(absolute).item()),
        "mean_abs": float(mx.mean(absolute.astype(mx.float32)).item()),
    }


def _memory_snapshot(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _prepare_fixture_branch(
    mx: Any,
    inputs: dict[str, Any],
    geometry: Any,
    *,
    prefix: str,
) -> Any:
    from kinomlx.models.krea2.sampling import prepare_branch
    from kinomlx.models.krea2.text_conditioning import Krea2PromptConditioning

    context = inputs[f"{prefix}_context"]
    mask = inputs[f"{prefix}_text_mask"]
    conditioning = Krea2PromptConditioning(
        hidden_states=context,
        attention_mask=mask,
        position_ids=mx.zeros((3, context.shape[0], context.shape[1]), dtype=mx.int32),
        prompts=(prefix,),
        token_identity=f"oracle-{prefix}",
    )
    return prepare_branch(
        inputs["initial_packed"],
        conditioning,
        geometry,
        activation_dtype=mx.bfloat16,
        reference_denoise=True,
    )


def _trajectory_samples(
    mx: Any,
    resources: Any,
    fixture_root: Path,
    components: Any,
) -> dict[str, object]:
    from kinomlx.models.krea2.sampling import (
        krea_cfg,
        pack_latents,
        resolve_geometry,
        resolve_recipe,
        timestep_schedule,
    )
    from kinomlx.models.krea2.types import Krea2Request
    from kinomlx.samplers.steps import euler_velocity_step

    inputs = mx.load(str(fixture_root / "image-input.safetensors"), format="safetensors")
    reference = mx.load(
        str(fixture_root / "denoise-trajectories.safetensors"),
        format="safetensors",
    )
    request = Krea2Request("oracle fixture", width=192, height=192)
    geometry = resolve_geometry(
        request,
        spatial_compression=resources.capabilities.spatial_compression,
        patch_size=resources.capabilities.transformer_patch_size,
    )
    recipe = resolve_recipe(
        resources.variant,
        request,
        geometry,
        resources.scheduler_config,
    )
    schedule = timestep_schedule(recipe)
    schedule_metric = _metrics(mx, schedule, inputs["official_schedule"])
    if float(schedule_metric["max_abs"]) > SCHEDULE_MAX_ABS_LIMIT:
        raise AssertionError(
            f"{resources.variant.value} schedule maximum error exceeds {SCHEDULE_MAX_ABS_LIMIT}"
        )
    if not mx.array_equal(
        pack_latents(
            inputs["noise_unpacked"],
            patch_size=resources.capabilities.transformer_patch_size,
        ),
        inputs["initial_packed"],
    ).item():
        raise AssertionError(f"{resources.variant.value} latent packing differs from oracle")

    positive = _prepare_fixture_branch(mx, inputs, geometry, prefix="positive")
    negative = (
        _prepare_fixture_branch(mx, inputs, geometry, prefix="negative")
        if recipe.cfg_enabled
        else None
    )
    position_metric = _metrics(
        mx,
        positive.position_ids.astype(mx.float32),
        inputs["official_position_ids"],
    )
    if position_metric["exact"] is not True:
        raise AssertionError(f"{resources.variant.value} image positions differ from oracle")
    combined_mask = mx.concatenate(
        [
            inputs["positive_text_mask"],
            mx.ones((1, geometry.image_sequence_length), dtype=mx.bool_),
        ],
        axis=1,
    )
    if not mx.array_equal(positive.attention_mask, combined_mask).item():
        raise AssertionError(f"{resources.variant.value} combined mask differs from oracle")

    selected_indices = tuple(sorted({0, recipe.steps // 2, recipe.steps - 1}))
    samples: list[dict[str, object]] = []
    forward_started = time.perf_counter()
    transformer_lease = components.transformer(resources)
    try:
        for index in selected_indices:
            state = reference["official_trajectory"][index]
            timestep = mx.broadcast_to(schedule[index].astype(state.dtype), (state.shape[0],))
            conditional = transformer_lease.value(
                state,
                encoder_hidden_states=positive.encoder_hidden_states,
                timestep=timestep,
                position_ids=positive.position_ids,
                attention_mask=positive.attention_mask,
            ).sample
            reference_offset = index if negative is None else index * 2
            conditional_metric = _metrics(
                mx,
                conditional,
                reference["official_model_outputs"][reference_offset],
            )
            _require_metric(
                f"{resources.variant.value} conditional step {index}",
                conditional_metric,
                MODEL_OUTPUT_NORMALIZED_RMSE_LIMIT,
            )
            unconditional_metric = None
            if negative is None:
                velocity = conditional
            else:
                unconditional = transformer_lease.value(
                    state,
                    encoder_hidden_states=negative.encoder_hidden_states,
                    timestep=timestep,
                    position_ids=negative.position_ids,
                    attention_mask=negative.attention_mask,
                ).sample
                unconditional_metric = _metrics(
                    mx,
                    unconditional,
                    reference["official_model_outputs"][reference_offset + 1],
                )
                _require_metric(
                    f"{resources.variant.value} unconditional step {index}",
                    unconditional_metric,
                    MODEL_OUTPUT_NORMALIZED_RMSE_LIMIT,
                )
                velocity = krea_cfg(conditional, unconditional, recipe.guidance)
            next_state = euler_velocity_step(
                state,
                velocity,
                float(schedule[index].item()),
                float(schedule[index + 1].item()),
            )
            state_metric = _metrics(
                mx,
                next_state,
                reference["official_trajectory"][index + 1],
            )
            _require_metric(
                f"{resources.variant.value} one-step state {index}",
                state_metric,
                ONE_STEP_STATE_NORMALIZED_RMSE_LIMIT,
            )
            samples.append(
                {
                    "index": index,
                    "timestep": float(timestep[0].item()),
                    "conditional": conditional_metric,
                    "unconditional": unconditional_metric,
                    "one_step_state": state_metric,
                }
            )
    finally:
        transformer_lease.close()
    forward_elapsed = time.perf_counter() - forward_started
    final_latent = mx.array(reference["official_final_unpacked"])
    mx.eval(final_latent)
    inputs.clear()
    reference.clear()
    return {
        "recipe": recipe.to_metadata(),
        "geometry": geometry.to_metadata(),
        "schedule": schedule_metric,
        "positions": position_metric,
        "selected_indices": list(selected_indices),
        "samples": samples,
        "forward_seconds": forward_elapsed,
        "official_final_latent": final_latent,
    }


def run_generation(
    output_dir: Path,
    *,
    turbo_fixture_dir: Path,
    raw_fixture_dir: Path,
    expected_package_root: Path,
    product_revision: str,
) -> Path:
    """Run sampled Raw/Turbo oracle trajectories and final decode through public ports."""
    if _REVISION.fullmatch(product_revision) is None:
        raise ValueError("product revision must be a full hexadecimal Git revision")
    roots = {
        "turbo": turbo_fixture_dir.expanduser().absolute(),
        "raw": raw_fixture_dir.expanduser().absolute(),
    }
    fixture_receipt = _fixture_receipt(roots)
    output_dir.mkdir(parents=True, exist_ok=False)

    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)

    import kinomlx.models.krea2 as krea2_package
    from kinomlx.models.krea2.components import NativeKrea2Components
    from kinomlx.models.krea2.output import Krea2StillOutputPlan, Krea2StillOutputSink
    from kinomlx.models.krea2.resources import prepare_resources
    from kinomlx.models.krea2.settings import Krea2Settings
    from kinomlx.settings import Settings
    from kinomlx.videotoolbox.images import load_image_rgb

    package_file = Path(krea2_package.__file__).resolve()
    expected_package = expected_package_root.resolve()
    if not package_file.is_relative_to(expected_package):
        raise AssertionError(
            f"Krea generation imported {package_file}, expected package root {expected_package}"
        )

    infrastructure = Settings.from_env()
    resources = {
        variant: prepare_resources(
            Krea2Settings(
                variant=variant,
                transformer_dtype="bfloat16",
                reference_denoise=True,
                vae_dtype="bfloat16",
            ),
            infrastructure=infrastructure,
        )
        for variant in ("turbo", "raw")
    }
    if resources["turbo"].revision != TURBO_REVISION:
        raise AssertionError("Krea generation resolved an unpinned Turbo revision")
    if resources["raw"].revision != RAW_REVISION:
        raise AssertionError("Krea generation resolved an unpinned Raw revision")

    components = NativeKrea2Components()
    memory_before = _memory_snapshot(mx)
    trajectories: dict[str, dict[str, object]] = {}
    final_latents: dict[str, Any] = {}
    for variant in ("turbo", "raw"):
        result = _trajectory_samples(
            mx,
            resources[variant],
            roots[variant],
            components,
        )
        final_latents[variant] = result.pop("official_final_latent")
        trajectories[variant] = result
        gc.collect()
        mx.clear_cache()

    decoded: dict[str, dict[str, object]] = {}
    decoder_lease = components.image_decoder(resources["turbo"])
    try:
        for variant in ("turbo", "raw"):
            output_path = output_dir / f"{variant}-oracle-final.png"
            product = decoder_lease.value(final_latents[variant])
            Krea2StillOutputSink(Krea2StillOutputPlan(output_path)).write(product)
            candidate = load_image_rgb(output_path)
            reference = load_image_rgb(roots[variant] / "official.png")
            metric = _pixel_metrics(mx, candidate, reference)
            if int(metric["max_abs"]) > DECODED_PIXEL_MAX_ABS_LIMIT:
                raise AssertionError(f"{variant} decoded maximum byte error exceeds limit")
            if float(metric["mean_abs"]) > DECODED_PIXEL_MEAN_ABS_LIMIT:
                raise AssertionError(f"{variant} decoded mean byte error exceeds limit")
            decoded[variant] = {
                "pixels": metric,
                "path": output_path.name,
                "sha256": _sha256(output_path),
                "bytes": output_path.stat().st_size,
            }
    finally:
        decoder_lease.close()
    del product, candidate, reference
    final_latents.clear()
    gc.collect()
    mx.clear_cache()
    memory_after = _memory_snapshot(mx)
    if memory_after["active_bytes"] > 1024**2 or memory_after["cache_bytes"] != 0:
        raise AssertionError(
            f"Krea generation retained MLX state after component release: {memory_after}"
        )

    loaded_blocked = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _BLOCKED)
    )
    if blocker.attempted or loaded_blocked:
        raise AssertionError(
            f"Krea generation used blocked dependencies: attempted={sorted(blocker.attempted)}, "
            f"loaded={loaded_blocked}"
        )

    receipt = {
        "schema_version": 1,
        "status": "pass",
        "package": {
            "module_path": str(package_file),
            "expected_package_root": str(expected_package),
            "version": importlib.metadata.version("kinomlx"),
            "revision": product_revision,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "mlx_version": importlib.metadata.version("mlx"),
            "blocked_import_policy": list(_BLOCKED),
            "blocked_import_attempts": sorted(blocker.attempted),
            "loaded_blocked_modules": loaded_blocked,
            "memory_before": memory_before,
            "memory_after": memory_after,
        },
        "fixtures": fixture_receipt,
        "resources": {
            variant: {
                "revision": selected.revision,
                "recipe_identity": selected.recipe_identity,
                "source_root": str(selected.source_root),
            }
            for variant, selected in resources.items()
        },
        "trajectories": trajectories,
        "decoded": decoded,
    }
    receipt_path = output_dir / "generation-composition-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turbo-fixture-dir", type=Path, required=True)
    parser.add_argument("--raw-fixture-dir", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--product-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    receipt = run_generation(
        options.output_dir,
        turbo_fixture_dir=options.turbo_fixture_dir,
        raw_fixture_dir=options.raw_fixture_dir,
        expected_package_root=options.expected_package_root,
        product_revision=options.product_revision,
    )
    sys.stdout.write(str(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
