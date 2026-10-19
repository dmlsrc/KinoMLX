"""Standalone source- or installed-wheel Krea 2 transformer gate."""

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
from transformer._fixtures import (
    BLOCK_NORMALIZED_RMSE_LIMIT,
    BOUNDARY_NORMALIZED_RMSE_LIMITS,
    RAW_REVISION,
    TEACHER_FINAL_STATE_LIMIT,
    TEACHER_OUTPUT_AGGREGATE_LIMIT,
    TEACHER_OUTPUT_STEP_LIMIT,
    TEACHER_STATE_STEP_LIMIT,
    TEACHER_TRAJECTORY_AGGREGATE_LIMIT,
    TRAJECTORY_FIXTURE_SHA256,
    TRANSFORMER_FIXTURE_SHA256,
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
_STAGES = ("requested", "shape-padding-control")
_STAGE_GEOMETRY = {
    "requested": {
        "combined_sequence_length": 656,
        "padding_tokens": 112,
        "padded_sequence_length": 768,
    },
    "shape-padding-control": {
        "combined_sequence_length": 768,
        "padding_tokens": 0,
        "padded_sequence_length": 768,
    },
}


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


def _validate_fixture_set(root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected_digest in expected.items():
        path = root / name
        digest = _sha256(path)
        if digest != expected_digest:
            raise AssertionError(f"Krea transformer fixture digest changed: {name}")
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
        "normalized_rmse": rmse / max(scale, 1e-12),
    }


def _require_metric(label: str, metric: dict[str, Any], limit: float) -> None:
    if not metric["finite"]:
        raise AssertionError(f"{label} contains nonfinite values")
    normalized_rmse = float(metric["normalized_rmse"])
    if normalized_rmse > limit:
        raise AssertionError(f"{label} normalized RMSE {normalized_rmse} exceeds {limit}")


def _run_stage(
    mx: Any,
    model: Any,
    fixture_dir: Path,
    stage: str,
) -> dict[str, object]:
    inputs = mx.load(
        str(fixture_dir / f"transformer-input-{stage}.safetensors"),
        format="safetensors",
    )
    references = mx.load(
        str(fixture_dir / f"official-transformer-boundaries-{stage}.safetensors"),
        format="safetensors",
    )
    combined_length = int(inputs["combined_attention_mask"].shape[1])
    text_length = int(inputs["context"].shape[1])
    valid_combined = [int(index) for index in inputs["valid_combined_indices"].tolist()]
    valid_text = [index for index in valid_combined if index < text_length]
    comparisons: dict[str, dict[str, Any]] = {}

    def observer(name: str, value: Any) -> None:
        reference_name = name
        candidate = value
        if name in {"text_fusion", "text_projection"}:
            reference_name = f"{name}_valid"
            candidate = value[:, valid_text]
        elif name == "combined_input" or name.startswith("block_"):
            reference_name = f"{name}_valid"
            candidate = value[:, valid_combined]
        elif name in {"rotary_cos", "rotary_sin"}:
            candidate = value[:, :combined_length]
        reference = references[reference_name]
        metric = _metrics(mx, candidate, reference)
        limit = (
            BLOCK_NORMALIZED_RMSE_LIMIT
            if name.startswith("block_")
            else BOUNDARY_NORMALIZED_RMSE_LIMITS[name]
        )
        _require_metric(f"{stage}:{name}", metric, limit)
        comparisons[name] = metric

    started = time.perf_counter()
    output = model(
        inputs["packed_latents"],
        encoder_hidden_states=inputs["context"],
        timestep=inputs["timestep"],
        position_ids=inputs["official_position_ids"],
        attention_mask=inputs["combined_attention_mask"],
        observer=observer,
    )
    mx.eval(output.sample)
    elapsed = time.perf_counter() - started
    expected_names = set(BOUNDARY_NORMALIZED_RMSE_LIMITS)
    expected_names.update(f"block_{index:02d}" for index in range(28))
    if set(comparisons) != expected_names:
        missing = sorted(expected_names - comparisons.keys())
        extra = sorted(comparisons.keys() - expected_names)
        raise AssertionError(f"incomplete Krea boundary capture: missing={missing}, extra={extra}")
    geometry = {
        "combined_sequence_length": output.combined_sequence_length,
        "padding_tokens": output.padding_tokens,
        "padded_sequence_length": output.padded_sequence_length,
    }
    if geometry != _STAGE_GEOMETRY[stage]:
        raise AssertionError(f"{stage} resolved unexpected sequence geometry: {geometry}")
    inputs.clear()
    references.clear()
    return {
        "elapsed_seconds": elapsed,
        "sequence": geometry,
        "boundaries": comparisons,
    }


def _euler_step(mx: Any, state: Any, velocity: Any, delta: float) -> Any:
    delta_f32 = mx.array(delta, dtype=mx.float32)
    scaled_velocity = (delta_f32 * velocity.astype(mx.float32)).astype(velocity.dtype)
    return state + scaled_velocity


def _run_trajectory(
    mx: Any,
    model: Any,
    fixture_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    inputs = mx.load(str(fixture_dir / "image-input.safetensors"), format="safetensors")
    reference = mx.load(
        str(fixture_dir / "denoise-trajectories.safetensors"),
        format="safetensors",
    )
    schedule = inputs["official_schedule"]
    text_mask = inputs["positive_text_mask"]
    image_mask = mx.ones(
        (text_mask.shape[0], inputs["initial_packed"].shape[1]),
        dtype=mx.bool_,
    )
    combined_mask = mx.concatenate([text_mask, image_mask], axis=1)

    teacher_outputs = []
    teacher_states = [inputs["initial_packed"]]
    teacher_steps = []
    teacher_started = time.perf_counter()
    for index in range(schedule.shape[0] - 1):
        state = reference["official_trajectory"][index]
        timestep = mx.broadcast_to(schedule[index].astype(state.dtype), (state.shape[0],))
        velocity = model(
            state,
            encoder_hidden_states=inputs["positive_context"],
            timestep=timestep,
            position_ids=inputs["official_position_ids"],
            attention_mask=combined_mask,
        ).sample
        delta = float(schedule[index + 1].item() - schedule[index].item())
        next_state = _euler_step(mx, state, velocity, delta)
        mx.eval(velocity, next_state)
        output_metric = _metrics(mx, velocity, reference["official_model_outputs"][index])
        state_metric = _metrics(mx, next_state, reference["official_trajectory"][index + 1])
        _require_metric(
            f"teacher-forced model output step {index}",
            output_metric,
            TEACHER_OUTPUT_STEP_LIMIT,
        )
        _require_metric(
            f"teacher-forced state step {index}",
            state_metric,
            TEACHER_STATE_STEP_LIMIT,
        )
        teacher_outputs.append(velocity)
        teacher_states.append(next_state)
        teacher_steps.append(
            {
                "index": index,
                "timestep": float(timestep[0].item()),
                "model_output": output_metric,
                "one_step_state": state_metric,
            }
        )
    teacher_elapsed = time.perf_counter() - teacher_started
    teacher_outputs_array = mx.stack(teacher_outputs, axis=0)
    teacher_states_array = mx.stack(teacher_states, axis=0)
    teacher_output_metric = _metrics(
        mx,
        teacher_outputs_array,
        reference["official_model_outputs"],
    )
    teacher_trajectory_metric = _metrics(
        mx,
        teacher_states_array,
        reference["official_trajectory"],
    )
    teacher_final_metric = _metrics(
        mx,
        teacher_states_array[-1],
        reference["official_trajectory"][-1],
    )
    _require_metric(
        "teacher-forced aggregate model outputs",
        teacher_output_metric,
        TEACHER_OUTPUT_AGGREGATE_LIMIT,
    )
    _require_metric(
        "teacher-forced aggregate one-step states",
        teacher_trajectory_metric,
        TEACHER_TRAJECTORY_AGGREGATE_LIMIT,
    )
    _require_metric(
        "teacher-forced final one-step state",
        teacher_final_metric,
        TEACHER_FINAL_STATE_LIMIT,
    )

    closed_outputs = []
    closed_states = [inputs["initial_packed"]]
    current = inputs["initial_packed"]
    closed_started = time.perf_counter()
    for index in range(schedule.shape[0] - 1):
        timestep = mx.broadcast_to(schedule[index].astype(current.dtype), (current.shape[0],))
        velocity = model(
            current,
            encoder_hidden_states=inputs["positive_context"],
            timestep=timestep,
            position_ids=inputs["official_position_ids"],
            attention_mask=combined_mask,
        ).sample
        delta = float(schedule[index + 1].item() - schedule[index].item())
        current = _euler_step(mx, current, velocity, delta)
        mx.eval(velocity, current)
        closed_outputs.append(velocity)
        closed_states.append(current)
    closed_elapsed = time.perf_counter() - closed_started
    closed_outputs_array = mx.stack(closed_outputs, axis=0)
    closed_states_array = mx.stack(closed_states, axis=0)
    mx.eval(closed_outputs_array, closed_states_array)
    closed_output_metric = _metrics(
        mx,
        closed_outputs_array,
        reference["official_model_outputs"],
    )
    closed_trajectory_metric = _metrics(
        mx,
        closed_states_array,
        reference["official_trajectory"],
    )
    closed_final_metric = _metrics(
        mx,
        closed_states_array[-1],
        reference["official_trajectory"][-1],
    )
    for label, metric in (
        ("closed-loop outputs", closed_output_metric),
        ("closed-loop trajectory", closed_trajectory_metric),
        ("closed-loop final state", closed_final_metric),
    ):
        if not metric["finite"]:
            raise AssertionError(f"{label} contains nonfinite values")

    stock = {
        "model_outputs": _metrics(
            mx,
            reference["diffusers_model_outputs"],
            reference["official_model_outputs"],
        ),
        "trajectory": _metrics(
            mx,
            reference["diffusers_trajectory"],
            reference["official_trajectory"],
        ),
        "final_state": _metrics(
            mx,
            reference["diffusers_trajectory"][-1],
            reference["official_trajectory"][-1],
        ),
    }
    sidecar = output_dir / "transformer-trajectories.safetensors"
    mx.save_safetensors(
        str(sidecar),
        {
            "teacher_forced_model_outputs": teacher_outputs_array,
            "teacher_forced_one_step_states": teacher_states_array,
            "closed_loop_model_outputs": closed_outputs_array,
            "closed_loop_trajectory": closed_states_array,
            "official_schedule": schedule,
        },
        metadata={
            "schema": "1",
            "teacher_forced_gate": "true",
            "closed_loop_gate": "finite-only-characterization",
        },
    )
    sidecar_record = {
        "path": sidecar.name,
        "sha256": _sha256(sidecar),
        "bytes": sidecar.stat().st_size,
    }
    inputs.clear()
    reference.clear()
    return {
        "euler": "FP32 scalar multiply rounded to BF16, then BF16 add",
        "teacher_forced": {
            "status": "pass",
            "elapsed_seconds": teacher_elapsed,
            "model_outputs": teacher_output_metric,
            "one_step_states": teacher_trajectory_metric,
            "final_one_step_state": teacher_final_metric,
            "steps": teacher_steps,
            "limits": {
                "output_step": TEACHER_OUTPUT_STEP_LIMIT,
                "output_aggregate": TEACHER_OUTPUT_AGGREGATE_LIMIT,
                "state_step": TEACHER_STATE_STEP_LIMIT,
                "state_aggregate": TEACHER_TRAJECTORY_AGGREGATE_LIMIT,
                "final_state": TEACHER_FINAL_STATE_LIMIT,
            },
        },
        "closed_loop": {
            "status": "finite-only-characterization",
            "reason": (
                "retained to expose backend-sensitive feedback drift; teacher-forced "
                "one-step states are the numerical correctness gate"
            ),
            "elapsed_seconds": closed_elapsed,
            "model_outputs": closed_output_metric,
            "trajectory": closed_trajectory_metric,
            "final_state": closed_final_metric,
        },
        "stock_diffusers_control": stock,
        "sidecar": sidecar_record,
    }


def _write_receipt(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_transformer(
    output_dir: Path,
    *,
    transformer_fixture_dir: Path,
    trajectory_fixture_dir: Path,
    expected_package_root: Path,
    product_revision: str,
) -> Path:
    """Run complete real MMDiT boundaries and saved-noise transformer evidence."""
    if not _REVISION.fullmatch(product_revision):
        raise ValueError("product_revision must be a 40- to 64-character lowercase hex commit")
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)

    RecordingReporter = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_reporting.py"))[
        "RecordingReporter"
    ]

    import kinomlx
    from kinomlx.models.krea2.components import Krea2Components
    from kinomlx.models.krea2.resources import prepare_resources
    from kinomlx.models.krea2.settings import Krea2Settings
    from kinomlx.models.krea2.transformer import KREA2_TRANSFORMER_CONSUMED_TENSORS
    from kinomlx.reporting import TimingReporter
    from kinomlx.settings import Settings

    package_file = Path(kinomlx.__file__).resolve()
    expected_package = expected_package_root.expanduser().resolve()
    if not package_file.is_relative_to(expected_package):
        raise AssertionError(f"loaded {package_file}, expected a package below {expected_package}")
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        raise FileExistsError(f"Krea transformer output directory already exists: {output_dir}")
    transformer_fixture_dir = transformer_fixture_dir.expanduser().absolute()
    trajectory_fixture_dir = trajectory_fixture_dir.expanduser().absolute()
    fixture_hashes = {
        "transformer": _validate_fixture_set(
            transformer_fixture_dir,
            TRANSFORMER_FIXTURE_SHA256,
        ),
        "trajectory": _validate_fixture_set(
            trajectory_fixture_dir,
            TRAJECTORY_FIXTURE_SHA256,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)

    infrastructure = Settings.from_env()
    turbo = prepare_resources(
        Krea2Settings(variant="turbo", transformer_dtype="bfloat16", reference_denoise=True),
        infrastructure=infrastructure,
    )
    raw = prepare_resources(
        Krea2Settings(variant="raw", transformer_dtype="bfloat16", reference_denoise=True),
        infrastructure=infrastructure,
    )
    if turbo.revision != TURBO_REVISION or raw.revision != RAW_REVISION:
        raise AssertionError("Krea transformer resolved an unpinned Raw or Turbo revision")
    if turbo.transformer_config != raw.transformer_config:
        raise AssertionError("Raw and Turbo no longer declare the same transformer graph")
    if turbo.transformer.graph_fingerprint != raw.transformer.graph_fingerprint:
        raise AssertionError("Raw and Turbo transformer graph identities differ")

    events = RecordingReporter()
    timing = TimingReporter(events)
    components = Krea2Components(reporter=timing)
    reset_peak = getattr(mx, "reset_peak_memory", None)
    peak_reset = False
    if reset_peak is not None:
        try:
            reset_peak()
            peak_reset = True
        except RuntimeError, TypeError, ValueError:
            pass
    memory_before, before_available = _memory_snapshot(mx)
    turbo_load_started = time.perf_counter()
    turbo_lease = components.transformer(turbo)
    turbo_load_elapsed = time.perf_counter() - turbo_load_started
    memory_turbo_loaded, turbo_loaded_available = _memory_snapshot(mx)
    try:
        stages = {
            stage: _run_stage(mx, turbo_lease.value, transformer_fixture_dir, stage)
            for stage in _STAGES
        }
        trajectory = _run_trajectory(
            mx,
            turbo_lease.value,
            trajectory_fixture_dir,
            output_dir,
        )
        memory_turbo_evaluated, turbo_evaluated_available = _memory_snapshot(mx)
    finally:
        turbo_lease.close()
    del turbo_lease
    gc.collect()
    mx.clear_cache()
    memory_after_turbo, after_turbo_available = _memory_snapshot(mx)

    raw_load_started = time.perf_counter()
    raw_lease = components.transformer(raw)
    raw_load_elapsed = time.perf_counter() - raw_load_started
    memory_raw_loaded, raw_loaded_available = _memory_snapshot(mx)
    try:
        # Binding intentionally leaves weights lazy. This Raw-only residency
        # probe materializes them explicitly; Turbo was evaluated by the
        # numerical stages above. Binding time is not evaluation time.
        raw_materialization_started = time.perf_counter()
        mx.eval(raw_lease.value.parameters())
        raw_materialization_elapsed = time.perf_counter() - raw_materialization_started
        memory_raw_evaluated, raw_evaluated_available = _memory_snapshot(mx)
    finally:
        raw_lease.close()
    del raw_lease
    gc.collect()
    mx.clear_cache()
    memory_after_raw, after_raw_available = _memory_snapshot(mx)
    memory_available = all(
        (
            before_available,
            turbo_loaded_available,
            turbo_evaluated_available,
            after_turbo_available,
            raw_loaded_available,
            raw_evaluated_available,
            after_raw_available,
        )
    )
    if not memory_available:
        raise AssertionError("MLX memory probes are unavailable; refusing an unmeasured gate")
    if memory_turbo_evaluated["active_bytes"] == 0 or memory_raw_evaluated["active_bytes"] == 0:
        raise AssertionError("MLX reported zero active memory after transformer evaluation")
    after_turbo_active = memory_after_turbo["active_bytes"]
    after_raw_active = memory_after_raw["active_bytes"]
    if not isinstance(after_turbo_active, int) or not isinstance(after_raw_active, int):
        raise AssertionError("MLX active-memory probes returned invalid values")
    if after_turbo_active > 1024**2 or after_raw_active > 1024**2:
        raise AssertionError("Krea transformer lease retained more than 1 MiB after release")
    if memory_after_turbo["cache_bytes"] != 0 or memory_after_raw["cache_bytes"] != 0:
        raise AssertionError("Krea transformer lease left MLX cache memory resident")

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
            "target_count": KREA2_TRANSFORMER_CONSUMED_TENSORS,
            "same_graph": True,
            "graph_fingerprint": turbo.transformer.graph_fingerprint,
            "turbo": {
                "revision": turbo.revision,
                "source_fingerprint": turbo.transformer.source_fingerprint,
                "source": str(turbo.transformer.source_path),
            },
            "raw": {
                "revision": raw.revision,
                "source_fingerprint": raw.transformer.source_fingerprint,
                "source": str(raw.transformer.source_path),
            },
        },
        "stages": stages,
        "trajectory": trajectory,
        "runtime": {
            "python_version": platform.python_version(),
            "mlx_version": importlib.metadata.version("mlx"),
            "peak_memory_reset": peak_reset,
            "memory_probes_available": memory_available,
            "turbo_load_seconds": turbo_load_elapsed,
            "raw_load_seconds": raw_load_elapsed,
            "raw_materialization_seconds": raw_materialization_elapsed,
            "memory_before": memory_before,
            "memory_turbo_loaded": memory_turbo_loaded,
            "memory_turbo_evaluated": memory_turbo_evaluated,
            "memory_after_turbo": memory_after_turbo,
            "memory_raw_loaded": memory_raw_loaded,
            "memory_raw_evaluated": memory_raw_evaluated,
            "memory_after_raw": memory_after_raw,
            "timing": timing.to_dict(),
            "blocked_import_policy": list(_BLOCKED),
            "blocked_import_attempts": sorted(blocker.attempted),
            "loaded_blocked_modules": offenders,
        },
    }
    receipt_path = output_dir / "transformer-transformer-receipt.json"
    _write_receipt(receipt_path, receipt)
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-fixture-dir", type=Path, required=True)
    parser.add_argument("--trajectory-fixture-dir", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--product-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    receipt = run_transformer(
        options.output_dir,
        transformer_fixture_dir=options.transformer_fixture_dir,
        trajectory_fixture_dir=options.trajectory_fixture_dir,
        expected_package_root=options.expected_package_root,
        product_revision=options.product_revision,
    )
    sys.stdout.write(str(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
