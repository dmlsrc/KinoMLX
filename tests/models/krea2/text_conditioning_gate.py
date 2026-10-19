"""Standalone source- or installed-wheel Krea 2 text-conditioning gate."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import runpy
import sys
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
from _text_conditioning_fixtures import (
    COSINE_FLOOR,
    FIXTURE_SHA256,
    MAX_SCALE_LIMIT,
    NORMALIZED_RMS_LIMIT,
    REPORT_SHA256,
    TURBO_REVISION,
    prompts_from_report,
    sha256,
    tap_metrics,
)

_BLOCKED = (
    "tokenizers",
    "transformers",
    "diffusers",
    "torch",
    "huggingface_hub",
    "numpy",
    "PIL",
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


def _write_receipt(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_text_conditioning(
    output_dir: Path,
    *,
    fixture_dir: Path,
    expected_package_root: Path,
    product_revision: str,
) -> Path:
    """Run the real prompt path with development-only imports blocked."""
    if not _REVISION.fullmatch(product_revision):
        raise ValueError("product_revision must be a 40- to 64-character lowercase hex commit")
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)

    RecordingReporter = runpy.run_path(str(Path(__file__).resolve().parents[2] / "_reporting.py"))[
        "RecordingReporter"
    ]

    import kinomlx
    from kinomlx.models.krea2.resources import prepare_resources
    from kinomlx.models.krea2.settings import Krea2Settings
    from kinomlx.models.krea2.text_conditioning import Krea2TextConditioner
    from kinomlx.models.krea2.tokenizer import (
        KREA_PREFIX_TOKENS,
        load_component_tokenizer,
        prepare_krea2_token_batch,
    )
    from kinomlx.reporting import TimingReporter
    from kinomlx.settings import Settings

    package_file = Path(kinomlx.__file__).resolve()
    expected = expected_package_root.expanduser().resolve()
    if not package_file.is_relative_to(expected):
        raise AssertionError(f"loaded {package_file}, expected a package below {expected}")

    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        raise FileExistsError(
            f"Krea text conditioning output directory already exists: {output_dir}"
        )
    fixture_dir = fixture_dir.expanduser().absolute()
    fixture_path = fixture_dir / "text-fixture.safetensors"
    report_path = fixture_dir / "text-report.json"
    if sha256(fixture_path) != FIXTURE_SHA256 or sha256(report_path) != REPORT_SHA256:
        raise AssertionError("Krea text conditioning fixture or report digest changed")

    resources = prepare_resources(
        Krea2Settings(variant="turbo"),
        infrastructure=Settings.from_env(),
    )
    if resources.revision != TURBO_REVISION:
        raise AssertionError(
            f"resolved Krea Turbo revision {resources.revision}, expected {TURBO_REVISION}"
        )

    fixture = mx.load(str(fixture_path), format="safetensors")
    input_ids = fixture["input_ids"].astype(mx.int32)
    attention_mask = fixture["attention_mask"].astype(mx.bool_)
    position_ids = fixture["position_ids_tagged_4_57_1"].astype(mx.int32)
    reference = fixture["official_tagged_hidden_taps_post_slice"]
    fixture.clear()
    prompts = prompts_from_report(report_path)
    tokenizer = load_component_tokenizer(resources.tokenizer)
    prepared = prepare_krea2_token_batch(tokenizer, prompts)
    if not mx.array_equal(input_ids, mx.array(prepared.input_ids, dtype=mx.int32)).item():
        raise AssertionError("native tokenizer IDs differ from the oracle fixture")
    if not mx.array_equal(
        attention_mask,
        mx.array(prepared.attention_mask, dtype=mx.bool_),
    ).item():
        raise AssertionError("native tokenizer mask differs from the oracle fixture")
    if not mx.array_equal(
        position_ids,
        mx.array(prepared.position_ids, dtype=mx.int32),
    ).item():
        raise AssertionError("native tokenizer positions differ from the oracle fixture")

    events = RecordingReporter()
    timing = TimingReporter(events)
    reset_peak = getattr(mx, "reset_peak_memory", None)
    peak_reset = False
    if reset_peak is not None:
        try:
            reset_peak()
            peak_reset = True
        except RuntimeError, TypeError, ValueError:
            pass
    memory_before, memory_before_available = _memory_snapshot(mx)
    started = time.perf_counter()
    conditioning = Krea2TextConditioner()(
        prompts,
        resources,
        reporter=timing,
    )
    elapsed = time.perf_counter() - started

    candidate = conditioning.positive.hidden_states
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise AssertionError(
            f"conditioning shape/dtype {candidate.shape}/{candidate.dtype} differs from "
            f"fixture {reference.shape}/{reference.dtype}"
        )
    if conditioning.negative is not None:
        raise AssertionError("Turbo positive-only gate constructed a negative branch")
    if not mx.array_equal(
        conditioning.positive.attention_mask,
        attention_mask[:, KREA_PREFIX_TOKENS:],
    ).item():
        raise AssertionError("post-Qwen conditioning mask differs from the oracle fixture")
    if not mx.array_equal(
        conditioning.positive.position_ids,
        position_ids[:, :, KREA_PREFIX_TOKENS:],
    ).item():
        raise AssertionError("post-Qwen conditioning positions differ from the oracle fixture")

    metrics = []
    for tap_index, selected_layer in enumerate(resources.capabilities.selected_text_layers):
        current = tap_metrics(
            candidate[:, :, tap_index, :],
            reference[:, :, tap_index, :],
        )
        if current.normalized_rms > NORMALIZED_RMS_LIMIT:
            raise AssertionError(
                f"Qwen tap {selected_layer} normalized RMS {current.normalized_rms} exceeds "
                f"{NORMALIZED_RMS_LIMIT}"
            )
        if current.cosine < COSINE_FLOOR:
            raise AssertionError(
                f"Qwen tap {selected_layer} cosine {current.cosine} is below {COSINE_FLOOR}"
            )
        if current.max_scale > MAX_SCALE_LIMIT:
            raise AssertionError(
                f"Qwen tap {selected_layer} max/scale {current.max_scale} exceeds {MAX_SCALE_LIMIT}"
            )
        metrics.append({"layer": selected_layer, **current.to_dict()})

    memory_with_product, memory_with_product_available = _memory_snapshot(mx)
    memory_probes_available = memory_before_available and memory_with_product_available
    if not memory_probes_available:
        raise AssertionError("MLX memory probes are unavailable; refusing an unmeasured gate")
    if memory_with_product["active_bytes"] == 0:
        raise AssertionError("MLX reported zero active memory while conditioning remained live")
    observed_shape = [int(dimension) for dimension in candidate.shape]
    observed_dtype = str(candidate.dtype).rsplit(".", maxsplit=1)[-1]
    provenance = conditioning.provenance.to_metadata()
    prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    del conditioning, candidate, reference, input_ids, attention_mask, position_ids
    gc.collect()
    mx.clear_cache()
    memory_after_release, memory_after_release_available = _memory_snapshot(mx)
    memory_probes_available = memory_probes_available and memory_after_release_available
    if not memory_probes_available:
        raise AssertionError("MLX memory probes became unavailable after release")
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
        "schema_version": 2,
        "status": "pass",
        "package": {
            "version": importlib.metadata.version("kinomlx"),
            "revision": product_revision,
            "module_path": str(package_file),
            "expected_package_root": str(expected),
        },
        "fixture": {
            "fixture_sha256": FIXTURE_SHA256,
            "report_sha256": REPORT_SHA256,
            "prompt_sha256": prompt_hashes,
        },
        "resources": {
            "revision": resources.revision,
            "recipe_identity": resources.recipe_identity,
            "tokenizer_source_identity": resources.tokenizer.source_fingerprint,
            "tokenizer_graph_identity": resources.tokenizer.graph_fingerprint,
            "text_source_identity": resources.text_encoder.source_fingerprint,
            "text_graph_identity": resources.text_encoder.graph_fingerprint,
            "text_dtype_policy": resources.text_encoder.dtype_policy,
        },
        "conditioning": {
            "shape": observed_shape,
            "dtype": observed_dtype,
            "selected_layers": list(resources.capabilities.selected_text_layers),
            "qwen_target_count": provenance["qwen_target_count"],
            "qwen_target_digest": provenance["qwen_target_digest"],
            "compatibility_identity": provenance["compatibility_identity"],
            "tap_metrics": metrics,
            "limits": {
                "normalized_rms": NORMALIZED_RMS_LIMIT,
                "cosine": COSINE_FLOOR,
                "max_scale": MAX_SCALE_LIMIT,
            },
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "python_version": platform.python_version(),
            "unicode_version": unicodedata.unidata_version,
            "mlx_version": importlib.metadata.version("mlx"),
            "peak_memory_reset": peak_reset,
            "memory_probes_available": memory_probes_available,
            "memory_before": memory_before,
            "memory_with_product": memory_with_product,
            "memory_after_release": memory_after_release,
            "timing": timing.to_dict(),
            "blocked_import_policy": list(_BLOCKED),
            "blocked_import_attempts": sorted(blocker.attempted),
            "loaded_blocked_modules": offenders,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "text-conditioning-text-receipt.json"
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
    receipt = run_text_conditioning(
        options.output_dir,
        fixture_dir=options.fixture_dir,
        expected_package_root=options.expected_package_root,
        product_revision=options.product_revision,
    )
    sys.stdout.write(str(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
