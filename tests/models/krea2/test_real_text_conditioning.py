"""Opt-in all-tap parity for the pinned Krea Qwen text fixture."""

import gc
import os
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.krea2.catalog import Krea2Variant, variant_spec
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.krea2.source_loading import load_component_tokenizer
from kinomlx.models.krea2.text_encoder import (
    Qwen3VLTextModel,
    load_qwen3_vl_text_weights,
)
from kinomlx.models.krea2.tokenizer import prepare_krea2_token_batch
from kinomlx.settings import Settings
from tests._reporting import RecordingReporter
from tests._snapshots import pinned_snapshot

from ._text_conditioning_fixtures import (
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


def _fixture_directory() -> Path:
    explicit = os.environ.get("KINO_KREA_ORACLE_TEXT_FIXTURE_DIR")
    if explicit is None:
        pytest.skip("set KINO_KREA_ORACLE_TEXT_FIXTURE_DIR to the pinned Krea text fixture")
    root = Path(explicit).expanduser().absolute()
    if not (root / "text-fixture.safetensors").is_file():
        pytest.skip("the pinned Krea text fixture is not present")
    return root


def _available_resources(infrastructure: Settings):
    pinned_snapshot(variant_spec(Krea2Variant.TURBO).repo_id, TURBO_REVISION)
    return prepare_resources(
        Krea2Settings(variant="turbo"),
        infrastructure=infrastructure,
    )


@pytest.mark.heavy
@pytest.mark.requires_metal
def test_real_qwen_text_graph_matches_all_twelve_oracle_taps() -> None:
    fixture_root = _fixture_directory()
    fixture_path = fixture_root / "text-fixture.safetensors"
    report_path = fixture_root / "text-report.json"
    assert sha256(fixture_path) == FIXTURE_SHA256
    assert sha256(report_path) == REPORT_SHA256

    infrastructure = Settings.from_env()
    resources = _available_resources(infrastructure)
    assert resources.revision == TURBO_REVISION

    loaded_fixture = mx.load(str(fixture_path), format="safetensors")
    input_ids = loaded_fixture["input_ids"].astype(mx.int32)
    attention_mask = loaded_fixture["attention_mask"].astype(mx.bool_)
    position_ids = loaded_fixture["position_ids_tagged_4_57_1"].astype(mx.int32)
    reference = loaded_fixture["official_tagged_hidden_taps_pre_slice"]
    loaded_fixture.clear()
    prompts = prompts_from_report(report_path)
    tokenizer = load_component_tokenizer(resources.tokenizer)
    prepared = prepare_krea2_token_batch(tokenizer, prompts)
    assert mx.array_equal(input_ids, mx.array(prepared.input_ids, dtype=mx.int32)).item()
    assert mx.array_equal(
        attention_mask,
        mx.array(prepared.attention_mask, dtype=mx.bool_),
    ).item()
    assert mx.array_equal(
        position_ids,
        mx.array(prepared.position_ids, dtype=mx.int32),
    ).item()

    model = Qwen3VLTextModel(
        resources.text_config,
        resources.capabilities.selected_text_layers,
    )
    reporter = RecordingReporter()
    try:
        receipt = load_qwen3_vl_text_weights(
            model,
            source_bindings=resources.text_encoder.bindings,
            reporter=reporter,
        )
        assert receipt.target_count == 386
        output = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            reporter=reporter,
        )
        assert output.selected_layers == resources.capabilities.selected_text_layers
        assert output.hidden_states.shape == reference.shape == (2, 546, 12, 2560)
        assert output.hidden_states.dtype == reference.dtype == mx.bfloat16
        assert mx.all(mx.isfinite(output.hidden_states)).item()
        for tap_index, selected_layer in enumerate(output.selected_layers):
            metrics = tap_metrics(
                output.hidden_states[:, :, tap_index, :],
                reference[:, :, tap_index, :],
            )
            assert metrics.normalized_rms <= NORMALIZED_RMS_LIMIT, (
                f"Qwen tap {selected_layer}: normalized RMS {metrics.normalized_rms} exceeds "
                f"{NORMALIZED_RMS_LIMIT}; cosine={metrics.cosine}, "
                f"max/scale={metrics.max_scale}"
            )
            assert metrics.cosine >= COSINE_FLOOR, (
                f"Qwen tap {selected_layer}: cosine {metrics.cosine} is below "
                f"{COSINE_FLOOR}; normalized RMS={metrics.normalized_rms}, "
                f"max/scale={metrics.max_scale}"
            )
            assert metrics.max_scale <= MAX_SCALE_LIMIT, (
                f"Qwen tap {selected_layer}: max/scale {metrics.max_scale} exceeds "
                f"{MAX_SCALE_LIMIT}; normalized RMS={metrics.normalized_rms}, "
                f"cosine={metrics.cosine}"
            )
        advances = [event for event in reporter.events if event[0] == "advance"]
        assert len(advances) == len(receipt.source_files)
        assert all(event[1] == "bind Qwen text weights" for event in advances)
        assert (
            "start",
            "encode prompt with Qwen3-VL",
            {"total": None, "unit": "it"},
        ) in reporter.events
    finally:
        del model
        gc.collect()
        mx.clear_cache()
