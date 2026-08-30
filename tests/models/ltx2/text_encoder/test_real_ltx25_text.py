"""Opt-in layerwise and whole-backbone parity for the LTX-tuned Gemma 4."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.text_encoder import Gemma4Model, load_gemma4_weights

_INPUTS_SHA256 = "f4891e4f0207b2afda878081ca9c94639f844ea1d658299188a0a399a8192864"
_OUTPUTS_SHA256 = "1489cf37bace15ae5c1e356577ff21909ea8d26dbecedd59a8b59c7e9f22715e"


def _required_path(environment_name: str, *, directory: bool = False) -> Path:
    raw = os.environ.get(environment_name)
    if raw is None:
        pytest.skip(f"{environment_name} is not configured")
    path = Path(raw).expanduser().absolute()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        pytest.skip(f"{environment_name} does not exist")
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _assert_boundary(name: str, candidate: mx.array, reference: mx.array) -> None:
    candidate = candidate.astype(mx.float32)
    reference = reference.astype(mx.float32)
    mx.eval(candidate, reference)
    assert tuple(candidate.shape) == tuple(reference.shape), name
    assert candidate.size > 0, name
    assert mx.all(mx.isfinite(candidate)).item(), name

    delta = mx.abs(candidate - reference)
    reference_max = float(mx.max(mx.abs(reference)).item())
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference))).item())
    scale = max(reference_max, reference_rms, 1e-12)
    max_abs = float(mx.max(delta).item())
    normalized_rms = float(mx.sqrt(mx.mean(mx.square(delta))).item()) / max(
        reference_rms,
        scale * 2**-12,
    )
    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    candidate_norm = float(mx.linalg.norm(candidate_flat).item())
    reference_norm = float(mx.linalg.norm(reference_flat).item())
    if candidate_norm == 0.0 or reference_norm == 0.0:
        cosine = 1.0 if candidate_norm == reference_norm == max_abs == 0.0 else 0.0
    else:
        cosine = float(
            mx.sum(candidate_flat * reference_flat).item() / (candidate_norm * reference_norm)
        )

    limit = 4 * 2**-7
    assert max_abs <= scale * limit, (
        f"{name}: max_abs={max_abs} exceeds {scale * limit}; "
        f"normalized_rms={normalized_rms}, cosine={cosine}"
    )
    assert normalized_rms <= limit, (
        f"{name}: normalized_rms={normalized_rms} exceeds {limit}; "
        f"max_abs={max_abs}, cosine={cosine}"
    )
    assert cosine >= 1.0 - 0.5 * limit**2, (
        f"{name}: cosine={cosine} is below {1.0 - 0.5 * limit**2}"
    )


@pytest.mark.requires_weights
def test_real_ltx25_gemma4_matches_all_frozen_boundaries() -> None:
    checkpoint = _required_path("KINO_LTX25_TEXT_ENCODER_PATH")
    fixture_dir = _required_path("KINO_LTX25_TEXT_FIXTURE_DIR", directory=True)
    inputs_path = fixture_dir / "gemma4-ltx-inputs.safetensors"
    outputs_path = fixture_dir / "gemma4-ltx-layer-states.safetensors"
    if not inputs_path.is_file() or not outputs_path.is_file():
        pytest.skip("the complete LTX-2.5 Gemma 4 fixture is not present")
    assert _sha256(inputs_path) == _INPUTS_SHA256
    assert _sha256(outputs_path) == _OUTPUTS_SHA256

    inputs = mx.load(str(inputs_path))
    references = mx.load(str(outputs_path))
    model = None
    try:
        model = Gemma4Model()
        assert load_gemma4_weights(model, checkpoint) == 666
        input_ids = inputs["input_ids"].astype(mx.int32)
        attention_mask = inputs["attention_mask"].astype(mx.int32)
        position_ids = inputs["position_ids"].astype(mx.int32)
        full_mask, sliding_mask = model.attention_masks(attention_mask)

        embedding = model.embed_tokens(input_ids)
        scale = mx.array(model.config.hidden_size**0.5, dtype=mx.bfloat16)
        embedding = embedding * scale.astype(embedding.dtype)
        _assert_boundary("embedding.local", embedding, references["embedding"])

        for index, layer in enumerate(model.layers):
            previous = "embedding" if index == 0 else f"layer.{index - 1:02d}"
            mask = sliding_mask if layer.layer_type == "sliding_attention" else full_mask
            output = layer(
                references[previous].astype(mx.bfloat16),
                mask,
                position_ids,
            )
            _assert_boundary(
                f"layer.{index:02d}.local",
                output,
                references[f"layer.{index:02d}"],
            )

        final = model.norm(references["layer.47"].astype(mx.bfloat16))
        _assert_boundary("final_norm.local", final, references["final_norm"])

        boundaries = model.forward_boundaries(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        assert tuple(boundaries) == (
            "embedding",
            *(f"layer.{index:02d}" for index in range(48)),
            "final_norm",
        )
        for name, value in boundaries.items():
            mx.eval(value)
            assert tuple(value.shape) == tuple(references[name].shape), name
            assert mx.all(mx.isfinite(value)).item(), name
            assert mx.any(value != 0).item(), name
    finally:
        del model, inputs, references
        gc.collect()
        mx.clear_cache()
