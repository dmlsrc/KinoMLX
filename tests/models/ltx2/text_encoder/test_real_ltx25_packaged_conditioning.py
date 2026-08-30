"""Opt-in parity for the packaged 2.5 projection and connector sources."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.kernels import rms_norm
from kinomlx.models.ltx2.text_encoder import (
    AVTextEncoderConfig,
    GemmaTokenizer,
    create_av_text_encoder_v2,
    encode_prompt,
    ensure_tokenizer_cache,
    load_av_text_encoder_v2_weights,
    norm_and_concat_per_token_rms,
)
from kinomlx.models.ltx2.text_encoder.connector import precompute_connector_rope

_FIXTURE_SHA256 = "d8384e955e8394374a5fd774ee87d9646f71a19150a31ddd4fb71f3eb44b4c90"
_PROMPT_CORPUS_SHA256 = "705c385a88aa4190cbb630ce13db84a955254711aed7a3495371d99f3dcb7129"
_PROMPT_CASES = (
    (
        "short_compact",
        "A lighthouse in sea mist",
        False,
        (2, 236776, 63937, 528, 5442, 8442),
        (1, 1, 1, 1, 1, 1),
    ),
    (
        "left_padded",
        "mist",
        True,
        (0, 0, 0, 0, 0, 0, 2, 35768),
        (0, 0, 0, 0, 0, 0, 1, 1),
    ),
    (
        "truncated",
        "word " * 100,
        True,
        (2, 3017, 3658, 3658, 3658, 3658, 3658, 3658),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ),
    (
        "whitespace",
        " \t\n ",
        True,
        (0, 0, 0, 0, 0, 0, 0, 2),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ),
    (
        "unicode",
        "\u65e5\u672c\u8a9e \u4e2d\u6587 \ud55c\uad6d\uc5b4 \U0001f3ac",
        True,
        (2, 94951, 17346, 237364, 50413, 237430, 236743, 245357),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ),
    (
        "literal_specials",
        "<bos> before <eos> after <pad> <unk>",
        True,
        (2, 1680, 236743, 1, 1308, 236743, 0, 236743),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ),
)


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


def _assert_accumulated_boundary(
    name: str,
    candidate: mx.array,
    reference: mx.array,
) -> None:
    """Check aggregate agreement after many backend-dependent BF16 transitions."""
    candidate = candidate.astype(mx.float32)
    reference = reference.astype(mx.float32)
    mx.eval(candidate, reference)
    assert tuple(candidate.shape) == tuple(reference.shape), name
    assert mx.all(mx.isfinite(candidate)).item(), name
    delta = candidate - reference
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference))).item())
    normalized_rms = float(mx.sqrt(mx.mean(mx.square(delta))).item()) / max(
        reference_rms,
        1e-12,
    )
    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    cosine = float(
        mx.sum(candidate_flat * reference_flat).item()
        / (mx.linalg.norm(candidate_flat).item() * mx.linalg.norm(reference_flat).item())
    )
    limit = 4 * 2**-7
    assert normalized_rms <= limit, (
        f"{name}: normalized_rms={normalized_rms} exceeds {limit}; cosine={cosine}"
    )
    assert cosine >= 1.0 - 0.5 * limit**2, (
        f"{name}: cosine={cosine} is below {1.0 - 0.5 * limit**2}"
    )


def _check_connector(
    connector,
    references: dict[str, mx.array],
    prefix: str,
) -> None:
    value = references[f"{prefix}.input"].astype(mx.bfloat16)
    positions = mx.arange(value.shape[1], dtype=mx.float32)[None, None, :]
    rope = precompute_connector_rope(
        positions,
        inner_dim=connector.inner_dim,
        num_heads=connector.num_attention_heads,
        theta=connector.positional_embedding_theta,
        max_positions=connector.positional_embedding_max_pos,
        output_dtype=value.dtype,
        double_precision=connector.double_precision_rope,
    )
    for index, block in enumerate(connector.transformer_1d_blocks):
        previous = "input" if index == 0 else f"block.{index - 1:02d}"
        value = block(
            references[f"{prefix}.{previous}"].astype(mx.bfloat16),
            mask=None,
            rope=rope,
        )
        _assert_boundary(
            f"{prefix}.block.{index:02d}.local",
            value,
            references[f"{prefix}.block.{index:02d}"],
        )
    final = rms_norm(
        references[f"{prefix}.block.07"].astype(mx.bfloat16),
        eps=connector.norm_eps,
    )
    _assert_boundary(f"{prefix}.output.local", final, references[f"{prefix}.output"])


@pytest.mark.requires_weights
def test_real_ltx25_packaged_projection_and_connectors_match_reference() -> None:
    text_checkpoint = _required_path("KINO_LTX25_TEXT_ENCODER_PATH")
    transformer_checkpoint = _required_path("KINO_LTX25_TRANSFORMER_PATH")
    gemma_fixture_dir = _required_path("KINO_LTX25_TEXT_FIXTURE_DIR", directory=True)
    fixture_dir = _required_path("KINO_LTX25_CONDITIONING_FIXTURE_DIR", directory=True)
    fixture_path = fixture_dir / "packaged-conditioning.safetensors"
    if not fixture_path.is_file():
        pytest.skip("the complete LTX-2.5 packaged-conditioning fixture is not present")
    assert _sha256(fixture_path) == _FIXTURE_SHA256

    gemma_states = mx.load(str(gemma_fixture_dir / "gemma4-ltx-layer-states.safetensors"))
    references = mx.load(str(fixture_path))
    model = None
    try:
        config = AVTextEncoderConfig.from_checkpoint(transformer_checkpoint)
        model = create_av_text_encoder_v2(config)
        assert (
            load_av_text_encoder_v2_weights(
                model,
                transformer_checkpoint,
                projection_path=text_checkpoint,
            )
            == 4 + 129 + 129
        )

        state_names = (
            "embedding",
            *(f"layer.{index:02d}" for index in range(47)),
            "final_norm",
        )
        hidden_states = tuple(
            gemma_states[name][:, -6:, :].astype(mx.bfloat16) for name in state_names
        )
        attention_mask = references["attention_mask.trimmed"].astype(mx.int32)
        stacked = mx.stack(hidden_states, axis=-1)
        projection_input = norm_and_concat_per_token_rms(stacked, attention_mask)
        _assert_boundary("projection_input", projection_input, references["projection_input"])

        video_projected, audio_projected = model.feature_extractor(
            hidden_states,
            attention_mask,
        )
        _assert_boundary("video.projected", video_projected, references["video.projected"])
        _assert_boundary("audio.projected", audio_projected, references["audio.projected"])

        video_input = model.embeddings_connector._append_registers(video_projected)
        audio_input = model.audio_embeddings_connector._append_registers(audio_projected)
        _assert_boundary("video.input", video_input, references["video.input"])
        _assert_boundary("audio.input", audio_input, references["audio.input"])
        _check_connector(model.embeddings_connector, references, "video")
        _check_connector(model.audio_embeddings_connector, references, "audio")

        video_output, video_additive_mask = model.embeddings_connector(video_projected, None)
        audio_output, audio_additive_mask = model.audio_embeddings_connector(audio_projected, None)
        for name, value, reference in (
            ("video.output.whole", video_output, references["video.output"]),
            ("audio.output.whole", audio_output, references["audio.output"]),
        ):
            mx.eval(value)
            assert tuple(value.shape) == tuple(reference.shape), name
            assert mx.all(mx.isfinite(value)).item(), name
            assert mx.any(value != 0).item(), name
        video_mask = (video_additive_mask[:, 0, 0, :] >= -0.5).astype(mx.int32)
        audio_mask = (audio_additive_mask[:, 0, 0, :] >= -0.5).astype(mx.int32)
        assert mx.array_equal(video_mask, references["connector_mask"]).item()
        assert mx.array_equal(audio_mask, references["connector_mask"]).item()
    finally:
        del model, gemma_states, references
        gc.collect()
        mx.clear_cache()


@pytest.mark.requires_weights
def test_real_ltx25_prompt_station_matches_packaged_context(tmp_path: Path) -> None:
    text_checkpoint = _required_path("KINO_LTX25_TEXT_ENCODER_PATH")
    transformer_checkpoint = _required_path("KINO_LTX25_TRANSFORMER_PATH")
    fixture_dir = _required_path("KINO_LTX25_CONDITIONING_FIXTURE_DIR", directory=True)
    references = mx.load(str(fixture_dir / "packaged-conditioning.safetensors"))
    output = None
    try:
        tokenizer_cache = ensure_tokenizer_cache(text_checkpoint, cache_root=tmp_path)
        output = encode_prompt(
            "A lighthouse in sea mist",
            gemma_path=text_checkpoint,
            projection_path=text_checkpoint,
            connector_path=transformer_checkpoint,
            config_path=transformer_checkpoint,
            model_generation="2.5",
            tokenizer_cache=tokenizer_cache,
            max_length=8,
            pad_prompt_to_max=True,
        )
        _assert_accumulated_boundary(
            "prompt.video",
            output.video_encoding,
            references["video.output"],
        )
        _assert_accumulated_boundary(
            "prompt.audio",
            output.audio_encoding,
            references["audio.output"],
        )
        assert mx.array_equal(output.attention_mask, references["connector_mask"]).item()
    finally:
        del output, references
        gc.collect()
        mx.clear_cache()


@pytest.mark.requires_weights
def test_real_ltx25_prompt_corpus_matches_reference(tmp_path: Path) -> None:
    text_checkpoint = _required_path("KINO_LTX25_TEXT_ENCODER_PATH")
    transformer_checkpoint = _required_path("KINO_LTX25_TRANSFORMER_PATH")
    fixture_dir = _required_path("KINO_LTX25_PROMPT_FIXTURE_DIR", directory=True)
    fixture_path = fixture_dir / "prompt-context-corpus.safetensors"
    if not fixture_path.is_file():
        pytest.skip("the complete LTX-2.5 prompt-context corpus is not present")
    assert _sha256(fixture_path) == _PROMPT_CORPUS_SHA256

    references = mx.load(str(fixture_path))
    tokenizer_cache = ensure_tokenizer_cache(text_checkpoint, cache_root=tmp_path)
    tokenizer = GemmaTokenizer(tokenizer_cache)
    output = None
    try:
        for index, (label, prompt, pad_to_max, expected_ids, expected_mask) in enumerate(
            _PROMPT_CASES
        ):
            input_ids, token_mask = tokenizer.encode(
                prompt,
                max_length=8,
                pad_to_max=pad_to_max,
            )
            assert input_ids.tolist() == [list(expected_ids)], label
            assert token_mask.tolist() == [list(expected_mask)], label
            assert mx.array_equal(
                input_ids,
                references[f"case.{index}.input_ids"].astype(mx.int32),
            ).item(), label
            assert mx.array_equal(
                token_mask,
                references[f"case.{index}.attention_mask"].astype(mx.int32),
            ).item(), label

            output = encode_prompt(
                prompt,
                gemma_path=text_checkpoint,
                projection_path=text_checkpoint,
                connector_path=transformer_checkpoint,
                config_path=transformer_checkpoint,
                model_generation="2.5",
                tokenizer_cache=tokenizer_cache,
                max_length=8,
                pad_prompt_to_max=pad_to_max,
            )
            _assert_accumulated_boundary(
                f"{label}.video",
                output.video_encoding,
                references[f"case.{index}.video"],
            )
            _assert_accumulated_boundary(
                f"{label}.audio",
                output.audio_encoding,
                references[f"case.{index}.audio"],
            )
            assert mx.array_equal(
                output.attention_mask,
                references[f"case.{index}.connector_mask"],
            ).item(), label
            del output
            output = None
            gc.collect()
            mx.clear_cache()
    finally:
        del output, tokenizer, references
        gc.collect()
        mx.clear_cache()


@pytest.mark.requires_weights
def test_real_ltx25_default_padding_geometry_matches_reference(tmp_path: Path) -> None:
    text_checkpoint = _required_path("KINO_LTX25_TEXT_ENCODER_PATH")
    transformer_checkpoint = _required_path("KINO_LTX25_TRANSFORMER_PATH")
    fixture_dir = _required_path("KINO_LTX25_PROMPT_FIXTURE_DIR", directory=True)
    fixture_path = fixture_dir / "prompt-context-corpus.safetensors"
    if not fixture_path.is_file():
        pytest.skip("the complete LTX-2.5 prompt-context corpus is not present")
    assert _sha256(fixture_path) == _PROMPT_CORPUS_SHA256

    references = mx.load(str(fixture_path))
    output = None
    try:
        tokenizer_cache = ensure_tokenizer_cache(text_checkpoint, cache_root=tmp_path)
        output = encode_prompt(
            "A lighthouse in sea mist",
            gemma_path=text_checkpoint,
            projection_path=text_checkpoint,
            connector_path=transformer_checkpoint,
            config_path=transformer_checkpoint,
            model_generation="2.5",
            tokenizer_cache=tokenizer_cache,
        )
        _assert_accumulated_boundary(
            "default-padding.video",
            output.video_encoding,
            references["case.0.video"],
        )
        _assert_accumulated_boundary(
            "default-padding.audio",
            output.audio_encoding,
            references["case.0.audio"],
        )
        assert mx.array_equal(
            output.attention_mask,
            references["case.0.connector_mask"],
        ).item()
    finally:
        del output, references
        gc.collect()
        mx.clear_cache()
