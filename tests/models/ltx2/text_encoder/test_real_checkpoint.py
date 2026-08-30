"""Opt-in end-to-end prompt encoding with official local weights."""

from __future__ import annotations

import gc

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.cache import ensure_weight_family_caches
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.text_encoder import encode_prompt
from kinomlx.settings import Settings


def _gemma_snapshot(settings: Settings):
    root = settings.hf_home / "hub/models--google--gemma-3-12b-it/snapshots"
    if not root.is_dir():
        return None
    return next(
        (
            path
            for path in sorted(root.iterdir())
            if (path / "tokenizer.model").is_file() and tuple(path.glob("model-*.safetensors"))
        ),
        None,
    )


@pytest.mark.requires_weights
def test_real_ltx23_prompt_encoding() -> None:
    settings = Settings.from_env()
    checkpoint = LTX2Settings.from_env().weights_path
    gemma = _gemma_snapshot(settings)
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("KINO_WEIGHTS_PATH is not configured")
    if gemma is None:
        pytest.skip("Gemma 3 12B weights are not cached under HF_HOME")
    connector = ensure_weight_family_caches(
        checkpoint,
        families=("connector",),
        cache_mode="auto",
        cache_root=settings.cache_dir,
    ).cache_paths["connector"]

    output = None
    try:
        output = encode_prompt(
            "a red cube",
            gemma_path=gemma,
            connector_path=connector,
            config_path=checkpoint,
        )
        assert tuple(output.video_encoding.shape) == (1, 1024, 4096)
        assert tuple(output.audio_encoding.shape) == (1, 1024, 2048)
        assert tuple(output.attention_mask.shape) == (1, 1024)
        assert mx.all(mx.isfinite(output.video_encoding)).item()
        assert mx.all(mx.isfinite(output.audio_encoding)).item()
        assert mx.all(output.attention_mask == 1).item()
    finally:
        del output
        gc.collect()
        mx.clear_cache()
