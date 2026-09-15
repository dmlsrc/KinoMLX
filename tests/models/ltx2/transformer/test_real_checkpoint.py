"""Opt-in real-checkpoint forward gate for the LTX-2.3 transformer."""

import gc
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from kinomlx.models.ltx2.metadata import checkpoint_config
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality, X0Model
from kinomlx.models.ltx2.weights.loading import bind_transformer
from kinomlx.models.ltx2.weights.schema import LAYOUT_KEY_PREFIX
from tests.models.ltx2._source_weights import (
    NO_LTX23_CHECKPOINT,
    ltx23_checkpoint,
    transformer_weights,
)


@pytest.mark.heavy
def test_real_ltx23_transformer_weights_binds_and_runs_video_forward() -> None:
    if os.environ.get("KINO_RUN_REAL_LTX23_TRANSFORMER") != "1":
        pytest.skip("set KINO_RUN_REAL_LTX23_TRANSFORMER=1 to build and test the full transformer")
    checkpoint = ltx23_checkpoint()
    if checkpoint is None:
        pytest.skip(NO_LTX23_CHECKPOINT)
    config = checkpoint_config(checkpoint).transformer
    assert config.model_generation == "2.3"
    model = None
    output = None
    audio_output = None
    try:
        model = LTXAVModel.from_config(config, compute_dtype=mx.bfloat16)
        bind_transformer(
            model,
            weight_loader=lambda _index: transformer_weights(checkpoint, config=config),
            transformer_dtype=mx.bfloat16,
            include_audio=True,
            resident_blocks=1,
        )
        assert model.num_blocks == 48
        assert X0Model(model).num_blocks == 48
        model.transformer_block_streamer.bind(model.transformer_blocks[0], 0)
        parameters = dict(tree_flatten(model.parameters()))
        cached = transformer_weights(checkpoint, config=config)
        expected = {
            key
            for key in cached
            if not key.startswith(LAYOUT_KEY_PREFIX)
            and (
                not key.startswith("transformer_blocks.") or key.startswith("transformer_blocks.0.")
            )
        }
        layouts = {
            key[len(LAYOUT_KEY_PREFIX) :]
            for key in cached
            if key.startswith(f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.")
        }
        expected -= {key.removesuffix("_t") for key in layouts}
        assert set(parameters) == expected

        video_positions = mx.array([[[[0, 1]], [[0, 32]], [[0, 32]]]])
        video = Modality(
            latent=mx.random.normal((1, 1, 128)),
            context=mx.random.normal((1, 1, 4096)),
            timesteps=mx.array([[0.5]]),
            sigma=mx.array([0.5]),
            positions=video_positions,
            context_mask=mx.ones((1, 1), dtype=mx.int32),
        )
        audio_positions = mx.array([[[[0, 1]]]])
        audio = Modality(
            latent=mx.random.normal((1, 1, 128)),
            context=mx.random.normal((1, 1, 2048)),
            timesteps=mx.array([[0.5]]),
            sigma=mx.array([0.5]),
            positions=audio_positions,
            context_mask=mx.ones((1, 1), dtype=mx.int32),
        )
        bound_indices: list[int] = []
        original_bind = model.transformer_block_streamer.bind

        def recording_bind(
            block,
            block_idx: int,
            *,
            evict_block_idx: int | None = None,
        ):
            bound_indices.append(block_idx)
            return original_bind(
                block,
                block_idx,
                evict_block_idx=evict_block_idx,
            )

        model.transformer_block_streamer.bind = recording_bind
        output, audio_output = model(video, audio)
        mx.eval(output, audio_output)
        assert bound_indices == list(range(48))
        assert tuple(output.shape) == (1, 1, 128)
        assert tuple(audio_output.shape) == (1, 1, 128)
        assert output.dtype == mx.bfloat16
        assert audio_output.dtype == mx.bfloat16
        assert mx.all(mx.isfinite(output)).item()
        assert mx.all(mx.isfinite(audio_output)).item()
        assert mx.any(output != 0).item()
        assert mx.any(audio_output != 0).item()
    finally:
        if model is not None:
            model.close_streamer()
        del model, output, audio_output
        gc.collect()
        mx.clear_cache()
