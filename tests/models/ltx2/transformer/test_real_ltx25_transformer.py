"""Opt-in one-token parity gate for the real LTX-2.5 transformer."""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.cache import load_transformer_weights_cached_streaming
from kinomlx.models.ltx2.metadata import checkpoint_config
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality
from kinomlx.settings import Settings

_INPUTS_SHA256 = "4ed4fc5a791f87ffd66560bddcd98d5e371c9624f11170cafcb4f46538038755"
_OUTPUTS_SHA256 = "d173a19e10e2518614572f901b1d0bfd40ca179eb2f3855effb3f342b48fbc6d"


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
def test_real_ltx25_one_token_joint_transformer_matches_frozen_reference() -> None:
    checkpoint = _required_path("KINO_LTX25_TRANSFORMER_PATH")
    fixture_dir = _required_path("KINO_LTX25_TRANSFORMER_FIXTURE_DIR", directory=True)
    inputs_path = fixture_dir / "transformer-one-token-inputs.safetensors"
    outputs_path = fixture_dir / "transformer-one-token.safetensors"
    if not inputs_path.is_file() or not outputs_path.is_file():
        pytest.skip("the complete LTX-2.5 transformer fixture is not present")
    assert _sha256(inputs_path) == _INPUTS_SHA256
    assert _sha256(outputs_path) == _OUTPUTS_SHA256

    config = checkpoint_config(checkpoint).transformer
    assert config.model_generation == "2.5"
    inputs = mx.load(str(inputs_path))
    references = mx.load(str(outputs_path))
    model = None
    video_output = None
    audio_output = None
    checked: list[str] = []
    try:
        model = LTXAVModel.from_config(
            config,
            compute_dtype=mx.bfloat16,
            use_steel_attention=False,
            compile_attention=False,
            fast_mode=False,
        )
        load_transformer_weights_cached_streaming(
            model,
            checkpoint,
            transformer_dtype=mx.bfloat16,
            cache_mode="auto",
            cache_root=Settings.from_env().cache_dir,
            include_audio=True,
            resident_blocks=1,
            constructor_config=config,
        )
        video = Modality(
            latent=inputs["video_latent"],
            context=inputs["video_text"],
            timesteps=inputs["video_timestep"].astype(mx.float32)
            / config.timestep_scale_multiplier,
            sigma=inputs["video_sigma"].astype(mx.float32) / config.timestep_scale_multiplier,
            positions=inputs["video_coords"],
            context_mask=inputs["video_text_mask"],
        )
        audio = Modality(
            latent=inputs["audio_latent"],
            context=inputs["audio_text"],
            timesteps=inputs["audio_timestep"].astype(mx.float32)
            / config.timestep_scale_multiplier,
            sigma=inputs["audio_sigma"].astype(mx.float32) / config.timestep_scale_multiplier,
            positions=inputs["audio_coords"],
            context_mask=inputs["audio_text_mask"],
        )

        video_args = model._video_preprocessor.prepare(video, audio)
        audio_args = model._audio_preprocessor.prepare(audio, video)
        _assert_boundary("proj_in", video_args.x, references["proj_in"])
        _assert_boundary(
            "audio_proj_in",
            audio_args.x,
            references["audio_proj_in"],
        )
        checked.extend(("proj_in", "audio_proj_in"))

        streamer = model.transformer_block_streamer
        assert streamer is not None
        for index in range(config.num_layers):
            previous_video = "proj_in" if index == 0 else f"block.{index - 1:02d}.0"
            previous_audio = "audio_proj_in" if index == 0 else f"block.{index - 1:02d}.1"
            block = streamer.bind(
                model.transformer_blocks[0],
                index,
                evict_block_idx=index - 1 if index else None,
            )
            block_video, block_audio = block(
                replace(
                    video_args,
                    x=references[previous_video].astype(mx.bfloat16),
                ),
                replace(
                    audio_args,
                    x=references[previous_audio].astype(mx.bfloat16),
                ),
            )
            assert block_video is not None
            assert block_audio is not None
            video_name = f"block.{index:02d}.0"
            audio_name = f"block.{index:02d}.1"
            _assert_boundary(video_name, block_video.x, references[video_name])
            _assert_boundary(audio_name, block_audio.x, references[audio_name])
            checked.extend((video_name, audio_name))

        local_video_output = model._output(
            references["block.47.0"].astype(mx.bfloat16),
            video_args.embedded_timestep,
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
        )
        local_audio_output = model._output(
            references["block.47.1"].astype(mx.bfloat16),
            audio_args.embedded_timestep,
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
        )
        _assert_boundary(
            "output.video",
            local_video_output,
            references["output.video"],
        )
        _assert_boundary(
            "output.audio",
            local_audio_output,
            references["output.audio"],
        )
        checked.extend(("output.video", "output.audio"))

        video_output, audio_output = model(video, audio)
        mx.eval(video_output, audio_output)
        assert checked == [
            "proj_in",
            "audio_proj_in",
            *(
                boundary
                for index in range(48)
                for boundary in (f"block.{index:02d}.0", f"block.{index:02d}.1")
            ),
            "output.video",
            "output.audio",
        ]
        assert tuple(video_output.shape) == tuple(references["output.video"].shape)
        assert tuple(audio_output.shape) == tuple(references["output.audio"].shape)
        assert mx.all(mx.isfinite(video_output)).item()
        assert mx.all(mx.isfinite(audio_output)).item()
        assert mx.any(video_output != 0).item()
        assert mx.any(audio_output != 0).item()
    finally:
        if model is not None:
            model.close_streamer()
        del model, inputs, references, video_output, audio_output
        gc.collect()
        mx.clear_cache()
