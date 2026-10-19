"""Small, deterministic Krea transformer fixtures for transformer tests."""

import mlx.core as mx

from kinomlx.models.krea2.metadata import Krea2TransformerConfig

RAW_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
TURBO_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"

TRANSFORMER_FIXTURE_SHA256 = {
    "transformer-input-requested.safetensors": (
        "910774b4d3b302b0aa5c0f0bcd603ea8b303c8f0bb7aea030d8f19de94f1face"
    ),
    "official-transformer-boundaries-requested.safetensors": (
        "fa42567d15a6490aa5f4acbeb41d56f1198d28c58da1c041a7df7c480b333957"
    ),
    "transformer-input-shape-padding-control.safetensors": (
        "1d39d52bd6705c6479d0b0876df2cad0842a12af0d348b4a611266d9691c3eb7"
    ),
    "official-transformer-boundaries-shape-padding-control.safetensors": (
        "675ce1305c5fa9f05425b47cf274c6fd9229598d25b0e66537e2ca8fe06d3ce1"
    ),
    "transformer-report.json": ("8cf78bf55a6bbaaf30530a78099475a5fd9f9da77831d031fa1a16bf4c669511"),
}

TRAJECTORY_FIXTURE_SHA256 = {
    "image-input.safetensors": ("4bb00dbfa68cb340eaf897f67176be3996a344f5b28c5f49bfa63bddea5bdc7f"),
    "denoise-trajectories.safetensors": (
        "ab32899a70958a607071fddb962be5342ff223ff48668d3c8697937707c9de95"
    ),
    "image-report.json": ("3ab0645ef6b6acf039b08afb75f955d65c4eb85d98ede7ede7901094f9a789f0"),
}

BOUNDARY_NORMALIZED_RMSE_LIMITS = {
    "image_projection": 0.0,
    "time_embed": 0.004,
    "time_mod": 0.004,
    "text_fusion": 0.004,
    "text_projection": 0.004,
    "combined_input": 0.004,
    "rotary_cos": 1e-6,
    "rotary_sin": 1e-6,
    "output": 0.03125,
}
BLOCK_NORMALIZED_RMSE_LIMIT = 0.0625
TEACHER_OUTPUT_STEP_LIMIT = 0.02
TEACHER_OUTPUT_AGGREGATE_LIMIT = 0.015625
TEACHER_STATE_STEP_LIMIT = 0.006
TEACHER_TRAJECTORY_AGGREGATE_LIMIT = 0.003
TEACHER_FINAL_STATE_LIMIT = 0.006


def tiny_transformer_config() -> Krea2TransformerConfig:
    """Return a complete graph small enough to execute in unit tests."""
    return Krea2TransformerConfig(
        attention_head_dim=8,
        axes_dims_rope=(2, 2, 4),
        in_channels=4,
        intermediate_size=32,
        norm_eps=1e-5,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_layers=2,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        num_text_layers=3,
        rope_theta=1000.0,
        text_hidden_dim=16,
        text_intermediate_size=24,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        timestep_embed_dim=8,
    )


def official_transformer_config() -> Krea2TransformerConfig:
    """Return the shared Raw/Turbo production graph declaration."""
    return Krea2TransformerConfig(
        attention_head_dim=128,
        axes_dims_rope=(32, 48, 48),
        in_channels=64,
        intermediate_size=16384,
        norm_eps=1e-5,
        num_attention_heads=48,
        num_key_value_heads=12,
        num_layers=28,
        num_layerwise_text_blocks=2,
        num_refiner_text_blocks=2,
        num_text_layers=12,
        rope_theta=1000.0,
        text_hidden_dim=2560,
        text_intermediate_size=6912,
        text_num_attention_heads=20,
        text_num_key_value_heads=20,
        timestep_embed_dim=256,
    )


def populate_transformer(model, targets, *, dtype=None) -> None:
    """Bind deterministic low-precision values to every allocation-light shell."""
    from kinomlx.modeling.layers import PretransposedLinear

    dtype = mx.float16 if dtype is None else dtype
    mx.random.seed(20260831)
    for name, target in sorted(targets.items()):
        if name.endswith((".norm.weight", "norm1.weight", "norm2.weight")):
            value = mx.zeros(target.shape, dtype=dtype)
        elif name.endswith("scale_shift_table"):
            value = mx.random.uniform(
                low=-0.02,
                high=0.02,
                shape=target.shape,
            ).astype(dtype)
        else:
            value = mx.random.uniform(
                low=-0.05,
                high=0.05,
                shape=target.shape,
            ).astype(dtype)
        if isinstance(target.owner, PretransposedLinear):
            value = mx.contiguous(value.T)
        setattr(target.owner, target.attribute, value)


__all__ = [
    "BLOCK_NORMALIZED_RMSE_LIMIT",
    "BOUNDARY_NORMALIZED_RMSE_LIMITS",
    "RAW_REVISION",
    "TEACHER_FINAL_STATE_LIMIT",
    "TEACHER_OUTPUT_AGGREGATE_LIMIT",
    "TEACHER_OUTPUT_STEP_LIMIT",
    "TEACHER_STATE_STEP_LIMIT",
    "TEACHER_TRAJECTORY_AGGREGATE_LIMIT",
    "TRAJECTORY_FIXTURE_SHA256",
    "TRANSFORMER_FIXTURE_SHA256",
    "TURBO_REVISION",
    "official_transformer_config",
    "populate_transformer",
    "tiny_transformer_config",
]
