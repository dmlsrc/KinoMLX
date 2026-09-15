"""Fail-closed admission for the currently supported Klein graph configurations."""

from dataclasses import dataclass
from pathlib import Path

from kinomlx._json import JsonObject, load_json_object
from kinomlx.io.identity import payload_digest

TEXT_TAPS = (9, 18, 27)


def validate_model_index(path: Path, *, is_distilled: bool) -> None:
    index = load_json_object(path)
    if index.get("_class_name") not in {"Flux2KleinPipeline", "Flux2KleinKVPipeline"}:
        raise ValueError("source is not a Klein pipeline")
    # Early base snapshots omit this optional setting; the selected profile
    # supplies it when absent.
    if "is_distilled" in index and index["is_distilled"] is not is_distilled:
        raise ValueError("source is_distilled contradicts the selected Klein variant")


@dataclass(frozen=True)
class Flux2TextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    head_dim: int = 128
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 36
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0


EXPECTED: dict[str, JsonObject] = {
    "text_encoder": {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "attention_bias": False,
        "hidden_act": "silu",
        "rope_scaling": None,
        "sliding_window": None,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    },
    "transformer": {
        "_class_name": "Flux2Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [32, 32, 32, 32],
        "eps": 1e-6,
        "guidance_embeds": False,
        "in_channels": 128,
        "joint_attention_dim": 12288,
        "mlp_ratio": 3.0,
        "num_attention_heads": 32,
        "num_layers": 8,
        "num_single_layers": 24,
        "out_channels": None,
        "patch_size": 1,
        "rope_theta": 2000,
        "timestep_guidance_channels": 256,
    },
    "vae": {
        "_class_name": "AutoencoderKLFlux2",
        "act_fn": "silu",
        "batch_norm_eps": 1e-4,
        "block_out_channels": [128, 256, 512, 512],
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 32,
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "patch_size": [2, 2],
        "use_post_quant_conv": True,
        "use_quant_conv": True,
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
    },
}

SCHEDULER: JsonObject = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
    "invert_sigmas": False,
    "stochastic_sampling": False,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
    "shift_terminal": None,
}


def validate_config(kind: str, path: Path) -> str:
    config = load_json_object(path)
    expected = SCHEDULER if kind == "scheduler" else EXPECTED[kind]
    for key, value in expected.items():
        if (
            key not in config
            or config[key] != value
            or (isinstance(config[key], bool) != isinstance(value, bool))
        ):
            raise ValueError(
                f"unsupported Klein {kind} {key}: expected {value!r}, got {config.get(key)!r}"
            )
    if (
        kind == "text_encoder"
        and config.get("layer_types", ["full_attention"] * 36) != ["full_attention"] * 36
    ):
        raise ValueError("Klein Qwen3 requires full attention in every layer")
    return payload_digest(expected)
