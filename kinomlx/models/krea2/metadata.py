"""Krea 2 graph and cross-component configuration."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from kinomlx._json import JsonObject, JsonValue, load_json_object
from kinomlx.io.identity import payload_digest

SELECTED_TEXT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def _object(value: JsonValue | None, *, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _integer(value: JsonValue | None, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: JsonValue | None, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: JsonValue | None, *, name: str) -> float | None:
    return None if value is None else _number(value, name=name)


def _boolean(value: JsonValue | None, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _string(value: JsonValue | None, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer_tuple(value: JsonValue | None, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{name} must be an integer array")
    return tuple(cast(list[int], value))


def _boolean_tuple(value: JsonValue | None, *, name: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must be a boolean array")
    return tuple(cast(list[bool], value))


def _float_tuple(value: JsonValue | None, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise ValueError(f"{name} must be a numeric array")
    return tuple(_number(item, name=f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class Krea2ModelIndex:
    """Pipeline-level graph and recipe declaration."""

    is_distilled: bool
    patch_size: int
    selected_text_layers: tuple[int, ...]

    @property
    def config_digest(self) -> str:
        return _config_digest(self)


@dataclass(frozen=True)
class Krea2TransformerConfig:
    """Allocation-bearing Krea MMDiT graph facts."""

    attention_head_dim: int
    axes_dims_rope: tuple[int, ...]
    in_channels: int
    intermediate_size: int
    norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    num_layers: int
    num_layerwise_text_blocks: int
    num_refiner_text_blocks: int
    num_text_layers: int
    rope_theta: float
    text_hidden_dim: int
    text_intermediate_size: int
    text_num_attention_heads: int
    text_num_key_value_heads: int
    timestep_embed_dim: int

    @property
    def hidden_dim(self) -> int:
        return self.attention_head_dim * self.num_attention_heads

    @property
    def config_digest(self) -> str:
        return _config_digest(self)


@dataclass(frozen=True)
class Krea2TextConfig:
    """Text-only Qwen graph consumed by Krea conditioning."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool
    vocab_size: int

    @property
    def config_digest(self) -> str:
        return _config_digest(self)


@dataclass(frozen=True)
class Krea2VAEConfig:
    """Qwen Image decoder allocation and latent geometry."""

    base_dim: int
    dim_mult: tuple[int, ...]
    input_channels: int
    z_dim: int
    num_res_blocks: int
    attn_scales: tuple[int, ...]
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]

    @property
    def spatial_compression(self) -> int:
        return 1 << (len(self.dim_mult) - 1)

    @property
    def config_digest(self) -> str:
        return _config_digest(self)


@dataclass(frozen=True)
class Krea2SchedulerConfig:
    """Recipe-neutral FlowMatch scheduler configuration facts."""

    base_image_seq_len: int
    base_shift: float
    invert_sigmas: bool
    max_image_seq_len: int
    max_shift: float
    num_train_timesteps: int
    shift: float
    shift_terminal: float | None
    stochastic_sampling: bool
    time_shift_type: str
    use_beta_sigmas: bool
    use_dynamic_shifting: bool
    use_exponential_sigmas: bool
    use_karras_sigmas: bool

    @property
    def config_digest(self) -> str:
        return _config_digest(self)


type _ConfigRecord = (
    Krea2ModelIndex
    | Krea2TransformerConfig
    | Krea2TextConfig
    | Krea2VAEConfig
    | Krea2SchedulerConfig
)


def _config_digest(record: _ConfigRecord) -> str:
    return payload_digest(cast(dict[str, JsonValue], asdict(record)))


def parse_model_index(path: Path | str) -> Krea2ModelIndex:
    value = load_json_object(path)
    if value.get("_class_name") != "Krea2Pipeline":
        raise ValueError("model_index.json does not declare Krea2Pipeline")
    result = Krea2ModelIndex(
        is_distilled=_boolean(value.get("is_distilled"), name="is_distilled"),
        patch_size=_integer(value.get("patch_size"), name="patch_size"),
        selected_text_layers=_integer_tuple(
            value.get("text_encoder_select_layers"),
            name="text_encoder_select_layers",
        ),
    )
    if result.patch_size != 2:
        raise ValueError(f"unsupported Krea 2 patch_size: {result.patch_size}")
    if result.selected_text_layers != SELECTED_TEXT_LAYERS:
        raise ValueError(
            f"unsupported Krea 2 text hidden-state selection: {result.selected_text_layers}"
        )
    return result


def default_model_index(*, is_distilled: bool) -> Krea2ModelIndex:
    """Return the implemented graph when independent components have no index."""
    return Krea2ModelIndex(
        is_distilled=is_distilled,
        patch_size=2,
        selected_text_layers=SELECTED_TEXT_LAYERS,
    )


def parse_transformer_config(path: Path | str) -> Krea2TransformerConfig:
    value = load_json_object(path)
    if value.get("_class_name") != "Krea2Transformer2DModel":
        raise ValueError("transformer config does not declare Krea2Transformer2DModel")
    result = Krea2TransformerConfig(
        attention_head_dim=_integer(value.get("attention_head_dim"), name="attention_head_dim"),
        axes_dims_rope=_integer_tuple(value.get("axes_dims_rope"), name="axes_dims_rope"),
        in_channels=_integer(value.get("in_channels"), name="in_channels"),
        intermediate_size=_integer(value.get("intermediate_size"), name="intermediate_size"),
        norm_eps=_number(value.get("norm_eps"), name="norm_eps"),
        num_attention_heads=_integer(value.get("num_attention_heads"), name="num_attention_heads"),
        num_key_value_heads=_integer(value.get("num_key_value_heads"), name="num_key_value_heads"),
        num_layers=_integer(value.get("num_layers"), name="num_layers"),
        num_layerwise_text_blocks=_integer(
            value.get("num_layerwise_text_blocks"), name="num_layerwise_text_blocks"
        ),
        num_refiner_text_blocks=_integer(
            value.get("num_refiner_text_blocks"), name="num_refiner_text_blocks"
        ),
        num_text_layers=_integer(value.get("num_text_layers"), name="num_text_layers"),
        rope_theta=_number(value.get("rope_theta"), name="rope_theta"),
        text_hidden_dim=_integer(value.get("text_hidden_dim"), name="text_hidden_dim"),
        text_intermediate_size=_integer(
            value.get("text_intermediate_size"), name="text_intermediate_size"
        ),
        text_num_attention_heads=_integer(
            value.get("text_num_attention_heads"), name="text_num_attention_heads"
        ),
        text_num_key_value_heads=_integer(
            value.get("text_num_key_value_heads"), name="text_num_key_value_heads"
        ),
        timestep_embed_dim=_integer(value.get("timestep_embed_dim"), name="timestep_embed_dim"),
    )
    expected = {
        "attention_head_dim": 128,
        "axes_dims_rope": (32, 48, 48),
        "in_channels": 64,
        "intermediate_size": 16384,
        "num_attention_heads": 48,
        "num_key_value_heads": 12,
        "num_layers": 28,
        "num_layerwise_text_blocks": 2,
        "num_refiner_text_blocks": 2,
        "num_text_layers": 12,
        "text_hidden_dim": 2560,
        "text_intermediate_size": 6912,
        "text_num_attention_heads": 20,
        "text_num_key_value_heads": 20,
        "timestep_embed_dim": 256,
    }
    for name, required in expected.items():
        actual = getattr(result, name)
        if actual != required:
            raise ValueError(f"unsupported Krea 2 transformer {name}: {actual}")
    if result.norm_eps != 1e-5 or result.rope_theta != 1000.0:
        raise ValueError("unsupported Krea 2 transformer normalization or RoPE configuration")
    if sum(result.axes_dims_rope) != result.attention_head_dim:
        raise ValueError("Krea 2 RoPE axes do not sum to the attention head dimension")
    return result


def parse_text_config(path: Path | str) -> Krea2TextConfig:
    value = load_json_object(path)
    if value.get("model_type") != "qwen3_vl":
        raise ValueError("text config does not declare qwen3_vl")
    text = _object(value.get("text_config"), name="text_config")
    rope = _object(text.get("rope_parameters"), name="text_config.rope_parameters")
    if text.get("attention_bias") is not False:
        raise ValueError("unsupported Krea 2 text encoder attention_bias")
    if text.get("hidden_act") != "silu":
        raise ValueError("unsupported Krea 2 text encoder hidden_act")
    if rope.get("rope_type") != "default":
        raise ValueError("unsupported Krea 2 text encoder rope_type")
    result = Krea2TextConfig(
        hidden_size=_integer(text.get("hidden_size"), name="text_config.hidden_size"),
        intermediate_size=_integer(
            text.get("intermediate_size"), name="text_config.intermediate_size"
        ),
        num_attention_heads=_integer(
            text.get("num_attention_heads"), name="text_config.num_attention_heads"
        ),
        num_hidden_layers=_integer(
            text.get("num_hidden_layers"), name="text_config.num_hidden_layers"
        ),
        num_key_value_heads=_integer(
            text.get("num_key_value_heads"), name="text_config.num_key_value_heads"
        ),
        head_dim=_integer(text.get("head_dim"), name="text_config.head_dim"),
        rms_norm_eps=_number(text.get("rms_norm_eps"), name="text_config.rms_norm_eps"),
        rope_theta=_number(rope.get("rope_theta"), name="rope_parameters.rope_theta"),
        mrope_section=_integer_tuple(
            rope.get("mrope_section"), name="rope_parameters.mrope_section"
        ),
        mrope_interleaved=_boolean(
            rope.get("mrope_interleaved"), name="rope_parameters.mrope_interleaved"
        ),
        vocab_size=_integer(text.get("vocab_size"), name="text_config.vocab_size"),
    )
    expected = {
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "mrope_section": (24, 20, 20),
        "mrope_interleaved": True,
        "vocab_size": 151936,
    }
    for name, required in expected.items():
        actual = getattr(result, name)
        if actual != required:
            raise ValueError(f"unsupported Krea 2 text encoder {name}: {actual}")
    if result.rms_norm_eps != 1e-6 or result.rope_theta != 5_000_000.0:
        raise ValueError("unsupported Krea 2 Qwen normalization or RoPE configuration")
    return result


def parse_vae_config(path: Path | str) -> Krea2VAEConfig:
    value = load_json_object(path)
    class_name = value.get("_class_name")
    if class_name == "AutoencoderKLQwenImage":
        input_channels = _integer(value.get("input_channels"), name="input_channels")
    elif class_name == "AutoencoderKLWan":
        input_channels = _integer(value.get("in_channels"), name="in_channels")
        output_channels = _integer(value.get("out_channels"), name="out_channels")
        if output_channels not in (3, 12):
            raise ValueError(f"unsupported Krea 2 VAE out_channels: {output_channels}")
        spatial_scale = _integer(value.get("scale_factor_spatial"), name="scale_factor_spatial")
        if spatial_scale != 8:
            raise ValueError(f"unsupported Krea 2 VAE scale_factor_spatial: {spatial_scale}")
    else:
        raise ValueError("VAE config does not declare AutoencoderKLQwenImage or AutoencoderKLWan")
    result = Krea2VAEConfig(
        base_dim=_integer(value.get("base_dim"), name="base_dim"),
        dim_mult=_integer_tuple(value.get("dim_mult"), name="dim_mult"),
        input_channels=input_channels,
        z_dim=_integer(value.get("z_dim"), name="z_dim"),
        num_res_blocks=_integer(value.get("num_res_blocks"), name="num_res_blocks"),
        attn_scales=_integer_tuple(value.get("attn_scales"), name="attn_scales"),
        latent_mean=_float_tuple(value.get("latents_mean"), name="latents_mean"),
        latent_std=_float_tuple(value.get("latents_std"), name="latents_std"),
    )
    expected = {
        "base_dim": 96,
        "dim_mult": (1, 2, 4, 4),
        "input_channels": 3,
        "z_dim": 16,
        "num_res_blocks": 2,
        "attn_scales": (),
    }
    for name, required in expected.items():
        actual = getattr(result, name)
        if actual != required:
            raise ValueError(f"unsupported Krea 2 VAE {name}: {actual}")
    temporal_downsample = _boolean_tuple(
        value.get("temperal_downsample"),
        name="temperal_downsample",
    )
    if temporal_downsample != (False, True, True):
        raise ValueError(f"unsupported Krea 2 VAE temperal_downsample: {temporal_downsample}")
    dropout = _number(value.get("dropout"), name="dropout")
    if dropout != 0.0:
        raise ValueError(f"unsupported Krea 2 VAE dropout: {dropout}")
    if len(result.latent_mean) != result.z_dim or len(result.latent_std) != result.z_dim:
        raise ValueError("Krea 2 VAE latent statistics do not match z_dim")
    if any(value <= 0.0 for value in result.latent_std):
        raise ValueError("Krea 2 VAE latent standard deviations must be positive")
    return result


def parse_scheduler_config(path: Path | str) -> Krea2SchedulerConfig:
    value = load_json_object(path)
    if value.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("scheduler config does not declare FlowMatchEulerDiscreteScheduler")
    result = Krea2SchedulerConfig(
        base_image_seq_len=_integer(value.get("base_image_seq_len"), name="base_image_seq_len"),
        base_shift=_number(value.get("base_shift"), name="base_shift"),
        invert_sigmas=_boolean(value.get("invert_sigmas"), name="invert_sigmas"),
        max_image_seq_len=_integer(value.get("max_image_seq_len"), name="max_image_seq_len"),
        max_shift=_number(value.get("max_shift"), name="max_shift"),
        num_train_timesteps=_integer(value.get("num_train_timesteps"), name="num_train_timesteps"),
        shift=_number(value.get("shift"), name="shift"),
        shift_terminal=_optional_number(value.get("shift_terminal"), name="shift_terminal"),
        stochastic_sampling=_boolean(value.get("stochastic_sampling"), name="stochastic_sampling"),
        time_shift_type=_string(value.get("time_shift_type"), name="time_shift_type"),
        use_beta_sigmas=_boolean(value.get("use_beta_sigmas"), name="use_beta_sigmas"),
        use_dynamic_shifting=_boolean(
            value.get("use_dynamic_shifting"), name="use_dynamic_shifting"
        ),
        use_exponential_sigmas=_boolean(
            value.get("use_exponential_sigmas"), name="use_exponential_sigmas"
        ),
        use_karras_sigmas=_boolean(value.get("use_karras_sigmas"), name="use_karras_sigmas"),
    )
    expected = default_scheduler_config()
    for name, required in asdict(expected).items():
        actual = getattr(result, name)
        if actual != required:
            raise ValueError(f"unsupported Krea 2 scheduler {name}: {actual}")
    return result


def default_scheduler_config() -> Krea2SchedulerConfig:
    """Return the implemented scheduler graph for component-only sources."""
    return Krea2SchedulerConfig(
        base_image_seq_len=256,
        base_shift=0.5,
        invert_sigmas=False,
        max_image_seq_len=6400,
        max_shift=1.15,
        num_train_timesteps=1000,
        shift=1.0,
        shift_terminal=None,
        stochastic_sampling=False,
        time_shift_type="exponential",
        use_beta_sigmas=False,
        use_dynamic_shifting=True,
        use_exponential_sigmas=False,
        use_karras_sigmas=False,
    )


def validate_cross_component_config(
    model: Krea2ModelIndex,
    transformer: Krea2TransformerConfig,
    text: Krea2TextConfig,
    vae: Krea2VAEConfig,
) -> None:
    """Fail before model loading when graph-bearing component dimensions disagree."""
    if transformer.text_hidden_dim != text.hidden_size:
        raise ValueError(
            "Krea 2 text hidden dimension mismatch: "
            f"transformer={transformer.text_hidden_dim}, text_encoder={text.hidden_size}"
        )
    if transformer.num_text_layers != len(model.selected_text_layers):
        raise ValueError("Krea 2 transformer text-layer count does not match selected Qwen taps")
    if max(model.selected_text_layers) > text.num_hidden_layers:
        raise ValueError("Krea 2 selected Qwen hidden layer exceeds the text graph")
    expected_channels = vae.z_dim * model.patch_size**2
    if transformer.in_channels != expected_channels:
        raise ValueError(
            "Krea 2 latent channel mismatch: "
            f"transformer={transformer.in_channels}, VAE x patch={expected_channels}"
        )
    if transformer.text_hidden_dim % transformer.text_num_attention_heads:
        raise ValueError("Krea 2 text-fusion attention dimensions are not integral")


__all__ = [
    "SELECTED_TEXT_LAYERS",
    "Krea2ModelIndex",
    "Krea2SchedulerConfig",
    "Krea2TextConfig",
    "Krea2TransformerConfig",
    "Krea2VAEConfig",
    "default_model_index",
    "default_scheduler_config",
    "load_json_object",
    "parse_model_index",
    "parse_scheduler_config",
    "parse_text_config",
    "parse_transformer_config",
    "parse_vae_config",
    "validate_cross_component_config",
]
