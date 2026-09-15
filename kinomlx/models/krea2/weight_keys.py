"""Krea 2 source bindings and native execution layouts."""

import itertools
from dataclasses import dataclass

from .metadata import (
    Krea2ModelIndex,
    Krea2TextConfig,
    Krea2TransformerConfig,
    Krea2VAEConfig,
)

KREA2_TRANSFORMER_WEIGHT_LAYOUT = "main-ff-down-io-v1"
KREA2_TRANSFORMER_PRETRANSPOSE = "linear-io"
KREA2_VAE_WEIGHT_LAYOUT = "still-current-frame-ohwi-v1"
KREA2_VAE_CONV2D_OHWI = "conv2d-ohwi"
KREA2_VAE_CURRENT_FRAME_OHWI = "causal-conv3d-current-frame-ohwi"
KREA2_VAE_FLATTEN = "flatten"

_MAIN_BLOCK_TAILS = (
    "attn.norm_k.weight",
    "attn.norm_q.weight",
    "attn.to_gate.weight",
    "attn.to_k.weight",
    "attn.to_out.0.weight",
    "attn.to_q.weight",
    "attn.to_v.weight",
    "ff.down.weight",
    "ff.gate.weight",
    "ff.up.weight",
    "norm1.weight",
    "norm2.weight",
    "scale_shift_table",
)
_TEXT_BLOCK_TAILS = _MAIN_BLOCK_TAILS[:-1]
_BASIC_TRANSFORMER_KEYS = (
    "final_layer.linear.bias",
    "final_layer.linear.weight",
    "final_layer.norm.weight",
    "final_layer.scale_shift_table",
    "img_in.bias",
    "img_in.weight",
    "text_fusion.projector.weight",
    "time_embed.linear_1.bias",
    "time_embed.linear_1.weight",
    "time_embed.linear_2.bias",
    "time_embed.linear_2.weight",
    "time_mod_proj.bias",
    "time_mod_proj.weight",
    "txt_in.linear_1.bias",
    "txt_in.linear_1.weight",
    "txt_in.linear_2.bias",
    "txt_in.linear_2.weight",
    "txt_in.norm.weight",
)
_TEXT_LAYER_TAILS = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
_NATIVE_BLOCK_TAIL_MAP = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "mod.lin": "scale_shift_table",
}
_NATIVE_BASIC_MAP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
}


@dataclass(frozen=True)
class KeyBinding:
    """One source tensor mapped to a logical component target."""

    source_key: str
    target_key: str
    reshape_rows: int | None = None
    transform: str | None = None


def expected_transformer_keys(config: Krea2TransformerConfig) -> frozenset[str]:
    """Return every canonical tensor target owned by the implemented graph."""
    keys = {
        f"transformer_blocks.{block}.{tail}"
        for block in range(config.num_layers)
        for tail in _MAIN_BLOCK_TAILS
    }
    keys.update(
        f"text_fusion.layerwise_blocks.{block}.{tail}"
        for block in range(config.num_layerwise_text_blocks)
        for tail in _TEXT_BLOCK_TAILS
    )
    keys.update(
        f"text_fusion.refiner_blocks.{block}.{tail}"
        for block in range(config.num_refiner_text_blocks)
        for tail in _TEXT_BLOCK_TAILS
    )
    keys.update(_BASIC_TRANSFORMER_KEYS)
    return frozenset(keys)


def transformer_weight_transform(key: str) -> str | None:
    """Select the measured main-block FF-down layout; leave text fusion alone."""
    if key.startswith("transformer_blocks.") and key.endswith(".ff.down.weight"):
        return KREA2_TRANSFORMER_PRETRANSPOSE
    return None


def native_transformer_target(name: str) -> tuple[str, int | None] | None:
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            return None
        mapped = _NATIVE_BLOCK_TAIL_MAP.get(parts[2])
        if mapped is not None:
            reshape_rows = 6 if parts[2] == "mod.lin" else None
            return f"transformer_blocks.{parts[1]}.{mapped}", reshape_rows
    for family in ("layerwise_blocks", "refiner_blocks"):
        prefix = f"txtfusion.{family}."
        if name.startswith(prefix):
            remainder = name.removeprefix(prefix)
            index, separator, tail = remainder.partition(".")
            if separator and index.isdigit():
                mapped = _NATIVE_BLOCK_TAIL_MAP.get(tail)
                if mapped is not None:
                    return f"text_fusion.{family}.{index}.{mapped}", None
    mapped = _NATIVE_BASIC_MAP.get(name)
    return None if mapped is None else (mapped, None)


def transformer_bindings(
    source_keys: tuple[str, ...],
    config: Krea2TransformerConfig,
) -> tuple[KeyBinding, ...]:
    """Normalize native or Diffusers keys and reject cross-layout aliases."""
    expected = expected_transformer_keys(config)
    native = any(key.startswith(("blocks.", "txtfusion.", "txtmlp.")) for key in source_keys)
    canonical = any(key in expected for key in source_keys)
    if native and canonical:
        raise ValueError("Krea 2 transformer mixes native and Diffusers key aliases")
    if not native and not canonical:
        raise ValueError("Krea 2 transformer has no recognized native or Diffusers tensors")

    by_target: dict[str, KeyBinding] = {}
    for source_key in source_keys:
        if native:
            mapped = native_transformer_target(source_key)
            if mapped is None:
                continue
            target_key, reshape_rows = mapped
        else:
            if source_key not in expected:
                continue
            target_key = source_key
            reshape_rows = None
        if target_key not in expected:
            continue
        if target_key in by_target:
            previous = by_target[target_key]
            raise ValueError(
                f"Krea 2 transformer aliases {previous.source_key!r} and "
                f"{source_key!r} onto {target_key!r}"
            )
        by_target[target_key] = KeyBinding(
            source_key, target_key, reshape_rows, transformer_weight_transform(target_key)
        )
    missing = sorted(expected - by_target.keys())
    if missing:
        raise ValueError(f"Krea 2 transformer is missing consumed tensor {missing[0]!r}")
    return tuple(by_target[key] for key in sorted(by_target))


def transformer_source_bindings(config: Krea2TransformerConfig) -> dict[str, KeyBinding]:
    """Enumerate source spellings and layout facts for sparse adapters."""
    expected = expected_transformer_keys(config)
    aliases = {name: KeyBinding(name, name) for name in expected}
    for name in expected:
        if ".attn.to_out.0." in name:
            alias = name.replace(".attn.to_out.0.", ".attn.to_out.")
            aliases[alias] = KeyBinding(alias, name)
    aliases.update(
        {source: KeyBinding(source, target) for source, target in _NATIVE_BASIC_MAP.items()}
    )
    groups = (
        ("blocks", config.num_layers),
        ("txtfusion.layerwise_blocks", config.num_layerwise_text_blocks),
        ("txtfusion.refiner_blocks", config.num_refiner_text_blocks),
    )
    for prefix, count in groups:
        for index in range(count):
            for tail in _NATIVE_BLOCK_TAIL_MAP:
                source = f"{prefix}.{index}.{tail}"
                mapped = native_transformer_target(source)
                if mapped is not None and mapped[0] in expected:
                    aliases[source] = KeyBinding(source, mapped[0], mapped[1])
    return aliases


def text_encoder_bindings(
    source_keys: tuple[str, ...],
    config: Krea2TextConfig,
    model: Krea2ModelIndex,
) -> tuple[KeyBinding, ...]:
    """Select only Qwen layers needed to produce Krea's hidden-state taps."""
    executed_layers = max(model.selected_text_layers)
    if executed_layers > config.num_hidden_layers:
        raise ValueError("selected Krea 2 text layer exceeds the Qwen source graph")
    expected = {"language_model.embed_tokens.weight"}
    expected.update(
        f"language_model.layers.{layer}.{tail}"
        for layer in range(executed_layers)
        for tail in _TEXT_LAYER_TAILS
    )
    available = set(source_keys)
    missing = sorted(expected - available)
    if missing:
        raise ValueError(f"Krea 2 text encoder is missing consumed tensor {missing[0]!r}")
    return tuple(
        KeyBinding(source_key=key, target_key=key.removeprefix("language_model."))
        for key in sorted(expected)
    )


def _vae_residual_keys(prefix: str, *, shortcut: bool) -> set[str]:
    keys = {
        f"{prefix}.conv1.bias",
        f"{prefix}.conv1.weight",
        f"{prefix}.conv2.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.norm1.gamma",
        f"{prefix}.norm2.gamma",
    }
    if shortcut:
        keys.update(
            {
                f"{prefix}.conv_shortcut.bias",
                f"{prefix}.conv_shortcut.weight",
            }
        )
    return keys


def expected_vae_keys(config: Krea2VAEConfig) -> frozenset[str]:
    """Return the exact Qwen Image tensors consumed by the still decoder."""
    if not config.dim_mult:
        raise ValueError("Krea 2 VAE dim_mult must not be empty")
    bottleneck = config.base_dim * config.dim_mult[-1]
    keys = {
        "post_quant_conv.bias",
        "post_quant_conv.weight",
        "decoder.conv_in.bias",
        "decoder.conv_in.weight",
        "decoder.conv_out.bias",
        "decoder.conv_out.weight",
        "decoder.norm_out.gamma",
        "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.mid_block.attentions.0.proj.bias",
        "decoder.mid_block.attentions.0.proj.weight",
        "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.mid_block.attentions.0.to_qkv.weight",
    }
    keys.update(_vae_residual_keys("decoder.mid_block.resnets.0", shortcut=False))
    keys.update(_vae_residual_keys("decoder.mid_block.resnets.1", shortcut=False))

    dims = [bottleneck, *(config.base_dim * value for value in reversed(config.dim_mult))]
    for block_index, (declared_in, out_dim) in enumerate(itertools.pairwise(dims)):
        in_dim = declared_in // 2 if block_index > 0 else declared_in
        for residual_index in range(config.num_res_blocks + 1):
            keys.update(
                _vae_residual_keys(
                    f"decoder.up_blocks.{block_index}.resnets.{residual_index}",
                    shortcut=in_dim != out_dim,
                )
            )
            in_dim = out_dim
        if block_index != len(config.dim_mult) - 1:
            keys.update(
                {
                    f"decoder.up_blocks.{block_index}.upsamplers.0.resample.1.bias",
                    f"decoder.up_blocks.{block_index}.upsamplers.0.resample.1.weight",
                }
            )
    return frozenset(keys)


def _vae_transform(key: str) -> str | None:
    if key.endswith(".gamma"):
        return KREA2_VAE_FLATTEN
    if not key.endswith(".weight"):
        return None
    if ".attentions." in key or ".upsamplers.0.resample.1." in key:
        return KREA2_VAE_CONV2D_OHWI
    return KREA2_VAE_CURRENT_FRAME_OHWI


_WAN_VAE_RESIDUAL_TAILS = {
    "0.gamma": "norm1.gamma",
    "2.bias": "conv1.bias",
    "2.weight": "conv1.weight",
    "3.gamma": "norm2.gamma",
    "6.bias": "conv2.bias",
    "6.weight": "conv2.weight",
}
_WAN_VAE_UP_RESIDUAL_INDEX = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    8: (2, 0),
    9: (2, 1),
    10: (2, 2),
    12: (3, 0),
    13: (3, 1),
    14: (3, 2),
}
_WAN_VAE_UP_SAMPLE_INDEX = {3: 0, 7: 1, 11: 2}


def native_wan_vae_target(key: str) -> str | None:
    """Map one original Wan VAE decoder key to the canonical weight graph."""
    if key == "conv2.weight":
        return "post_quant_conv.weight"
    if key == "conv2.bias":
        return "post_quant_conv.bias"
    if key == "decoder.conv1.weight":
        return "decoder.conv_in.weight"
    if key == "decoder.conv1.bias":
        return "decoder.conv_in.bias"
    if key == "decoder.head.0.gamma":
        return "decoder.norm_out.gamma"
    if key == "decoder.head.2.weight":
        return "decoder.conv_out.weight"
    if key == "decoder.head.2.bias":
        return "decoder.conv_out.bias"

    for source_index, target_index in ((0, 0), (2, 1)):
        prefix = f"decoder.middle.{source_index}.residual."
        if key.startswith(prefix):
            tail = _WAN_VAE_RESIDUAL_TAILS.get(key.removeprefix(prefix))
            return None if tail is None else f"decoder.mid_block.resnets.{target_index}.{tail}"
    attention_prefix = "decoder.middle.1."
    if key.startswith(attention_prefix):
        tail = key.removeprefix(attention_prefix)
        if tail in {
            "norm.gamma",
            "proj.bias",
            "proj.weight",
            "to_qkv.bias",
            "to_qkv.weight",
        }:
            return f"decoder.mid_block.attentions.0.{tail}"

    upsample_prefix = "decoder.upsamples."
    if not key.startswith(upsample_prefix):
        return None
    remainder = key.removeprefix(upsample_prefix)
    index_text, separator, tail = remainder.partition(".")
    if not separator or not index_text.isdigit():
        return None
    index = int(index_text)
    residual_prefix = "residual."
    if tail.startswith(residual_prefix):
        location = _WAN_VAE_UP_RESIDUAL_INDEX.get(index)
        mapped_tail = _WAN_VAE_RESIDUAL_TAILS.get(tail.removeprefix(residual_prefix))
        if location is None or mapped_tail is None:
            return None
        block, residual = location
        return f"decoder.up_blocks.{block}.resnets.{residual}.{mapped_tail}"
    shortcut_prefix = "shortcut."
    if index == 4 and tail.startswith(shortcut_prefix):
        parameter = tail.removeprefix(shortcut_prefix)
        if parameter in {"bias", "weight"}:
            return f"decoder.up_blocks.1.resnets.0.conv_shortcut.{parameter}"
    if index in _WAN_VAE_UP_SAMPLE_INDEX and tail.startswith("resample.1."):
        parameter = tail.removeprefix("resample.1.")
        if parameter in {"bias", "weight"}:
            block = _WAN_VAE_UP_SAMPLE_INDEX[index]
            return f"decoder.up_blocks.{block}.upsamplers.0.resample.1.{parameter}"
    return None


def vae_bindings(
    source_keys: tuple[str, ...],
    config: Krea2VAEConfig,
) -> tuple[KeyBinding, ...]:
    """Bind Diffusers or original Wan names while dropping temporal baggage."""
    expected = expected_vae_keys(config)
    available = set(source_keys)
    canonical = expected & available
    native_by_target: dict[str, str] = {}
    for source_key in source_keys:
        target_key = native_wan_vae_target(source_key)
        if target_key is None or target_key not in expected:
            continue
        previous = native_by_target.get(target_key)
        if previous is not None:
            raise ValueError(
                f"Krea 2 VAE aliases {previous!r} and {source_key!r} onto {target_key!r}"
            )
        native_by_target[target_key] = source_key
    if canonical and native_by_target:
        raise ValueError("Krea 2 VAE mixes original Wan and Diffusers key aliases")
    if native_by_target:
        by_target = native_by_target
    elif canonical:
        by_target = {key: key for key in canonical}
    else:
        raise ValueError("Krea 2 VAE has no recognized original Wan or Diffusers decoder tensors")
    missing = sorted(expected - by_target.keys())
    if missing:
        raise ValueError(f"Krea 2 VAE is missing consumed decoder tensor {missing[0]!r}")
    return tuple(
        KeyBinding(
            source_key=by_target[target_key],
            target_key=target_key,
            transform=_vae_transform(target_key),
        )
        for target_key in sorted(expected)
    )


__all__ = [
    "KREA2_TRANSFORMER_PRETRANSPOSE",
    "KREA2_TRANSFORMER_WEIGHT_LAYOUT",
    "KREA2_VAE_CONV2D_OHWI",
    "KREA2_VAE_CURRENT_FRAME_OHWI",
    "KREA2_VAE_FLATTEN",
    "KREA2_VAE_WEIGHT_LAYOUT",
    "KeyBinding",
    "expected_transformer_keys",
    "expected_vae_keys",
    "native_wan_vae_target",
    "text_encoder_bindings",
    "transformer_bindings",
    "transformer_weight_transform",
    "vae_bindings",
]
