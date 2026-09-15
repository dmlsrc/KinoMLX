"""Krea 2 consumed-target selection and native/Diffusers key normalization."""

from pathlib import Path

import pytest

from kinomlx.models.krea2.metadata import parse_transformer_config, parse_vae_config
from kinomlx.models.krea2.weight_keys import (
    KREA2_TRANSFORMER_PRETRANSPOSE,
    expected_transformer_keys,
    expected_vae_keys,
    native_wan_vae_target,
    transformer_bindings,
    vae_bindings,
)

from ._fixtures import make_snapshot

_MAIN_INVERSE = {
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.to_gate.weight": "attn.gate.weight",
    "attn.to_k.weight": "attn.wk.weight",
    "attn.to_out.0.weight": "attn.wo.weight",
    "attn.to_q.weight": "attn.wq.weight",
    "attn.to_v.weight": "attn.wv.weight",
    "ff.down.weight": "mlp.down.weight",
    "ff.gate.weight": "mlp.gate.weight",
    "ff.up.weight": "mlp.up.weight",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
    "scale_shift_table": "mod.lin",
}
_BASIC_INVERSE = {
    "final_layer.linear.bias": "last.linear.bias",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.scale_shift_table": "last.modulation.lin",
    "img_in.bias": "first.bias",
    "img_in.weight": "first.weight",
    "text_fusion.projector.weight": "txtfusion.projector.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.norm.weight": "txtmlp.0.scale",
}
_VAE_RESIDUAL_INVERSE = {
    "norm1.gamma": "0.gamma",
    "conv1.bias": "2.bias",
    "conv1.weight": "2.weight",
    "norm2.gamma": "3.gamma",
    "conv2.bias": "6.bias",
    "conv2.weight": "6.weight",
}
_VAE_UP_RESIDUAL_INVERSE = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 8,
    (2, 1): 9,
    (2, 2): 10,
    (3, 0): 12,
    (3, 1): 13,
    (3, 2): 14,
}
_VAE_UP_SAMPLE_INVERSE = {0: 3, 1: 7, 2: 11}


def _native_key(canonical: str) -> str:
    if canonical.startswith("transformer_blocks."):
        _, index, tail = canonical.split(".", 2)
        return f"blocks.{index}.{_MAIN_INVERSE[tail]}"
    for family in ("layerwise_blocks", "refiner_blocks"):
        prefix = f"text_fusion.{family}."
        if canonical.startswith(prefix):
            index, tail = canonical.removeprefix(prefix).split(".", 1)
            return f"txtfusion.{family}.{index}.{_MAIN_INVERSE[tail]}"
    return _BASIC_INVERSE[canonical]


def _native_vae_key(canonical: str) -> str:
    direct = {
        "post_quant_conv.bias": "conv2.bias",
        "post_quant_conv.weight": "conv2.weight",
        "decoder.conv_in.bias": "decoder.conv1.bias",
        "decoder.conv_in.weight": "decoder.conv1.weight",
        "decoder.norm_out.gamma": "decoder.head.0.gamma",
        "decoder.conv_out.bias": "decoder.head.2.bias",
        "decoder.conv_out.weight": "decoder.head.2.weight",
    }
    if canonical in direct:
        return direct[canonical]
    mid_residual = "decoder.mid_block.resnets."
    if canonical.startswith(mid_residual):
        index, tail = canonical.removeprefix(mid_residual).split(".", 1)
        source_index = {"0": 0, "1": 2}[index]
        return f"decoder.middle.{source_index}.residual.{_VAE_RESIDUAL_INVERSE[tail]}"
    mid_attention = "decoder.mid_block.attentions.0."
    if canonical.startswith(mid_attention):
        return f"decoder.middle.1.{canonical.removeprefix(mid_attention)}"
    up_blocks = "decoder.up_blocks."
    if canonical.startswith(up_blocks):
        block_text, remainder = canonical.removeprefix(up_blocks).split(".", 1)
        block = int(block_text)
        if remainder.startswith("resnets."):
            residual_text, tail = remainder.removeprefix("resnets.").split(".", 1)
            source_index = _VAE_UP_RESIDUAL_INVERSE[(block, int(residual_text))]
            if tail.startswith("conv_shortcut."):
                return (
                    f"decoder.upsamples.{source_index}.shortcut."
                    f"{tail.removeprefix('conv_shortcut.')}"
                )
            return f"decoder.upsamples.{source_index}.residual.{_VAE_RESIDUAL_INVERSE[tail]}"
        source_index = _VAE_UP_SAMPLE_INVERSE[block]
        return f"decoder.upsamples.{source_index}.{remainder.removeprefix('upsamplers.0.')}"
    raise AssertionError(f"no original Wan VAE key for {canonical}")


def test_native_monolith_maps_every_target_once_and_reshapes_main_modulation(
    tmp_path: Path,
) -> None:
    source = make_snapshot(tmp_path / "source")
    config = parse_transformer_config(source / "transformer" / "config.json")
    canonical = sorted(expected_transformer_keys(config))
    native = tuple(_native_key(key) for key in canonical)

    bindings = transformer_bindings(native, config)

    assert len(bindings) == 430
    assert {binding.target_key for binding in bindings} == set(canonical)
    modulation = next(
        binding
        for binding in bindings
        if binding.target_key == "transformer_blocks.0.scale_shift_table"
    )
    assert modulation.source_key == "blocks.0.mod.lin"
    assert modulation.reshape_rows == 6
    transposed = {binding.target_key for binding in bindings if binding.transform is not None}
    assert transposed == {f"transformer_blocks.{index}.ff.down.weight" for index in range(28)}
    assert all(
        binding.transform == KREA2_TRANSFORMER_PRETRANSPOSE
        for binding in bindings
        if binding.target_key in transposed
    )


def test_transformer_rejects_cross_layout_aliases_and_missing_consumed_targets(
    tmp_path: Path,
) -> None:
    source = make_snapshot(tmp_path / "source")
    config = parse_transformer_config(source / "transformer" / "config.json")
    canonical = tuple(sorted(expected_transformer_keys(config)))
    native = tuple(_native_key(key) for key in canonical)

    with pytest.raises(ValueError, match="mixes native and Diffusers key aliases"):
        transformer_bindings((*native, canonical[0]), config)
    with pytest.raises(ValueError, match="missing consumed tensor"):
        transformer_bindings(canonical[1:], config)


def test_vae_source_selects_exact_still_graph_and_rejects_missing_targets(
    tmp_path: Path,
) -> None:
    source = make_snapshot(tmp_path / "source")
    config = parse_vae_config(source / "vae" / "config.json")
    expected = expected_vae_keys(config)
    bindings = vae_bindings(
        (
            *sorted(expected),
            "encoder.conv_in.weight",
            "quant_conv.weight",
            "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
        ),
        config,
    )
    assert {binding.target_key for binding in bindings} == expected
    assert not any("time_conv" in binding.target_key for binding in bindings)
    assert len(bindings) == 104
    missing = min(expected)
    with pytest.raises(ValueError, match=missing.replace(".", r"\.")):
        vae_bindings(tuple(sorted(expected - {missing})), config)


def test_original_wan_vae_names_map_to_the_same_exact_still_graph(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source")
    config = parse_vae_config(source / "vae" / "config.json")
    expected = expected_vae_keys(config)
    native = tuple(_native_vae_key(key) for key in sorted(expected))

    bindings = vae_bindings(
        (*native, "encoder.conv1.weight", "decoder.upsamples.3.time_conv.2.weight"),
        config,
    )

    assert len(bindings) == 104
    assert {binding.target_key for binding in bindings} == expected
    assert all(
        native_wan_vae_target(binding.source_key) == binding.target_key for binding in bindings
    )
    assert not any("time_conv" in binding.source_key for binding in bindings)
    with pytest.raises(ValueError, match="mixes original Wan and Diffusers"):
        vae_bindings((*native, "decoder.conv_out.weight"), config)
