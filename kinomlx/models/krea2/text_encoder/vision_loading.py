"""Exact source binding for Qwen3-VL visual weights."""

from pathlib import Path

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import load_weights
from kinomlx.modeling.loading import WeightTarget, resolve_weight_bindings, weight_files

from .vision import Qwen3VLVisionModel, QwenVisionNorm


def vision_weight_targets(model: Qwen3VLVisionModel) -> dict[str, WeightTarget]:
    """Describe original checkpoint shapes, including the Conv3d patch kernel."""
    config = model.config
    width, hidden = config.hidden_size, config.intermediate_size
    targets: dict[str, WeightTarget] = {}

    def linear(prefix: str, layer: nn.Linear, inputs: int, outputs: int) -> None:
        targets[f"visual.{prefix}.weight"] = WeightTarget(layer, "weight", (outputs, inputs))
        targets[f"visual.{prefix}.bias"] = WeightTarget(layer, "bias", (outputs,))

    def norm(prefix: str, layer: QwenVisionNorm) -> None:
        targets[f"visual.{prefix}.weight"] = WeightTarget(layer, "weight", (layer.dims,))
        targets[f"visual.{prefix}.bias"] = WeightTarget(layer, "bias", (layer.dims,))

    targets["visual.patch_embed.proj.weight"] = WeightTarget(
        model.patch_projection,
        "weight",
        (width, 3, config.temporal_patch_size, config.patch_size, config.patch_size),
    )
    targets["visual.patch_embed.proj.bias"] = WeightTarget(model.patch_projection, "bias", (width,))
    targets["visual.pos_embed.weight"] = WeightTarget(
        model.pos_embed, "weight", (config.num_position_embeddings, width)
    )
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}"
        norm(f"{prefix}.norm1", block.norm1)
        norm(f"{prefix}.norm2", block.norm2)
        linear(f"{prefix}.attn.qkv", block.attn.qkv, width, 3 * width)
        linear(f"{prefix}.attn.proj", block.attn.proj, width, width)
        linear(f"{prefix}.mlp.linear_fc1", block.mlp.linear_fc1, width, hidden)
        linear(f"{prefix}.mlp.linear_fc2", block.mlp.linear_fc2, hidden, width)
    mergers = [("merger", model.merger)] + [
        (f"deepstack_merger_list.{index}", merger)
        for index, merger in enumerate(model.deepstack_merger_list)
    ]
    for prefix, merger in mergers:
        norm(f"{prefix}.norm", merger.norm)
        linear(f"{prefix}.linear_fc1", merger.linear_fc1, merger.width, merger.width)
        linear(f"{prefix}.linear_fc2", merger.linear_fc2, merger.width, config.out_hidden_size)
    return targets


def load_qwen3_vl_vision_weights(
    model: Qwen3VLVisionModel, source_path: Path | str, *, dtype: mx.Dtype = mx.bfloat16
) -> int:
    """Load only visual targets from a Qwen checkpoint after complete preflight."""
    files = weight_files(source_path, component="Qwen3-VL vision")
    targets = vision_weight_targets(model)
    bindings = resolve_weight_bindings(
        files, targets, lambda key: (key,), component="Qwen3-VL vision"
    )
    for path in files:
        weights = load_weights(path)
        for binding in bindings:
            if binding.source_path != path:
                continue
            value = weights[binding.source_key]
            if binding.logical_key == "visual.patch_embed.proj.weight":
                value = value.reshape(value.shape[0], -1)
            setattr(binding.target.owner, binding.target.attribute, value.astype(dtype))
        weights.clear()
    mx.eval(model.parameters())
    return len(bindings)
