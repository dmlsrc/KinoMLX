"""Exact source-checkpoint binding for the first-frame Qwen Image encoder."""

from pathlib import Path

import mlx.core as mx

from kinomlx._json import JsonObject
from kinomlx.io.identity import payload_digest
from kinomlx.io.safetensors import load_weights
from kinomlx.modeling.loading import WeightTarget, resolve_weight_bindings, weight_files

from ..preparation import prepare_weight
from ..resource_types import Krea2TensorBinding
from ..weight_keys import (
    KREA2_VAE_CONV2D_OHWI,
    KREA2_VAE_CURRENT_FRAME_OHWI,
    KREA2_VAE_FLATTEN,
)
from .encoder import Krea2Downsample, Krea2ImageEncoder
from .loading import Krea2VAELoadReceipt, _attention_targets, _conv_targets, _residual_targets


def encoder_weight_targets(model: Krea2ImageEncoder) -> dict[str, WeightTarget]:
    """Enumerate the native encoder graph, excluding unused temporal kernels."""
    targets: dict[str, WeightTarget] = {}
    core = model.encoder
    _conv_targets(targets, prefix="encoder.conv_in", convolution=core.conv_in)
    for index, block in enumerate(core.down_blocks):
        prefix = f"encoder.down_blocks.{index}"
        if isinstance(block, Krea2Downsample):
            _conv_targets(targets, prefix=f"{prefix}.resample.1", convolution=block.resample)
        else:
            _residual_targets(targets, prefix=prefix, residual=block)
    for index, residual in enumerate(core.mid_block.resnets):
        _residual_targets(targets, prefix=f"encoder.mid_block.resnets.{index}", residual=residual)
    _attention_targets(
        targets, prefix="encoder.mid_block.attentions.0", attention=core.mid_block.attentions[0]
    )
    targets["encoder.norm_out.gamma"] = WeightTarget(
        core.norm_out, "gamma", (core.norm_out.channels,)
    )
    _conv_targets(targets, prefix="encoder.conv_out", convolution=core.conv_out)
    _conv_targets(targets, prefix="quant_conv", convolution=model.quant_conv)
    return targets


def encoder_source_shape(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Describe source OIHW/OITHW convolution and channel-normalization layouts."""
    if key.endswith(".gamma"):
        return (*shape, 1, 1) if ".attentions." in key else (*shape, 1, 1, 1)
    if key.endswith(".weight"):
        output, height, width, inputs = shape
        if ".attentions." in key or ".resample.1." in key:
            return output, inputs, height, width
        return output, inputs, height, height, width
    return shape


def load_krea2_encoder_weights(
    model: Krea2ImageEncoder, source_path: Path | str
) -> Krea2VAELoadReceipt:
    """Bind original Qwen VAE weights after preflighting every consumed shape."""
    files = weight_files(source_path, component="Krea 2 VAE encoder source")
    targets = encoder_weight_targets(model)
    source_targets = {
        key: WeightTarget(target.owner, target.attribute, encoder_source_shape(key, target.shape))
        for key, target in targets.items()
    }
    bindings = resolve_weight_bindings(
        files, source_targets, lambda key: (key,), component="Krea 2 VAE encoder"
    )
    for path in files:
        weights = load_weights(path)
        for binding in bindings:
            if binding.source_path != path:
                continue
            value = weights[binding.source_key]
            transform = (
                KREA2_VAE_CURRENT_FRAME_OHWI
                if value.ndim == 5
                else KREA2_VAE_CONV2D_OHWI
                if binding.logical_key.endswith(".weight")
                else KREA2_VAE_FLATTEN
            )
            value = prepare_weight(
                value,
                Krea2TensorBinding(
                    path, binding.source_key, binding.logical_key, transform=transform
                ),
                dtype=model.compute_dtype,
            )
            setattr(binding.target.owner, binding.target.attribute, value)
        weights.clear()
    mx.eval(model.parameters())
    payload: JsonObject = {key: list(target.shape) for key, target in sorted(targets.items())}
    receipt = Krea2VAELoadReceipt(
        target_count=len(bindings),
        target_digest=payload_digest(payload),
        source_files=tuple(path.name for path in files),
    )
    object.__setattr__(model, "load_receipt", receipt)
    return receipt
