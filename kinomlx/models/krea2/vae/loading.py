"""Exact source binding for the Krea 2 Qwen Image decoder."""

from dataclasses import dataclass

import mlx.core as mx

from kinomlx._json import JsonObject
from kinomlx.io.identity import payload_digest
from kinomlx.modeling.dtypes import resolve_float_dtype
from kinomlx.modeling.loading import WeightTarget
from kinomlx.reporting import Reporter

from ..resource_types import Krea2TensorBinding
from ..source_loading import bind_source_weights
from ..weight_keys import expected_vae_keys
from .model import (
    Krea2AttentionBlock,
    Krea2ImageDecoder,
    Krea2ResidualBlock,
    Krea2StillConv,
)

KREA2_VAE_CONSUMED_TENSORS = 104


@dataclass(frozen=True)
class Krea2VAELoadReceipt:
    """Stable identity of one exact Qwen Image decoder target graph."""

    target_count: int
    target_digest: str
    source_files: tuple[str, ...]


def _conv_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    convolution: Krea2StillConv,
) -> None:
    targets[f"{prefix}.weight"] = WeightTarget(
        convolution.conv,
        "weight",
        convolution.weight_shape,
    )
    targets[f"{prefix}.bias"] = WeightTarget(
        convolution.conv,
        "bias",
        (convolution.out_channels,),
    )


def _residual_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    residual: Krea2ResidualBlock,
) -> None:
    targets[f"{prefix}.norm1.gamma"] = WeightTarget(
        residual.norm1,
        "gamma",
        (residual.in_channels,),
    )
    targets[f"{prefix}.norm2.gamma"] = WeightTarget(
        residual.norm2,
        "gamma",
        (residual.out_channels,),
    )
    _conv_targets(targets, prefix=f"{prefix}.conv1", convolution=residual.conv1)
    _conv_targets(targets, prefix=f"{prefix}.conv2", convolution=residual.conv2)
    if residual.conv_shortcut is not None:
        _conv_targets(
            targets,
            prefix=f"{prefix}.conv_shortcut",
            convolution=residual.conv_shortcut,
        )


def _attention_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    attention: Krea2AttentionBlock,
) -> None:
    targets[f"{prefix}.norm.gamma"] = WeightTarget(
        attention.norm,
        "gamma",
        (attention.channels,),
    )
    _conv_targets(targets, prefix=f"{prefix}.to_qkv", convolution=attention.to_qkv)
    _conv_targets(targets, prefix=f"{prefix}.proj", convolution=attention.proj)


def _targets(model: Krea2ImageDecoder) -> dict[str, WeightTarget]:
    targets: dict[str, WeightTarget] = {}
    _conv_targets(targets, prefix="post_quant_conv", convolution=model.post_quant_conv)
    _conv_targets(targets, prefix="decoder.conv_in", convolution=model.decoder.conv_in)
    for index, residual in enumerate(model.decoder.mid_block.resnets):
        _residual_targets(
            targets,
            prefix=f"decoder.mid_block.resnets.{index}",
            residual=residual,
        )
    _attention_targets(
        targets,
        prefix="decoder.mid_block.attentions.0",
        attention=model.decoder.mid_block.attentions[0],
    )
    for block_index, block in enumerate(model.decoder.up_blocks):
        for residual_index, residual in enumerate(block.resnets):
            _residual_targets(
                targets,
                prefix=f"decoder.up_blocks.{block_index}.resnets.{residual_index}",
                residual=residual,
            )
        for upsample_index, upsample in enumerate(block.upsamplers):
            _conv_targets(
                targets,
                prefix=(f"decoder.up_blocks.{block_index}.upsamplers.{upsample_index}.resample.1"),
                convolution=upsample.resample,
            )
    targets["decoder.norm_out.gamma"] = WeightTarget(
        model.decoder.norm_out,
        "gamma",
        (model.decoder.norm_out.channels,),
    )
    _conv_targets(targets, prefix="decoder.conv_out", convolution=model.decoder.conv_out)

    expected = expected_vae_keys(model.config)
    if frozenset(targets) != expected:
        missing = sorted(expected - targets.keys())
        extra = sorted(targets.keys() - expected)
        raise RuntimeError(
            "Krea VAE target graph disagrees with the canonical weight mapping: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return targets


def load_krea2_vae_weights(
    model: Krea2ImageDecoder,
    *,
    expected_dtype: str | mx.Dtype | None = None,
    source_bindings: tuple[Krea2TensorBinding, ...],
    reporter: Reporter | None = None,
) -> Krea2VAELoadReceipt:
    """Bind every canonical still-decoder source tensor exactly once."""
    targets = _targets(model)
    if expected_dtype is None:
        raise ValueError("Krea 2 VAE loading requires the component dtype policy")
    # Apply the selected VAE dtype policy lazily.
    bound, files = bind_source_weights(
        targets,
        source_bindings,
        dtype=resolve_float_dtype(expected_dtype, owner="Krea 2 VAE"),
        phase="bind VAE weights",
        reporter=reporter,
    )
    if bound != len(targets):
        raise RuntimeError(f"Krea VAE loader bound {bound} of {len(targets)} targets")
    target_payload: JsonObject = {
        key: list(target.shape) for key, target in sorted(targets.items())
    }
    receipt = Krea2VAELoadReceipt(
        target_count=bound,
        target_digest=payload_digest(target_payload),
        source_files=tuple(path.name for path in files),
    )
    object.__setattr__(model, "load_receipt", receipt)
    return receipt


__all__ = [
    "KREA2_VAE_CONSUMED_TENSORS",
    "Krea2VAELoadReceipt",
    "load_krea2_vae_weights",
]
