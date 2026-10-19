"""Exact source and prepared-weight binding for the Krea 2 MMDiT graph."""

from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx._json import JsonObject
from kinomlx.io.identity import payload_digest
from kinomlx.modeling.dtypes import FloatDType, resolve_float_dtype
from kinomlx.modeling.loading import (
    WeightTarget,
)
from kinomlx.reporting import Reporter

from ..resource_types import Krea2TensorBinding
from ..settings import REFERENCE_FP16_ERROR
from ..source_loading import bind_source_weights
from ..weight_keys import KREA2_TRANSFORMER_PRETRANSPOSE, expected_transformer_keys
from .model import (
    Krea2Attention,
    Krea2SwiGLU,
    Krea2TextFusionBlock,
    Krea2Transformer,
    Krea2TransformerBlock,
)

KREA2_TRANSFORMER_CONSUMED_TENSORS = 430


@dataclass(frozen=True)
class Krea2TransformerLoadReceipt:
    """Stable identity of one exact transformer target graph and source set."""

    target_count: int
    target_digest: str
    source_files: tuple[str, ...]


def _attention_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    attention: Krea2Attention,
) -> None:
    hidden_size = attention.hidden_size
    kv_size = attention.num_kv_heads * attention.head_dim
    targets.update(
        {
            f"{prefix}attn.norm_k.weight": WeightTarget(
                attention.norm_k,
                "weight",
                (attention.head_dim,),
            ),
            f"{prefix}attn.norm_q.weight": WeightTarget(
                attention.norm_q,
                "weight",
                (attention.head_dim,),
            ),
            f"{prefix}attn.to_gate.weight": WeightTarget(
                attention.to_gate,
                "weight",
                (hidden_size, hidden_size),
            ),
            f"{prefix}attn.to_k.weight": WeightTarget(
                attention.to_k,
                "weight",
                (kv_size, hidden_size),
            ),
            f"{prefix}attn.to_out.0.weight": WeightTarget(
                attention.to_out,
                "weight",
                (hidden_size, hidden_size),
            ),
            f"{prefix}attn.to_q.weight": WeightTarget(
                attention.to_q,
                "weight",
                (hidden_size, hidden_size),
            ),
            f"{prefix}attn.to_v.weight": WeightTarget(
                attention.to_v,
                "weight",
                (kv_size, hidden_size),
            ),
        }
    )


def _feed_forward_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    feed_forward: Krea2SwiGLU,
    intermediate_size: int,
) -> None:
    hidden_size = feed_forward.hidden_size
    targets.update(
        {
            f"{prefix}ff.down.weight": WeightTarget(
                feed_forward.down,
                "weight",
                (hidden_size, intermediate_size),
            ),
            f"{prefix}ff.gate.weight": WeightTarget(
                feed_forward.gate,
                "weight",
                (intermediate_size, hidden_size),
            ),
            f"{prefix}ff.up.weight": WeightTarget(
                feed_forward.up,
                "weight",
                (intermediate_size, hidden_size),
            ),
        }
    )


def _text_block_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    block: Krea2TextFusionBlock,
    intermediate_size: int,
) -> None:
    _attention_targets(targets, prefix=prefix, attention=block.attn)
    _feed_forward_targets(
        targets,
        prefix=prefix,
        feed_forward=block.ff,
        intermediate_size=intermediate_size,
    )
    hidden_size = block.attn.hidden_size
    targets.update(
        {
            f"{prefix}norm1.weight": WeightTarget(block.norm1, "weight", (hidden_size,)),
            f"{prefix}norm2.weight": WeightTarget(block.norm2, "weight", (hidden_size,)),
        }
    )


def _main_block_targets(
    targets: dict[str, WeightTarget],
    *,
    prefix: str,
    block: Krea2TransformerBlock,
    intermediate_size: int,
) -> None:
    _attention_targets(targets, prefix=prefix, attention=block.attn)
    _feed_forward_targets(
        targets,
        prefix=prefix,
        feed_forward=block.ff,
        intermediate_size=intermediate_size,
    )
    hidden_size = block.attn.hidden_size
    targets.update(
        {
            f"{prefix}norm1.weight": WeightTarget(block.norm1, "weight", (hidden_size,)),
            f"{prefix}norm2.weight": WeightTarget(block.norm2, "weight", (hidden_size,)),
            f"{prefix}scale_shift_table": WeightTarget(
                block,
                "scale_shift_table",
                (6, hidden_size),
            ),
        }
    )


def transformer_weight_targets(model: Krea2Transformer) -> dict[str, WeightTarget]:
    config = model.config
    hidden_size = config.hidden_dim
    targets: dict[str, WeightTarget] = {
        "final_layer.linear.bias": WeightTarget(
            model.final_layer.linear,
            "bias",
            (config.in_channels,),
        ),
        "final_layer.linear.weight": WeightTarget(
            model.final_layer.linear,
            "weight",
            (config.in_channels, hidden_size),
        ),
        "final_layer.norm.weight": WeightTarget(
            model.final_layer.norm,
            "weight",
            (hidden_size,),
        ),
        "final_layer.scale_shift_table": WeightTarget(
            model.final_layer,
            "scale_shift_table",
            (2, hidden_size),
        ),
        "img_in.bias": WeightTarget(model.img_in, "bias", (hidden_size,)),
        "img_in.weight": WeightTarget(
            model.img_in,
            "weight",
            (hidden_size, config.in_channels),
        ),
        "text_fusion.projector.weight": WeightTarget(
            model.text_fusion.projector,
            "weight",
            (1, config.num_text_layers),
        ),
        "time_embed.linear_1.bias": WeightTarget(
            model.time_embed.linear_1,
            "bias",
            (hidden_size,),
        ),
        "time_embed.linear_1.weight": WeightTarget(
            model.time_embed.linear_1,
            "weight",
            (hidden_size, config.timestep_embed_dim),
        ),
        "time_embed.linear_2.bias": WeightTarget(
            model.time_embed.linear_2,
            "bias",
            (hidden_size,),
        ),
        "time_embed.linear_2.weight": WeightTarget(
            model.time_embed.linear_2,
            "weight",
            (hidden_size, hidden_size),
        ),
        "time_mod_proj.bias": WeightTarget(
            model.time_mod_proj,
            "bias",
            (6 * hidden_size,),
        ),
        "time_mod_proj.weight": WeightTarget(
            model.time_mod_proj,
            "weight",
            (6 * hidden_size, hidden_size),
        ),
        "txt_in.linear_1.bias": WeightTarget(
            model.txt_in.linear_1,
            "bias",
            (hidden_size,),
        ),
        "txt_in.linear_1.weight": WeightTarget(
            model.txt_in.linear_1,
            "weight",
            (hidden_size, config.text_hidden_dim),
        ),
        "txt_in.linear_2.bias": WeightTarget(
            model.txt_in.linear_2,
            "bias",
            (hidden_size,),
        ),
        "txt_in.linear_2.weight": WeightTarget(
            model.txt_in.linear_2,
            "weight",
            (hidden_size, hidden_size),
        ),
        "txt_in.norm.weight": WeightTarget(
            model.txt_in.norm,
            "weight",
            (config.text_hidden_dim,),
        ),
    }
    for index, layerwise_block in enumerate(model.text_fusion.layerwise_blocks):
        _text_block_targets(
            targets,
            prefix=f"text_fusion.layerwise_blocks.{index}.",
            block=layerwise_block,
            intermediate_size=config.text_intermediate_size,
        )
    for index, refiner_block in enumerate(model.text_fusion.refiner_blocks):
        _text_block_targets(
            targets,
            prefix=f"text_fusion.refiner_blocks.{index}.",
            block=refiner_block,
            intermediate_size=config.text_intermediate_size,
        )
    for index, transformer_block in enumerate(model.transformer_blocks):
        _main_block_targets(
            targets,
            prefix=f"transformer_blocks.{index}.",
            block=transformer_block,
            intermediate_size=config.intermediate_size,
        )
    expected = expected_transformer_keys(config)
    if frozenset(targets) != expected:
        missing = sorted(expected - targets.keys())
        extra = sorted(targets.keys() - expected)
        raise RuntimeError(
            "Krea transformer target graph disagrees with the canonical weight mapping: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return targets


def load_krea2_transformer_weights(
    model: Krea2Transformer,
    *,
    expected_dtype: FloatDType | mx.Dtype | None = None,
    source_bindings: tuple[Krea2TensorBinding, ...],
    reporter: Reporter | None = None,
) -> Krea2TransformerLoadReceipt:
    """Bind every canonical transformer source tensor exactly once."""
    if (
        model.reference_denoise
        and expected_dtype is not None
        and resolve_float_dtype(expected_dtype, owner="Krea 2 transformer") == mx.float16
    ):
        raise ValueError(REFERENCE_FP16_ERROR)
    targets = transformer_weight_targets(model)
    mapping = {binding.target_key: binding for binding in source_bindings}
    if len(mapping) != len(source_bindings) or mapping.keys() != targets.keys():
        raise ValueError("Krea transformer source bindings do not match the target graph")
    physical_targets = {
        key: replace(target, shape=tuple(reversed(target.shape)))
        if mapping[key].transform == KREA2_TRANSFORMER_PRETRANSPOSE
        else target
        for key, target in targets.items()
    }
    bound, files = bind_source_weights(
        physical_targets,
        source_bindings,
        dtype=resolve_float_dtype(
            expected_dtype or FloatDType.BFLOAT16, owner="Krea 2 transformer"
        ),
        phase="bind transformer weights",
        reporter=reporter,
    )
    # Reference execution restores source orientation after layout preparation.
    if model.reference_denoise:
        for block in model.transformer_blocks:
            block.ff.down.weight = mx.contiguous(block.ff.down.weight.T)
    if bound != len(targets):
        raise RuntimeError(f"Krea transformer loader bound {bound} of {len(targets)} targets")
    target_payload: JsonObject = {
        key: list(target.shape) for key, target in sorted(targets.items())
    }
    return Krea2TransformerLoadReceipt(
        target_count=bound,
        target_digest=payload_digest(target_payload),
        source_files=tuple(path.name for path in files),
    )


__all__ = [
    "KREA2_TRANSFORMER_CONSUMED_TENSORS",
    "Krea2TransformerLoadReceipt",
    "load_krea2_transformer_weights",
    "transformer_weight_targets",
]
