"""Complete consumed-target loading for the LTX-tuned Gemma 4 backbone."""

from __future__ import annotations

from pathlib import Path

from kinomlx.reporting import Reporter

from ._loading import WeightTarget, bind_weight_targets
from .gemma4 import Gemma4Model


def _targets(model: Gemma4Model) -> dict[str, WeightTarget]:
    config = model.config
    targets = {
        "embed_tokens.weight": WeightTarget(
            model.embed_tokens,
            "weight",
            (config.vocab_size, config.hidden_size),
        ),
        "norm.weight": WeightTarget(model.norm, "weight", (config.hidden_size,)),
    }
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}."
        attention = layer.self_attn
        query_dims = attention.num_heads * attention.head_dim
        kv_dims = attention.num_kv_heads * attention.head_dim
        targets.update(
            {
                f"{prefix}self_attn.q_proj.weight": WeightTarget(
                    attention.q_proj,
                    "weight",
                    (query_dims, config.hidden_size),
                ),
                f"{prefix}self_attn.k_proj.weight": WeightTarget(
                    attention.k_proj,
                    "weight",
                    (kv_dims, config.hidden_size),
                ),
                f"{prefix}self_attn.o_proj.weight": WeightTarget(
                    attention.o_proj,
                    "weight",
                    (config.hidden_size, query_dims),
                ),
                f"{prefix}self_attn.q_norm.weight": WeightTarget(
                    attention.q_norm,
                    "weight",
                    (attention.head_dim,),
                ),
                f"{prefix}self_attn.k_norm.weight": WeightTarget(
                    attention.k_norm,
                    "weight",
                    (attention.head_dim,),
                ),
                f"{prefix}mlp.gate_proj.weight": WeightTarget(
                    layer.mlp.gate_proj,
                    "weight",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}mlp.up_proj.weight": WeightTarget(
                    layer.mlp.up_proj,
                    "weight",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}mlp.down_proj.weight": WeightTarget(
                    layer.mlp.down_proj,
                    "weight",
                    (config.hidden_size, config.intermediate_size),
                ),
                f"{prefix}layer_scalar": WeightTarget(layer, "layer_scalar", (1,)),
            }
        )
        if attention.v_proj is not None:
            targets[f"{prefix}self_attn.v_proj.weight"] = WeightTarget(
                attention.v_proj,
                "weight",
                (kv_dims, config.hidden_size),
            )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            targets[f"{prefix}{name}.weight"] = WeightTarget(
                getattr(layer, name),
                "weight",
                (config.hidden_size,),
            )
    return targets


def load_gemma4_weights(
    model: Gemma4Model,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Bind every Gemma 4 target once while accepting unrelated source baggage."""
    source = Path(path).expanduser().absolute()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX-2.5 Gemma 4 safetensors not found: {source}")
    return bind_weight_targets(
        (source,),
        _targets(model),
        lambda logical_key: (f"model.{logical_key}",),
        component="LTX-2.5 Gemma 4",
        phase="load Gemma 4 weights",
        reporter=reporter,
    )


__all__ = ["load_gemma4_weights"]
