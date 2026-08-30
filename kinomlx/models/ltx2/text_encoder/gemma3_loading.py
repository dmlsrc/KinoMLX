"""Complete consumed-target loading for the allocation-light Gemma 3 backbone."""

from __future__ import annotations

from pathlib import Path

from kinomlx.reporting import Reporter

from ._loading import WeightTarget, bind_weight_targets
from .gemma3 import Gemma3Model


def _weight_files(path: Path | str) -> tuple[Path, ...]:
    root = Path(path).expanduser()
    if root.is_file():
        if root.suffix != ".safetensors":
            raise ValueError(f"Gemma weight file must be safetensors: {root}")
        return (root,)
    if not root.is_dir():
        raise FileNotFoundError(f"Gemma model path does not exist: {root}")
    shards = tuple(sorted(root.glob("model-*.safetensors")))
    if shards:
        return shards
    single = root / "model.safetensors"
    if single.is_file():
        return (single,)
    raise FileNotFoundError(f"no Gemma safetensors found in {root}")


def _targets(model: Gemma3Model) -> dict[str, WeightTarget]:
    config = model.config
    targets: dict[str, WeightTarget] = {
        "embed_tokens.weight": WeightTarget(
            model.embed_tokens,
            "weight",
            (config.vocab_size, config.hidden_size),
        ),
        "norm.weight": WeightTarget(model.norm, "weight", (config.hidden_size,)),
    }
    query_dims = config.num_attention_heads * config.head_dim
    kv_dims = config.num_key_value_heads * config.head_dim
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}."
        targets.update(
            {
                f"{prefix}self_attn.q_proj.weight": WeightTarget(
                    layer.self_attn.q_proj,
                    "weight",
                    (query_dims, config.hidden_size),
                ),
                f"{prefix}self_attn.k_proj.weight": WeightTarget(
                    layer.self_attn.k_proj,
                    "weight",
                    (kv_dims, config.hidden_size),
                ),
                f"{prefix}self_attn.v_proj.weight": WeightTarget(
                    layer.self_attn.v_proj,
                    "weight",
                    (kv_dims, config.hidden_size),
                ),
                f"{prefix}self_attn.o_proj.weight": WeightTarget(
                    layer.self_attn.o_proj,
                    "weight",
                    (config.hidden_size, query_dims),
                ),
                f"{prefix}self_attn.q_norm.weight": WeightTarget(
                    layer.self_attn.q_norm,
                    "weight",
                    (config.head_dim,),
                ),
                f"{prefix}self_attn.k_norm.weight": WeightTarget(
                    layer.self_attn.k_norm,
                    "weight",
                    (config.head_dim,),
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
            }
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


def load_gemma3_weights(
    model: Gemma3Model,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Bind every Gemma 3 target once while accepting unrelated source baggage."""
    files = _weight_files(path)
    return bind_weight_targets(
        files,
        _targets(model),
        lambda logical_key: (f"model.{logical_key}",),
        component="LTX-2.3 Gemma 3",
        phase="load Gemma 3 weights",
        reporter=reporter,
    )


__all__ = ["load_gemma3_weights"]
