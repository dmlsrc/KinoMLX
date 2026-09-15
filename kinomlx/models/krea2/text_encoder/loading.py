"""Exact 386-tensor source binding for Krea's truncated Qwen text graph."""

from dataclasses import dataclass

from kinomlx._json import JsonObject
from kinomlx.io.identity import payload_digest
from kinomlx.modeling.loading import WeightTarget
from kinomlx.reporting import Reporter

from ..metadata import Krea2TextConfig
from ..resource_types import Krea2TensorBinding
from ..source_loading import bind_source_weights
from .model import Qwen3VLTextModel

KREA2_QWEN_CONSUMED_TENSORS = 386


@dataclass(frozen=True)
class Qwen3VLLoadReceipt:
    """Stable identity of the exact Qwen target graph bound for one station."""

    target_count: int
    target_digest: str
    source_files: tuple[str, ...]


def _target_shapes(
    config: Krea2TextConfig,
    selected_layers: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    if not selected_layers:
        raise ValueError("Krea requires at least one selected Qwen hidden state")
    if tuple(sorted(set(selected_layers))) != selected_layers:
        raise ValueError("Krea selected Qwen layers must be sorted and unique")
    if selected_layers[0] <= 0 or selected_layers[-1] > config.num_hidden_layers:
        raise ValueError("Krea selected Qwen layer is outside the source graph")

    query_dims = config.num_attention_heads * config.head_dim
    kv_dims = config.num_key_value_heads * config.head_dim
    targets: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (config.vocab_size, config.hidden_size),
    }
    for index in range(selected_layers[-1]):
        prefix = f"layers.{index}."
        targets.update(
            {
                f"{prefix}self_attn.q_proj.weight": (query_dims, config.hidden_size),
                f"{prefix}self_attn.k_proj.weight": (kv_dims, config.hidden_size),
                f"{prefix}self_attn.v_proj.weight": (kv_dims, config.hidden_size),
                f"{prefix}self_attn.o_proj.weight": (config.hidden_size, query_dims),
                f"{prefix}self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
                f"{prefix}input_layernorm.weight": (config.hidden_size,),
                f"{prefix}post_attention_layernorm.weight": (config.hidden_size,),
            }
        )
    return targets


def qwen3_vl_target_identity(
    config: Krea2TextConfig,
    selected_layers: tuple[int, ...],
) -> tuple[int, str]:
    """Return the exact allocation-free target-graph identity."""
    shapes = _target_shapes(config, selected_layers)
    payload: JsonObject = {key: list(shape) for key, shape in sorted(shapes.items())}
    return len(shapes), payload_digest(payload)


def _targets(model: Qwen3VLTextModel) -> dict[str, WeightTarget]:
    config = model.config
    shapes = _target_shapes(config, model.selected_layers)
    targets: dict[str, WeightTarget] = {
        "embed_tokens.weight": WeightTarget(
            model.embed_tokens,
            "weight",
            shapes["embed_tokens.weight"],
        )
    }
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}."
        targets.update(
            {
                f"{prefix}self_attn.q_proj.weight": WeightTarget(
                    layer.self_attn.q_proj,
                    "weight",
                    shapes[f"{prefix}self_attn.q_proj.weight"],
                ),
                f"{prefix}self_attn.k_proj.weight": WeightTarget(
                    layer.self_attn.k_proj,
                    "weight",
                    shapes[f"{prefix}self_attn.k_proj.weight"],
                ),
                f"{prefix}self_attn.v_proj.weight": WeightTarget(
                    layer.self_attn.v_proj,
                    "weight",
                    shapes[f"{prefix}self_attn.v_proj.weight"],
                ),
                f"{prefix}self_attn.o_proj.weight": WeightTarget(
                    layer.self_attn.o_proj,
                    "weight",
                    shapes[f"{prefix}self_attn.o_proj.weight"],
                ),
                f"{prefix}self_attn.q_norm.weight": WeightTarget(
                    layer.self_attn.q_norm,
                    "weight",
                    shapes[f"{prefix}self_attn.q_norm.weight"],
                ),
                f"{prefix}self_attn.k_norm.weight": WeightTarget(
                    layer.self_attn.k_norm,
                    "weight",
                    shapes[f"{prefix}self_attn.k_norm.weight"],
                ),
                f"{prefix}mlp.gate_proj.weight": WeightTarget(
                    layer.mlp.gate_proj,
                    "weight",
                    shapes[f"{prefix}mlp.gate_proj.weight"],
                ),
                f"{prefix}mlp.up_proj.weight": WeightTarget(
                    layer.mlp.up_proj,
                    "weight",
                    shapes[f"{prefix}mlp.up_proj.weight"],
                ),
                f"{prefix}mlp.down_proj.weight": WeightTarget(
                    layer.mlp.down_proj,
                    "weight",
                    shapes[f"{prefix}mlp.down_proj.weight"],
                ),
                f"{prefix}input_layernorm.weight": WeightTarget(
                    layer.input_layernorm,
                    "weight",
                    shapes[f"{prefix}input_layernorm.weight"],
                ),
                f"{prefix}post_attention_layernorm.weight": WeightTarget(
                    layer.post_attention_layernorm,
                    "weight",
                    shapes[f"{prefix}post_attention_layernorm.weight"],
                ),
            }
        )
    return targets


def load_qwen3_vl_text_weights(
    model: Qwen3VLTextModel,
    *,
    source_bindings: tuple[Krea2TensorBinding, ...],
    reporter: Reporter | None = None,
) -> Qwen3VLLoadReceipt:
    """Bind every consumed source tensor exactly once and ignore no target."""
    targets = _targets(model)
    target_count, target_digest = qwen3_vl_target_identity(
        model.config,
        model.selected_layers,
    )
    if len(targets) != target_count or target_count != KREA2_QWEN_CONSUMED_TENSORS:
        raise ValueError(
            f"Krea Qwen graph has {len(targets)} targets; expected {KREA2_QWEN_CONSUMED_TENSORS}"
        )
    # Text weights retain their source dtype.
    bound, files = bind_source_weights(
        targets,
        source_bindings,
        dtype=None,
        phase="bind Qwen text weights",
        reporter=reporter,
    )
    if bound != len(targets):
        raise RuntimeError(f"Krea Qwen loader bound {bound} of {len(targets)} targets")
    return Qwen3VLLoadReceipt(
        target_count=bound,
        target_digest=target_digest,
        source_files=tuple(path.name for path in files),
    )


__all__ = [
    "KREA2_QWEN_CONSUMED_TENSORS",
    "Qwen3VLLoadReceipt",
    "load_qwen3_vl_text_weights",
    "qwen3_vl_target_identity",
]
