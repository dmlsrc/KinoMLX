"""Complete projection and connector loading for both LTX generations."""

from __future__ import annotations

from pathlib import Path

from kinomlx.reporting import Reporter

from ._loading import (
    WeightTarget,
    bind_resolved_weights,
    bind_weight_targets,
    resolve_weight_bindings,
)
from .connector import Embeddings1DConnector
from .encoder import AudioVideoGemmaTextEncoderModel


def _linear_targets(
    prefix: str,
    layer: object,
    input_dims: int,
    output_dims: int,
    *,
    bias: bool,
) -> dict[str, WeightTarget]:
    targets = {
        f"{prefix}.weight": WeightTarget(layer, "weight", (output_dims, input_dims)),
    }
    if bias:
        targets[f"{prefix}.bias"] = WeightTarget(layer, "bias", (output_dims,))
    return targets


def _connector_targets(
    connector: Embeddings1DConnector,
    prefix: str,
    *,
    ff_bias: bool,
) -> dict[str, WeightTarget]:
    dims = connector.inner_dim
    heads = connector.num_attention_heads
    register_count = connector.num_learnable_registers
    targets: dict[str, WeightTarget] = {}
    if register_count is not None:
        targets[f"{prefix}.learnable_registers"] = WeightTarget(
            connector,
            "learnable_registers",
            (register_count, dims),
        )
    for index, block in enumerate(connector.transformer_1d_blocks):
        block_prefix = f"{prefix}.transformer_1d_blocks.{index}"
        attention_prefix = f"{block_prefix}.attn1"
        for name in ("to_q", "to_k", "to_v"):
            targets.update(
                _linear_targets(
                    f"{attention_prefix}.{name}",
                    getattr(block.attn1, name),
                    dims,
                    dims,
                    bias=True,
                )
            )
        targets.update(
            _linear_targets(
                f"{attention_prefix}.to_out.0",
                block.attn1.to_out,
                dims,
                dims,
                bias=True,
            )
        )
        targets[f"{attention_prefix}.q_norm.weight"] = WeightTarget(
            block.attn1.q_norm,
            "weight",
            (dims,),
        )
        targets[f"{attention_prefix}.k_norm.weight"] = WeightTarget(
            block.attn1.k_norm,
            "weight",
            (dims,),
        )
        if block.attn1.to_gate_logits is not None:
            targets.update(
                _linear_targets(
                    f"{attention_prefix}.to_gate_logits",
                    block.attn1.to_gate_logits,
                    dims,
                    heads,
                    bias=True,
                )
            )
        targets.update(
            _linear_targets(
                f"{block_prefix}.ff.net.0.proj",
                block.ff.project_in.proj,
                dims,
                dims * 4,
                bias=ff_bias,
            )
        )
        targets.update(
            _linear_targets(
                f"{block_prefix}.ff.net.2",
                block.ff.project_out,
                dims * 4,
                dims,
                bias=ff_bias,
            )
        )
    return targets


def _targets(
    model: AudioVideoGemmaTextEncoderModel,
) -> dict[str, WeightTarget]:
    config = model.config
    flat_dims = config.hidden_dim * config.num_gemma_states
    targets: dict[str, WeightTarget] = {}
    targets.update(
        _linear_targets(
            "text_embedding_projection.video_aggregate_embed",
            model.feature_extractor.video_aggregate_embed,
            flat_dims,
            config.video_inner_dim,
            bias=True,
        )
    )
    targets.update(
        _linear_targets(
            "text_embedding_projection.audio_aggregate_embed",
            model.feature_extractor.audio_aggregate_embed,
            flat_dims,
            config.audio_inner_dim,
            bias=True,
        )
    )
    targets.update(
        _connector_targets(
            model.embeddings_connector,
            "model.diffusion_model.video_embeddings_connector",
            ff_bias=config.ff_bias,
        )
    )
    targets.update(
        _connector_targets(
            model.audio_embeddings_connector,
            "model.diffusion_model.audio_embeddings_connector",
            ff_bias=config.ff_bias,
        )
    )
    return targets


def _projection_targets(model: AudioVideoGemmaTextEncoderModel) -> dict[str, WeightTarget]:
    return {
        key: target
        for key, target in _targets(model).items()
        if key.startswith("text_embedding_projection.")
    }


def _connector_targets_only(model: AudioVideoGemmaTextEncoderModel) -> dict[str, WeightTarget]:
    return {
        key: target
        for key, target in _targets(model).items()
        if key.startswith("model.diffusion_model.")
    }


def load_av_text_encoder_v2_weights(
    model: AudioVideoGemmaTextEncoderModel,
    path: Path | str,
    *,
    projection_path: Path | str | None = None,
    reporter: Reporter | None = None,
) -> int:
    """Load all consumed targets from a monolith/cache or split packaged sources."""
    connector_source = Path(path).expanduser().absolute()
    projection_source = (
        connector_source
        if projection_path is None
        else Path(projection_path).expanduser().absolute()
    )
    for source in {projection_source, connector_source}:
        if not source.is_file() or source.suffix != ".safetensors":
            raise FileNotFoundError(f"LTX text-conditioning safetensors not found: {source}")
    if projection_source == connector_source:
        return bind_weight_targets(
            (connector_source,),
            _targets(model),
            lambda logical_key: (logical_key,),
            component="LTX text projection/connectors",
            phase="load AV text connectors",
            reporter=reporter,
        )

    projection_bindings = resolve_weight_bindings(
        (projection_source,),
        _projection_targets(model),
        lambda logical_key: (logical_key,),
        component="LTX text projection",
    )
    connector_bindings = resolve_weight_bindings(
        (connector_source,),
        _connector_targets_only(model),
        lambda logical_key: (logical_key,),
        component="LTX AV text connectors",
    )
    projection_count = bind_resolved_weights(
        (projection_source,),
        projection_bindings,
        phase="load LTX text projection",
        reporter=reporter,
    )
    connector_count = bind_resolved_weights(
        (connector_source,),
        connector_bindings,
        phase="load AV text connectors",
        reporter=reporter,
    )
    return projection_count + connector_count


__all__ = ["load_av_text_encoder_v2_weights"]
