"""Test-owned synthetic parameter materialization for small text graphs."""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.models.ltx2.text_encoder._loading import WeightTarget
from kinomlx.models.ltx2.text_encoder.encoder import AudioVideoGemmaTextEncoderModel
from kinomlx.models.ltx2.text_encoder.gemma3 import Gemma3Model
from kinomlx.models.ltx2.text_encoder.gemma3_loading import _targets as _gemma3_targets
from kinomlx.models.ltx2.text_encoder.gemma4 import Gemma4Model
from kinomlx.models.ltx2.text_encoder.gemma4_loading import _targets as _gemma4_targets
from kinomlx.models.ltx2.text_encoder.loading import _targets as _connector_targets


def _materialize_targets(
    targets: Mapping[str, WeightTarget],
    *,
    seed: int,
) -> None:
    key = mx.random.key(seed)
    for target in targets.values():
        key, subkey = mx.random.split(key)
        value = mx.random.uniform(
            low=-0.05,
            high=0.05,
            shape=target.shape,
            key=subkey,
        )
        setattr(target.owner, target.attribute, value)


def initialize_test_parameters(model: nn.Module, *, seed: int = 0) -> None:
    """Materialize one supported shell from its canonical checkpoint targets."""
    if isinstance(model, Gemma3Model):
        targets = _gemma3_targets(model)
    elif isinstance(model, Gemma4Model):
        targets = _gemma4_targets(model)
    elif isinstance(model, AudioVideoGemmaTextEncoderModel):
        targets = _connector_targets(model)
    else:
        raise TypeError(f"unsupported synthetic text model: {type(model).__name__}")
    _materialize_targets(targets, seed=seed)


__all__ = ["initialize_test_parameters"]
