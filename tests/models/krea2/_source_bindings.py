"""Canonical source bindings for allocation-light model fixtures."""

from kinomlx.models.krea2.resource_types import Krea2TensorBinding
from kinomlx.models.krea2.text_encoder.loading import _targets as text_targets
from kinomlx.models.krea2.transformer.loading import transformer_weight_targets
from kinomlx.models.krea2.vae.loading import _targets as vae_targets
from kinomlx.models.krea2.weight_keys import transformer_weight_transform


def fixture_bindings(model, path, kind):
    targets = {"transformer": transformer_weight_targets, "text": text_targets, "vae": vae_targets}[
        kind
    ](model)
    return tuple(
        Krea2TensorBinding(
            path,
            key,
            key,
            transform=transformer_weight_transform(key) if kind == "transformer" else None,
        )
        for key in targets
    )
