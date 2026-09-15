"""LTX checkpoint key conversion and family routing."""

from kinomlx.models.ltx2.weights.keys.audio_vae import (
    is_audio_vae_key,
    is_component_local_audio_vae_key,
    is_component_local_vocoder_key,
    is_vocoder_key,
)
from kinomlx.models.ltx2.weights.keys.text_encoder import is_connector_key
from kinomlx.models.ltx2.weights.keys.transformer import (
    convert_checkpoint_key,
    convert_pytorch_key_to_mlx,
    flatten_to_nested,
)
from kinomlx.models.ltx2.weights.keys.video_vae import (
    is_component_local_video_vae_key,
    is_video_vae_key,
    lookup_weight,
)

from ...transformer.graph import DIFFUSION_PREFIX


def weight_family_for_key(
    key: str,
    *,
    source_component: str | None = None,
) -> str | None:
    """Return the split auxiliary family for a stock LTX checkpoint key."""
    if is_video_vae_key(key):
        return "video_vae"
    if is_audio_vae_key(key):
        return "audio_vae"
    if is_vocoder_key(key):
        return "vocoder"
    if is_connector_key(key):
        return "connector"
    if source_component == "video_vae" and is_component_local_video_vae_key(key):
        return "video_vae"
    if source_component == "audio_vae_vocoder":
        if is_component_local_audio_vae_key(key):
            return "audio_vae"
        if is_component_local_vocoder_key(key):
            return "vocoder"
    return None


def canonical_weight_family_key(family: str, key: str) -> str:
    """Remove packaging-only wrappers from a prepared family weight key."""
    prefixes = {
        "video_vae": "vae.",
        "audio_vae": "audio_vae.",
        "vocoder": "vocoder.",
    }
    prefix = prefixes.get(family)
    return key.removeprefix(prefix) if prefix is not None else key


__all__ = [
    "DIFFUSION_PREFIX",
    "canonical_weight_family_key",
    "convert_checkpoint_key",
    "convert_pytorch_key_to_mlx",
    "flatten_to_nested",
    "lookup_weight",
    "weight_family_for_key",
]
