"""LTX checkpoint key conversion and family routing."""

from .audio_vae import (
    is_audio_vae_key,
    is_component_local_audio_vae_key,
    is_component_local_vocoder_key,
    is_vocoder_key,
)
from .text_encoder import is_connector_key
from .transformer import (
    DIFFUSION_PREFIX,
    convert_checkpoint_key,
    convert_pytorch_key_to_mlx,
    flatten_to_nested,
)
from .video_vae import (
    is_component_local_video_vae_key,
    is_video_vae_key,
    lookup_weight,
)


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


__all__ = [
    "DIFFUSION_PREFIX",
    "convert_checkpoint_key",
    "convert_pytorch_key_to_mlx",
    "flatten_to_nested",
    "lookup_weight",
    "weight_family_for_key",
]
