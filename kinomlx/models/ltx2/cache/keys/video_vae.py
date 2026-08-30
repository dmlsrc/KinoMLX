"""Video VAE family routing and structure-driven lookup helpers."""

from collections.abc import Mapping

VIDEO_VAE_PREFIX = "vae."
VIDEO_VAE_COMPONENT_PREFIXES = (
    "encoder.",
    "decoder.",
    "per_channel_statistics.",
    "vae_encoder.",
    "vae_decoder.",
)


def is_video_vae_key(key: str) -> bool:
    return key.startswith(VIDEO_VAE_PREFIX)


def is_component_local_video_vae_key(key: str) -> bool:
    """Identify video-VAE namespaces only after the source is typed as one."""
    prefixes = (VIDEO_VAE_PREFIX, *VIDEO_VAE_COMPONENT_PREFIXES)
    return key.startswith(prefixes) or any(f".{prefix}" in key for prefix in prefixes)


def lookup_weight[T](
    weights: Mapping[str, T],
    primary_key: str,
    fallback_key: str | None = None,
) -> T:
    """Look up a structure-derived weight with one optional legacy fallback."""
    if primary_key in weights:
        return weights[primary_key]
    if fallback_key is not None and fallback_key in weights:
        return weights[fallback_key]
    suffix = f" or {fallback_key!r}" if fallback_key is not None else ""
    raise KeyError(f"missing video VAE weight {primary_key!r}{suffix}")


__all__ = [
    "VIDEO_VAE_COMPONENT_PREFIXES",
    "VIDEO_VAE_PREFIX",
    "is_component_local_video_vae_key",
    "is_video_vae_key",
    "lookup_weight",
]
