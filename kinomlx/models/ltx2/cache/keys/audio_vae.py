"""Audio VAE and vocoder family routing."""

AUDIO_VAE_PREFIX = "audio_vae."
VOCODER_PREFIX = "vocoder."


def is_audio_vae_key(key: str) -> bool:
    return key.startswith(AUDIO_VAE_PREFIX)


def is_vocoder_key(key: str) -> bool:
    return key.startswith(VOCODER_PREFIX)


def _has_namespace(key: str, prefixes: str | tuple[str, ...]) -> bool:
    values = (prefixes,) if isinstance(prefixes, str) else prefixes
    return key.startswith(values) or any(f".{prefix}" in key for prefix in values)


def is_component_local_audio_vae_key(key: str) -> bool:
    """Identify audio-VAE keys after the physical source is typed as audio."""
    return _has_namespace(key, AUDIO_VAE_PREFIX) or _has_namespace(
        key,
        ("encoder.", "decoder.", "per_channel_statistics."),
    )


def is_component_local_vocoder_key(key: str) -> bool:
    """Identify a vocoder namespace inside a typed audio component source."""
    return _has_namespace(key, VOCODER_PREFIX) or _has_namespace(
        key,
        ("bwe_generator.", "mel_stft."),
    )


__all__ = [
    "AUDIO_VAE_PREFIX",
    "VOCODER_PREFIX",
    "is_component_local_audio_vae_key",
    "is_component_local_vocoder_key",
    "is_audio_vae_key",
    "is_vocoder_key",
]
