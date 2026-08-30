"""Text-projection and AV connector checkpoint family routing."""

CONNECTOR_PREFIXES = (
    "text_embedding_projection.",
    "model.diffusion_model.video_embeddings_connector.",
    "model.diffusion_model.audio_embeddings_connector.",
    "model.diffusion_model.embeddings_connector.",
)


def is_connector_key(key: str) -> bool:
    """Return whether ``key`` belongs in the connector family cache."""
    return key.startswith(CONNECTOR_PREFIXES)


__all__ = ["CONNECTOR_PREFIXES", "is_connector_key"]
