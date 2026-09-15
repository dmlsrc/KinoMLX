"""Semantic configuration groups consumed only by LTX-2 assembly."""

from enum import StrEnum


class LTX2ConfigGroup(StrEnum):
    """Cross-field roles used by LTX-2 precedence and restart policy."""

    MODEL_SOURCE = "model_source"
    MODEL_MONOLITHIC_SOURCE = "model_monolithic_source"
    MODEL_SPLIT_SOURCE = "model_split_source"
    MODEL_GEMMA_SOURCE = "model_gemma_source"
    MODEL_TEXT_ENCODER_SOURCE = "model_text_encoder_source"
    MODEL_GENERATION_SELECTOR = "model_generation_selector"
    MODEL_VIDEO_VAE_SELECTOR = "model_video_vae_selector"
    MODEL_VIDEO_VAE_SOURCE = "model_video_vae_source"


__all__ = ["LTX2ConfigGroup"]
