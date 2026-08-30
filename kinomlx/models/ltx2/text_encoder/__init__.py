"""Shared Gemma 3/4 prompt encoding and LTX-2.x audio/video connectors."""

from .connector import BasicTransformerBlock1D, Embeddings1DConnector, RopeType
from .encoder import (
    AudioVideoGemmaEncoderOutput,
    AudioVideoGemmaTextEncoderModel,
    AVTextEncoderConfig,
    create_av_text_encoder_v2,
    create_av_text_encoder_v2_from_checkpoint,
    encode_prompt,
)
from .features import GemmaFeaturesExtractorV2, norm_and_concat_per_token_rms
from .gemma3 import GEMMA3_LAYER_TYPES, Gemma3Config, Gemma3Model
from .gemma3_loading import load_gemma3_weights
from .gemma4 import GEMMA4_LAYER_TYPES, Gemma4Config, Gemma4Model, Gemma4RMSNorm
from .gemma4_loading import load_gemma4_weights
from .loading import load_av_text_encoder_v2_weights
from .tokenizer import GemmaTokenizer
from .tokenizer_cache import (
    TokenizerCache,
    TokenizerSource,
    derive_tokenizer_model,
    ensure_tokenizer_cache,
    load_tokenizer_derivation,
    resolve_tokenizer_source,
)

__all__ = [
    "AVTextEncoderConfig",
    "GEMMA3_LAYER_TYPES",
    "GEMMA4_LAYER_TYPES",
    "AudioVideoGemmaEncoderOutput",
    "AudioVideoGemmaTextEncoderModel",
    "BasicTransformerBlock1D",
    "Embeddings1DConnector",
    "Gemma3Config",
    "Gemma3Model",
    "Gemma4Config",
    "Gemma4Model",
    "Gemma4RMSNorm",
    "GemmaFeaturesExtractorV2",
    "GemmaTokenizer",
    "RopeType",
    "TokenizerCache",
    "TokenizerSource",
    "create_av_text_encoder_v2",
    "create_av_text_encoder_v2_from_checkpoint",
    "encode_prompt",
    "ensure_tokenizer_cache",
    "load_tokenizer_derivation",
    "load_av_text_encoder_v2_weights",
    "load_gemma3_weights",
    "load_gemma4_weights",
    "norm_and_concat_per_token_rms",
    "derive_tokenizer_model",
    "resolve_tokenizer_source",
]
