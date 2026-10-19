"""The closed string sets LTX-2 decides the membership of.

These live in one leaf module rather than beside the records that hold them,
because a vocabulary is needed by both the record that carries a value and the
rule that judges it, and those two import each other. ``types.py`` imports
``request_rules.py``, and ``settings.py`` imports ``settings_rules.py``, so a
vocabulary declared in either half cannot be reached from the other.

Each set is this project's decision, which is what earns it a name. A
generation is chosen from the checkpoint's own structure rather than read from
a field, the Gemma variant selects between two releases this project pairs with
LTX-2.3, the two dtype vocabularies both carry an ``auto`` that resolves here,
and the stage sampler's ``auto`` resolves against the checkpoint generation.
The diffusion video decoder's attention receipt names its backend and its
fallback reason, which the decoder records and its kernel route decides.
Argparse's ``nargs`` spellings and the safetensors header spellings are not
here, because their membership is decided elsewhere.
"""

from enum import StrEnum


class LTX2ModelGeneration(StrEnum):
    """The checkpoint generations this version implements."""

    LTX23 = "2.3"
    LTX25 = "2.5"


MODEL_GENERATION_CHOICES = tuple(item.value for item in LTX2ModelGeneration)


class GemmaVariant(StrEnum):
    """Which published Gemma release supplies the text encoder."""

    QAT = "qat"
    PLAIN = "plain"


GEMMA_VARIANT_CHOICES = tuple(item.value for item in GemmaVariant)


# ``auto`` decodes in FP16 with either video VAE, SDR or HDR, and decodes again
# in BF16 when the first decoded frames are not finite. Explicit values are
# controlled diagnostics and are also useful for decode-only restarts.
class LTX2VideoVAEDecodeDType(StrEnum):
    """The dtype the video VAE decodes at."""

    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


VAE_DECODE_DTYPE_CHOICES = tuple(item.value for item in LTX2VideoVAEDecodeDType)


# ``auto`` selects float16 only for a streamed float16 transformer, where every
# reload repeats the fusion; a resident transformer fuses once and keeps
# float32 even at float16.
class LTX2AdapterAccumulation(StrEnum):
    """The dtype adapter fusion accumulates in."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"


ADAPTER_ACCUMULATION_CHOICES = tuple(item.value for item in LTX2AdapterAccumulation)


# ``anchor`` replaces the canvas's first latent frame and holds it clean, so at
# full strength the frame-0 latent is the encoded input. ``guide`` appends the
# image as clean keyframe tokens and leaves the first latent frame denoised.
# Only the frame-0 condition reads this; every other index appends either way.
class LTX2FirstFrameMode(StrEnum):
    """How the frame-0 image condition is applied."""

    ANCHOR = "anchor"
    GUIDE = "guide"


FIRST_FRAME_MODE_CHOICES = tuple(item.value for item in LTX2FirstFrameMode)


# ``auto`` runs the sampler the checkpoint generation publishes for that stage:
# deterministic Euler for both LTX-2.3 stages, RF ancestral Euler for both
# LTX-2.5 stages. An explicit value replaces one stage's sampler on either
# generation, for experiments, without touching the sigma schedules.
class LTX2StageSampler(StrEnum):
    """Which sampler one distilled stage runs."""

    AUTO = "auto"
    DETERMINISTIC = "deterministic"
    ANCESTRAL = "ancestral"


STAGE_SAMPLER_CHOICES = tuple(item.value for item in LTX2StageSampler)


# The diffusion video decoder's attention receipt names how its neighborhood
# attention ran: the masked MLX path for every call, the FP16 kernel route for
# every call, or each for some.
class DiffusionAttentionBackend(StrEnum):
    """Which implementation computed a decode's neighborhood attention."""

    MASKED_SDPA = "mlx-sdpa-exact-neighborhood"
    FP16_KERNELS = "fp16-neighborhood-kernels"
    MIXED = "mixed"


class DiffusionKernelFallback(StrEnum):
    """Why a diffusion video decoder block kept the masked attention path."""

    DISABLED = "disabled"
    DTYPE = "dtype"
    BATCH = "batch"
    HEAD_DIM = "head-dim"
    WINDOW = "window"


__all__ = [
    "ADAPTER_ACCUMULATION_CHOICES",
    "FIRST_FRAME_MODE_CHOICES",
    "GEMMA_VARIANT_CHOICES",
    "MODEL_GENERATION_CHOICES",
    "STAGE_SAMPLER_CHOICES",
    "VAE_DECODE_DTYPE_CHOICES",
    "DiffusionAttentionBackend",
    "DiffusionKernelFallback",
    "GemmaVariant",
    "LTX2AdapterAccumulation",
    "LTX2FirstFrameMode",
    "LTX2ModelGeneration",
    "LTX2StageSampler",
    "LTX2VideoVAEDecodeDType",
]
