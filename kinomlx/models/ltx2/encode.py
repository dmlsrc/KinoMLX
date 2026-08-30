"""Pure prompt and conditioning-image encoding functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from kinomlx.io.image import load_image, load_raw_exr
from kinomlx.io.safetensors import load_weights_with_metadata
from kinomlx.reporting import NullReporter, Reporter

from .components import _VideoEncoderCallablePort
from .text_encoder.encoder import AudioVideoGemmaEncoderOutput, encode_prompt
from .text_encoder.tokenizer_cache import TokenizerCache
from .types import HDRAuthoring

_CONTEXT_DTYPES = frozenset({mx.bfloat16, mx.float16, mx.float32})
_LOG = logging.getLogger(__name__)

TextConditioningMetadataPolicy = Literal["require", "observe"]


def _require_finite(name: str, value: mx.array, source: Path) -> None:
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError(f"{source}: {name} must contain only finite values")


def encode_text(
    prompt: str,
    *,
    gemma_path: Path | str,
    connector_path: Path | str,
    config_path: Path | str,
    model_generation: str = "2.3",
    projection_path: Path | str | None = None,
    tokenizer_cache: TokenizerCache | None = None,
    pad_prompt_to_max: bool = True,
    reporter: Reporter | None = None,
) -> AudioVideoGemmaEncoderOutput:
    """Encode one prompt through Gemma and the LTX A/V connectors."""
    return encode_prompt(
        prompt,
        gemma_path=gemma_path,
        connector_path=connector_path,
        config_path=config_path,
        model_generation=model_generation,
        projection_path=projection_path,
        tokenizer_cache=tokenizer_cache,
        max_length=1024,
        pad_prompt_to_max=pad_prompt_to_max,
        reporter=reporter,
    )


def load_text_conditioning(
    path: Path | str,
    *,
    reporter: Reporter | None = None,
    metadata_policy: TextConditioningMetadataPolicy = "require",
) -> tuple[AudioVideoGemmaEncoderOutput, dict[str, str]]:
    """Load and validate a KinoMLX text-conditioning sidecar.

    ``metadata_policy="require"`` retains the ordinary saved-conditioning
    replay contract. Restart recipes use ``"observe"``: descriptive metadata
    and unrelated tensor baggage are reported but never authorize loading,
    while every consumed tensor still has to fit exactly.
    """
    if metadata_policy not in {"require", "observe"}:
        raise ValueError("metadata_policy must be require or observe")
    source = Path(path)
    sink = reporter if reporter is not None else NullReporter()
    sink.phase_start("load text conditioning")
    try:
        arrays, metadata = load_weights_with_metadata(source)
        required = {"video_encoding", "audio_encoding", "attention_mask"}
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{source}: missing text-conditioning tensors {missing}")
        extra = sorted(set(arrays) - required)
        if extra:
            if metadata_policy == "require":
                raise ValueError(f"{source}: unexpected text-conditioning tensors {extra}")
            _LOG.warning(
                "%s: ignoring %d unconsumed text-conditioning tensors",
                source,
                len(extra),
            )

        schema_version = metadata.get("schema_version")
        artifact_name = metadata.get("artifact")
        schema_differs = schema_version not in {"2", "3"}
        artifact_differs = artifact_name != "ltx2_text_conditioning"
        if metadata_policy == "require" and schema_differs:
            raise ValueError(f"{source}: unsupported text-conditioning schema {schema_version!r}")
        if metadata_policy == "require" and artifact_differs:
            raise ValueError(f"{source}: expected an LTX-2 text-conditioning artifact")

        metadata_differences = []
        if schema_differs:
            metadata_differences.append(f"schema_version={schema_version!r}")
        if artifact_differs:
            metadata_differences.append(f"artifact={artifact_name!r}")
        if metadata_differences:
            details = ", ".join(metadata_differences)
            _LOG.warning(
                "%s: advisory text-conditioning metadata differs (%s); "
                "continuing to consumed-tensor validation",
                source,
                details,
            )

        video = arrays["video_encoding"]
        audio = arrays["audio_encoding"]
        mask = arrays["attention_mask"]
        if video.ndim != 3 or video.shape[0] != 1 or video.shape[2] != 4096:
            raise ValueError(
                f"{source}: video_encoding must be (1, tokens, 4096), got {tuple(video.shape)}"
            )
        if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[2] != 2048:
            raise ValueError(
                f"{source}: audio_encoding must be (1, tokens, 2048), got {tuple(audio.shape)}"
            )
        if mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError(
                f"{source}: attention_mask must be (1, tokens), got {tuple(mask.shape)}"
            )
        if video.shape[1] != audio.shape[1] or video.shape[1] != mask.shape[1]:
            raise ValueError(f"{source}: text-conditioning token counts do not match")
        if video.shape[1] <= 0:
            raise ValueError(f"{source}: text conditioning must have a positive token count")
        for name, value in (("video_encoding", video), ("audio_encoding", audio)):
            if value.dtype not in _CONTEXT_DTYPES:
                raise ValueError(
                    f"{source}: {name} must use a supported floating dtype, got {value.dtype}"
                )
            _require_finite(name, value, source)
        _require_finite("attention_mask", mask, source)
        if not bool(mx.all(mx.logical_or(mx.equal(mask, 0), mx.equal(mask, 1))).item()):
            raise ValueError(f"{source}: attention_mask must contain only binary values")
        mx.eval(video, audio, mask)
        return AudioVideoGemmaEncoderOutput(video, audio, mask), metadata
    finally:
        sink.phase_end("load text conditioning")


def encode_image(
    path: Path | str,
    video_encoder: _VideoEncoderCallablePort,
    *,
    width: int,
    height: int,
    compute_dtype: mx.Dtype,
    hdr_authoring: HDRAuthoring | None = None,
    reporter: Reporter | None = None,
) -> mx.array:
    """Cover-crop one typed image and encode it as a one-frame video latent."""
    source = Path(path)
    if source.suffix.lower() == ".exr":
        if hdr_authoring is None:
            raise ValueError("EXR conditioning requires an explicit HDR signal interpretation")
        from kinomlx.media.hdr import (
            acescct_to_scene_linear,
            convert_scene_linear_primaries,
            scene_linear_to_acescct,
        )
        from kinomlx.media.signals import ColorPrimaries

        image = load_raw_exr(source, size=(width, height))
        if not bool(mx.all(mx.isfinite(image)).item()):
            raise ValueError(f"EXR conditioning must contain only finite values: {source}")
        if hdr_authoring == "ACESCCT":
            if bool(mx.any((image < 0.0) | (image > 1.0)).item()):
                raise ValueError(f"ACEScct EXR codes must be in [0, 1]: {source}")
            source_linear = acescct_to_scene_linear(image)
        else:
            source_linear = image
            linear = source_linear
            if hdr_authoring == "SRGB_LINEAR":
                linear = convert_scene_linear_primaries(
                    linear,
                    source=ColorPrimaries.REC709,
                    target=ColorPrimaries.ACESCG,
                )
            elif hdr_authoring != "ACESCG":
                raise ValueError(f"unsupported EXR signal interpretation {hdr_authoring!r}")
            image = scene_linear_to_acescct(linear)
        # Native HDR preserves the condition's exposure distribution, so a
        # plate with no above-reference-white content yields an effectively
        # SDR result inside an HDR container. Measure in the file's own
        # linear domain, before any primaries conversion.
        peak_linear = float(cast(int | float, mx.max(source_linear).item()))
        if peak_linear <= 1.001:
            _LOG.warning(
                "EXR condition %s peaks at %.3f in scene-linear terms (nothing above "
                "SDR reference white); the generation preserves the condition's "
                "exposure distribution, so this HDR run will carry effectively SDR "
                "content - expand the still to a genuine HDR plate first",
                source,
                peak_linear,
            )
    else:
        if hdr_authoring is not None:
            raise ValueError("HDR signal interpretation applies only to EXR conditioning")
        image = load_image(source, size=(width, height))
    video = (image * 2.0 - 1.0).transpose(2, 0, 1)[None, :, None, :, :]
    latent = video_encoder(
        video.astype(compute_dtype),
        reporter=reporter,
    )
    mx.eval(latent)
    return latent


__all__ = [
    "TextConditioningMetadataPolicy",
    "encode_image",
    "encode_text",
    "load_text_conditioning",
]
