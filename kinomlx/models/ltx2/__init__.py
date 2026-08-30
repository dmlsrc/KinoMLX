"""LTX-2 public API with lazy imports for lightweight CLI discovery."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

_EXPORT_MODULES = {
    "DistilledRequest": "kinomlx.models.ltx2.types",
    "DistilledRestart": "kinomlx.models.ltx2.pipelines.restart",
    "EncodedTextConditioning": "kinomlx.models.ltx2.text_conditioning",
    "GenerationOutput": "kinomlx.models.ltx2.runner",
    "HDRAuthoring": "kinomlx.models.ltx2.types",
    "HDRReferenceConditioningConfig": "kinomlx.models.ltx2.types",
    "ImageConditioningConfig": "kinomlx.models.ltx2.types",
    "LTX2ArtifactConfig": "kinomlx.models.ltx2.artifacts",
    "LTX2Error": "kinomlx.models.ltx2.runner",
    "LTX2Resources": "kinomlx.models.ltx2.resources",
    "LTX2Runner": "kinomlx.models.ltx2.runner",
    "LTX2Settings": "kinomlx.models.ltx2.settings",
    "NativeTextConditioner": "kinomlx.models.ltx2.text_conditioning",
    "Recipe": "kinomlx.models.ltx2.runner",
    "TextConditioner": "kinomlx.models.ltx2.text_conditioning",
    "TextConditioningProvenance": "kinomlx.models.ltx2.text_conditioning",
    "VideoVAEDecodeDType": "kinomlx.models.ltx2.types",
    "VideoVAETilingConfig": "kinomlx.models.ltx2.types",
    "generate_distilled": "kinomlx.models.ltx2.pipelines.distilled",
    "prepare_resources": "kinomlx.models.ltx2.resources",
    "restart_distilled": "kinomlx.models.ltx2.pipelines.restart",
}

__all__ = sorted(_EXPORT_MODULES)

if TYPE_CHECKING:
    from . import artifacts as _artifacts
    from . import resources as _resources
    from . import runner as _runner
    from . import settings as _settings
    from . import text_conditioning as _text_conditioning
    from . import types as _types
    from .pipelines import distilled as _distilled
    from .pipelines import restart as _restart

    # Static mirrors of the lazy map preserve exact public types for mypy.
    DistilledRequest = _types.DistilledRequest
    DistilledRestart = _restart.DistilledRestart
    EncodedTextConditioning = _text_conditioning.EncodedTextConditioning
    GenerationOutput = _runner.GenerationOutput
    HDRAuthoring = _types.HDRAuthoring
    HDRReferenceConditioningConfig = _types.HDRReferenceConditioningConfig
    ImageConditioningConfig = _types.ImageConditioningConfig
    LTX2ArtifactConfig = _artifacts.LTX2ArtifactConfig
    LTX2Error = _runner.LTX2Error
    LTX2Resources = _resources.LTX2Resources
    LTX2Runner = _runner.LTX2Runner
    LTX2Settings = _settings.LTX2Settings
    NativeTextConditioner = _text_conditioning.NativeTextConditioner
    Recipe = _runner.Recipe
    TextConditioner = _text_conditioning.TextConditioner
    TextConditioningProvenance = _text_conditioning.TextConditioningProvenance
    VideoVAEDecodeDType = _types.VideoVAEDecodeDType
    VideoVAETilingConfig = _types.VideoVAETilingConfig
    generate_distilled = _distilled.generate_distilled
    prepare_resources = _resources.prepare_resources
    restart_distilled = _restart.restart_distilled


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return cast(object, getattr(import_module(module_name), name))


def __dir__() -> list[str]:
    """Expose lazy public names to completion and introspection tools."""
    return sorted(set(globals()) | set(__all__))
