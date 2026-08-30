"""KinoMLX - native multimodal MLX inference on Apple Silicon.

See README.md for an overview. Heavy model objects are imported lazily so
``import kinomlx`` remains a lightweight settings and discovery operation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from . import output as _output
from . import settings as _settings

ArtifactSet = _output.ArtifactSet
GenerationSink = _output.GenerationSink
HDRGenerationSink = _output.HDRGenerationSink
Settings = _settings.Settings
VideoToolboxGenerationSink = _output.VideoToolboxGenerationSink

__version__ = "0.0.1"

_EXPORT_TARGETS: dict[str, tuple[str, str] | None] = {
    "ArtifactSet": None,
    "DistilledRequest": ("kinomlx.models.ltx2", "DistilledRequest"),
    "DistilledRestart": ("kinomlx.models.ltx2", "DistilledRestart"),
    "ExpansionResult": ("kinomlx.models.gmnet", "ExpansionResult"),
    "GenerationOutput": ("kinomlx.models.ltx2", "GenerationOutput"),
    "GenerationSink": None,
    "GMNetArtifactSet": ("kinomlx.models.gmnet", "GMNetArtifactSet"),
    "GMNetError": ("kinomlx.models.gmnet", "GMNetError"),
    "GMNetOutputConfig": ("kinomlx.models.gmnet", "GMNetOutputConfig"),
    "GMNetOutputError": ("kinomlx.models.gmnet", "GMNetOutputError"),
    "GMNetOutputPlan": ("kinomlx.models.gmnet", "GMNetOutputPlan"),
    "GMNetOutputReservation": ("kinomlx.models.gmnet", "GMNetOutputReservation"),
    "GMNetOutputSink": ("kinomlx.models.gmnet", "GMNetOutputSink"),
    "GMNetRequest": ("kinomlx.models.gmnet", "GMNetRequest"),
    "GMNetResources": ("kinomlx.models.gmnet", "GMNetResources"),
    "GMNetRunner": ("kinomlx.models.gmnet", "GMNetRunner"),
    "GMNetSettings": ("kinomlx.models.gmnet", "GMNetSettings"),
    "HDRAuthoring": ("kinomlx.models.ltx2", "HDRAuthoring"),
    "HDRGenerationSink": None,
    "HDRReferenceConditioningConfig": (
        "kinomlx.models.ltx2",
        "HDRReferenceConditioningConfig",
    ),
    "ImageConditioningConfig": ("kinomlx.models.ltx2", "ImageConditioningConfig"),
    "LTX2Runner": ("kinomlx.models.ltx2", "LTX2Runner"),
    "LTX2Settings": ("kinomlx.models.ltx2", "LTX2Settings"),
    "Settings": None,
    "VideoToolboxGenerationSink": None,
    "__version__": None,
    "expand_gmnet": ("kinomlx.models.gmnet", "expand_gmnet"),
    "generate_distilled": ("kinomlx.models.ltx2", "generate_distilled"),
    "plan_gmnet_output": ("kinomlx.models.gmnet", "plan_gmnet_output"),
    "prepare_gmnet_resources": ("kinomlx.models.gmnet", "prepare_resources"),
    "prepare_resources": ("kinomlx.models.ltx2", "prepare_resources"),
    "restart_distilled": ("kinomlx.models.ltx2", "restart_distilled"),
    "write_gmnet_output": ("kinomlx.models.gmnet", "write_gmnet_output"),
}

__all__ = sorted(_EXPORT_TARGETS)

if TYPE_CHECKING:
    import kinomlx.models.gmnet as _gmnet
    import kinomlx.models.ltx2 as _ltx2

    # Static mirrors of the lazy targets preserve exact top-level API types.
    DistilledRequest = _ltx2.DistilledRequest
    DistilledRestart = _ltx2.DistilledRestart
    ExpansionResult = _gmnet.ExpansionResult
    GenerationOutput = _ltx2.GenerationOutput
    GMNetArtifactSet = _gmnet.GMNetArtifactSet
    GMNetError = _gmnet.GMNetError
    GMNetOutputConfig = _gmnet.GMNetOutputConfig
    GMNetOutputError = _gmnet.GMNetOutputError
    GMNetOutputPlan = _gmnet.GMNetOutputPlan
    GMNetOutputReservation = _gmnet.GMNetOutputReservation
    GMNetOutputSink = _gmnet.GMNetOutputSink
    GMNetRequest = _gmnet.GMNetRequest
    GMNetResources = _gmnet.GMNetResources
    GMNetRunner = _gmnet.GMNetRunner
    GMNetSettings = _gmnet.GMNetSettings
    HDRAuthoring = _ltx2.HDRAuthoring
    HDRReferenceConditioningConfig = _ltx2.HDRReferenceConditioningConfig
    ImageConditioningConfig = _ltx2.ImageConditioningConfig
    LTX2Runner = _ltx2.LTX2Runner
    LTX2Settings = _ltx2.LTX2Settings
    expand_gmnet = _gmnet.expand_gmnet
    generate_distilled = _ltx2.generate_distilled
    plan_gmnet_output = _gmnet.plan_gmnet_output
    prepare_gmnet_resources = _gmnet.prepare_resources
    prepare_resources = _ltx2.prepare_resources
    restart_distilled = _ltx2.restart_distilled
    write_gmnet_output = _gmnet.write_gmnet_output


def __getattr__(name: str) -> object:
    target = _EXPORT_TARGETS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return cast(object, getattr(import_module(module_name), attribute_name))


def __dir__() -> list[str]:
    """Expose lazy public names to completion and introspection tools."""
    return sorted(set(globals()) | set(__all__))
