"""GMNet - learned gain-map prediction for SDR-to-HDR inverse tone mapping.

An independent MLX reimplementation of the GMNet architecture from
"Learning Gain Map for Inverse Tone Mapping" (Liao et al., ICLR 2025).
The network predicts a normalized gain map plus a Qmax scalar over a
display-referred SDR still; the package reconstructs scene-linear HDR
from them and hands the result to the native EXR / PQ-HEIC terminals.

Self-contained per the multi-model convention: no imports from other
model packages. Selected from the main CLI as ``kinomlx --model gmnet``.
Exports resolve lazily so importing the package (and its CLI module)
stays free of MLX until a model or expansion symbol is actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

_EXPORT_MODULES = {
    "ExpansionResult": "kinomlx.models.gmnet.expand",
    "GMNet": "kinomlx.models.gmnet.net",
    "GMNetArtifactSet": "kinomlx.models.gmnet.output",
    "GMNetComponents": "kinomlx.models.gmnet.components",
    "GMNetError": "kinomlx.models.gmnet.runner",
    "GMNetOutputConfig": "kinomlx.models.gmnet.types",
    "GMNetOutputError": "kinomlx.models.gmnet.output",
    "GMNetOutputPlan": "kinomlx.models.gmnet.output",
    "GMNetOutputReservation": "kinomlx.models.gmnet.output",
    "GMNetOutputSink": "kinomlx.models.gmnet.output",
    "GMNetRequest": "kinomlx.models.gmnet.types",
    "GMNetResources": "kinomlx.models.gmnet.resources",
    "GMNetRunner": "kinomlx.models.gmnet.runner",
    "GMNetSettings": "kinomlx.models.gmnet.settings",
    "GMNetVariant": "kinomlx.models.gmnet.catalog",
    "GMNetVariantSpec": "kinomlx.models.gmnet.catalog",
    "NativeGMNetComponents": "kinomlx.models.gmnet.components",
    "expand_image": "kinomlx.models.gmnet.expand",
    "expand_gmnet": "kinomlx.models.gmnet.pipeline",
    "load_gmnet_weights": "kinomlx.models.gmnet.net",
    "plan_gmnet_output": "kinomlx.models.gmnet.output",
    "prepare_resources": "kinomlx.models.gmnet.resources",
    "reconstruct_linear_hdr": "kinomlx.models.gmnet.expand",
    "resolve_variant_weights": "kinomlx.models.gmnet.catalog",
    "variant_spec": "kinomlx.models.gmnet.catalog",
    "write_gmnet_output": "kinomlx.models.gmnet.output",
    "write_gain_map_sidecar": "kinomlx.models.gmnet.expand",
}

__all__ = sorted(_EXPORT_MODULES)


if TYPE_CHECKING:
    from . import catalog as _catalog
    from . import components as _components
    from . import expand as _expand
    from . import net as _net
    from . import output as _output
    from . import pipeline as _pipeline
    from . import resources as _resources
    from . import runner as _runner
    from . import settings as _settings
    from . import types as _types

    # Static mirrors of the lazy map preserve exact public types for mypy.
    ExpansionResult = _expand.ExpansionResult
    GMNet = _net.GMNet
    GMNetArtifactSet = _output.GMNetArtifactSet
    GMNetComponents = _components.GMNetComponents
    GMNetError = _runner.GMNetError
    GMNetOutputConfig = _types.GMNetOutputConfig
    GMNetOutputError = _output.GMNetOutputError
    GMNetOutputPlan = _output.GMNetOutputPlan
    GMNetOutputReservation = _output.GMNetOutputReservation
    GMNetOutputSink = _output.GMNetOutputSink
    GMNetRequest = _types.GMNetRequest
    GMNetResources = _resources.GMNetResources
    GMNetRunner = _runner.GMNetRunner
    GMNetSettings = _settings.GMNetSettings
    GMNetVariant = _catalog.GMNetVariant
    GMNetVariantSpec = _catalog.GMNetVariantSpec
    NativeGMNetComponents = _components.NativeGMNetComponents
    expand_image = _expand.expand_image
    expand_gmnet = _pipeline.expand_gmnet
    load_gmnet_weights = _net.load_gmnet_weights
    plan_gmnet_output = _output.plan_gmnet_output
    prepare_resources = _resources.prepare_resources
    reconstruct_linear_hdr = _expand.reconstruct_linear_hdr
    resolve_variant_weights = _catalog.resolve_variant_weights
    variant_spec = _catalog.variant_spec
    write_gain_map_sidecar = _expand.write_gain_map_sidecar
    write_gmnet_output = _output.write_gmnet_output


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return cast(object, getattr(import_module(module_name), name))


def __dir__() -> list[str]:
    """Expose lazy public names to completion and introspection tools."""
    return sorted(set(globals()) | set(__all__))
