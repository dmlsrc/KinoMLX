"""Safe, torch-free checkpoint conversion with model-specific extensions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

_EXPORT_MODULES = {
    "GenericConversionReceipt": "kinomlx.weights.convert",
    "RestrictedCheckpoint": "kinomlx.weights.torch_checkpoint",
    "RestrictedCheckpointError": "kinomlx.weights.torch_checkpoint",
    "WeightConversionError": "kinomlx.weights.convert",
    "convert_checkpoint": "kinomlx.weights.convert",
    "load_restricted_checkpoint": "kinomlx.weights.torch_checkpoint",
    "scan_pickle_globals": "kinomlx.weights.torch_checkpoint",
    "suspicious_globals": "kinomlx.weights.torch_checkpoint",
}

__all__ = sorted(_EXPORT_MODULES)


if TYPE_CHECKING:
    from . import convert as _convert
    from . import torch_checkpoint as _torch_checkpoint

    # Static mirrors of the lazy map preserve exact public types for mypy.
    GenericConversionReceipt = _convert.GenericConversionReceipt
    RestrictedCheckpoint = _torch_checkpoint.RestrictedCheckpoint
    RestrictedCheckpointError = _torch_checkpoint.RestrictedCheckpointError
    WeightConversionError = _convert.WeightConversionError
    convert_checkpoint = _convert.convert_checkpoint
    load_restricted_checkpoint = _torch_checkpoint.load_restricted_checkpoint
    scan_pickle_globals = _torch_checkpoint.scan_pickle_globals
    suspicious_globals = _torch_checkpoint.suspicious_globals


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return cast(object, getattr(import_module(module_name), name))


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
