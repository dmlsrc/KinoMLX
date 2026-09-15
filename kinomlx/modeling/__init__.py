"""Model-neutral construction and checkpoint-binding helpers."""

from .layers import conv2d_shell, embedding_shell, linear_shell
from .loading import (
    WeightBinding,
    WeightTarget,
    bind_resolved_weights,
    bind_weight_targets,
    resolve_weight_bindings,
    weight_files,
)

__all__ = [
    "WeightBinding",
    "WeightTarget",
    "bind_resolved_weights",
    "bind_weight_targets",
    "conv2d_shell",
    "embedding_shell",
    "linear_shell",
    "resolve_weight_bindings",
    "weight_files",
]
