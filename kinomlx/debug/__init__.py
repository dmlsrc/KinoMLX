"""Opt-in reproducibility artifacts written at orchestration boundaries."""

from .memory import capture_mlx_memory_counters, create_mlx_memory_sampler
from .metadata import (
    EXECUTION_SIDECAR_SPECS,
    ExecutionSidecarSpec,
    RunRecord,
    SidecarError,
    SidecarPaths,
    execution_sidecar_paths,
    initialize_execution_log,
    normalized_video_path,
    selected_execution_sidecar_paths,
    sidecar_failure,
    sidecar_selected,
    write_effective_config,
)
from .sidecars import SidecarArtifactSink

__all__ = [
    "EXECUTION_SIDECAR_SPECS",
    "ExecutionSidecarSpec",
    "RunRecord",
    "SidecarArtifactSink",
    "SidecarError",
    "SidecarPaths",
    "capture_mlx_memory_counters",
    "create_mlx_memory_sampler",
    "execution_sidecar_paths",
    "initialize_execution_log",
    "normalized_video_path",
    "selected_execution_sidecar_paths",
    "sidecar_selected",
    "sidecar_failure",
    "write_effective_config",
]
