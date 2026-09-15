"""KinoMLX shared infrastructure for native MLX inference on Apple Silicon.

Model APIs live under :mod:`kinomlx.models`; keeping them out of this root
namespace prevents the first integrated model from defining generic-looking
names for every later model.
"""

from .memory_policy import MLXMemoryPolicy, apply_memory_policy, memory_managed
from .output import (
    ArtifactSet,
    GenerationSink,
    HDRGenerationSink,
    VideoToolboxGenerationSink,
)
from .settings import Settings

__version__ = "0.0.2"

__all__ = [
    "ArtifactSet",
    "GenerationSink",
    "HDRGenerationSink",
    "MLXMemoryPolicy",
    "Settings",
    "VideoToolboxGenerationSink",
    "__version__",
    "apply_memory_policy",
    "memory_managed",
]
