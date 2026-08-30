"""VideoToolbox-backed video post-processing.

This subpackage bridges MLX-decoded video frames to Apple's hardware video
pipeline: VideoToolbox Super Resolution (`VsrSession`), Frame Rate
Conversion (`VtfrcSession`), and AVAssetWriter (`AVWriter`). All modules
under this namespace require the mandatory PyObjC framework dependencies.
A missing framework wrapper is therefore an incomplete installation and fails
immediately at import.

Public surface:

    from kinomlx.videotoolbox import (
        VsrSession,        # spatial upscale via VTSuperResolutionScaler*
        VtfrcSession,      # temporal frame-rate conversion via VTFrameRateConversion*
        AVWriter,          # HEVC + audio encoder via AVAssetWriter
        AudioTrack,        # in-memory PCM -> CMSampleBuffer wrapper
        CutDetector,       # scene-cut policy for native frame history
        FrameSource,       # accepted MLX batch/sequence/iterator contract
        ProgressStack,     # optional caller-owned progress protocol
        encode_video_videotoolbox,
    )

Submodules expose lower-level helpers:

    pixel_buffers   CVPixelBuffer create/read/write, CMTime helpers
    comparison      Side-by-side composite for `comparison.mp4`

The stacked progress-bar primitives are injected by the caller (KinoMLX
uses a rich-based UI); the VT encoder no longer imports them directly.
"""

from __future__ import annotations

from .audio import AudioTrack
from .cut_detect import CutDetector
from .encode import FrameSource, ProgressStack, encode_video_videotoolbox
from .errors import VideoToolboxError, VideoToolboxUnavailableError
from .temporal import VtfrcSession
from .vsr import VsrSession
from .writer import AVWriter

__all__ = [
    "AVWriter",
    "AudioTrack",
    "CutDetector",
    "FrameSource",
    "ProgressStack",
    "VsrSession",
    "VtfrcSession",
    "VideoToolboxError",
    "VideoToolboxUnavailableError",
    "encode_video_videotoolbox",
]
