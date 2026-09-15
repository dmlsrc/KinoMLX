"""Protocol for one latent-state conditioning operation."""

from typing import Protocol

from ..types import LatentState
from .tools import VideoLatentTools


class EncodedCondition(Protocol):
    """Pure encoded operation over one patchified latent state."""

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Return the conditioned state."""
        ...


__all__ = ["EncodedCondition"]
