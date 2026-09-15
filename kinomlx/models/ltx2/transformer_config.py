"""Immutable constructor facts for the LTX-2 transformer graph."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConstructorConfig:
    """Graph facts shared by metadata inspection and resource construction."""

    model_generation: str
    declared_model_version: str | None
    num_layers: int
    video_in_channels: int
    video_out_channels: int
    video_heads: int
    video_head_dim: int
    audio_heads: int
    audio_head_dim: int
    audio_out_channels: int
    video_context_dim: int
    audio_context_dim: int
    caption_channels: int
    video_max_pos: tuple[int, int, int]
    audio_max_pos: tuple[int]
    positional_embedding_theta: float
    timestep_scale_multiplier: float
    av_ca_timestep_scale_multiplier: float
    norm_eps: float
    ff_bias: bool
    audio_ff_bias: bool
    use_keyframes_abs_pos_embedding: bool
    use_prompt_adaln_single: bool
    config_digest: str
    inferred_fields: tuple[str, ...] = ()

    @property
    def video_hidden_dim(self) -> int:
        return self.video_heads * self.video_head_dim

    @property
    def audio_hidden_dim(self) -> int:
        return self.audio_heads * self.audio_head_dim


__all__ = ["TransformerConstructorConfig"]
