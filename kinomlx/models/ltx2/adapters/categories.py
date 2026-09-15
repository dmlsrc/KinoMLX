"""LTX-2 adapter exclusion categories."""

ADAPTER_EXCLUDE_CATEGORIES = frozenset(
    {
        "video",
        "audio",
        "cross",
        "attn",
        "gate",
        "ff",
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "video_to_audio_attn",
        "audio_to_video_attn",
        "audio_ff",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_gate_logits",
        "project_in",
        "project_out",
        "adaln",
        "prompt_adaln",
        "scale_shift",
        "prompt_scale_shift",
        "gate_adaln",
        "av_ca",
        "cross_control",
        "distill_control",
    }
)

__all__ = ["ADAPTER_EXCLUDE_CATEGORIES"]
