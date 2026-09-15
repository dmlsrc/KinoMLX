"""Used weight groups follow the same modality branches as transformer blocks."""

import pytest

from kinomlx.models.ltx2.transformer.execution import TransformerExecution


@pytest.mark.parametrize(
    ("present", "enabled", "groups"),
    [
        ((True, False), (True, True), {"video", "video_input"}),
        ((False, True), (True, True), {"audio", "audio_input"}),
        (
            (True, True),
            (True, True),
            {"video", "audio", "video_input", "audio_input", "a2v", "v2a", "cross"},
        ),
        ((True, True), (True, False), {"video", "video_input", "audio_input", "a2v", "cross"}),
        ((True, True), (False, True), {"audio", "video_input", "audio_input", "v2a", "cross"}),
        ((True, True), (False, False), set()),
    ],
)
def test_weight_groups_cover_absent_and_context_only_modalities(present, enabled, groups):
    execution = TransformerExecution.resolve(
        video_present=present[0],
        audio_present=present[1],
        video_enabled=enabled[0],
        audio_enabled=enabled[1],
    )
    examples = {
        "video": (
            "adaln_single.linear.weight",
            "proj_out.weight",
            "transformer_blocks.0.ff.project_out.weight",
            "transformer_blocks.0.attn1.to_q.weight",
        ),
        "audio": (
            "audio_adaln_single.linear.weight",
            "audio_proj_out.weight",
            "transformer_blocks.0.audio_ff.project_out.weight",
            "transformer_blocks.0.audio_attn2.to_k.weight",
        ),
        "video_input": ("patchify_proj.weight",),
        "audio_input": ("audio_patchify_proj.weight",),
        "a2v": (
            "av_ca_a2v_gate_adaln_single.linear.weight",
            "transformer_blocks.0.audio_to_video_attn.to_k.weight",
        ),
        "v2a": (
            "av_ca_v2a_gate_adaln_single.linear.weight",
            "transformer_blocks.0.video_to_audio_attn.to_q.weight",
        ),
        "cross": (
            "av_ca_audio_scale_shift_adaln_single.linear.weight",
            "av_ca_video_scale_shift_adaln_single.linear.weight",
            "transformer_blocks.0.scale_shift_table_a2v_ca_audio",
        ),
    }
    for group, names in examples.items():
        for name in names:
            assert execution.uses_weight(name) == (group in groups), name
            if name.endswith(".weight"):
                assert execution.uses_weight("__layout__." + name + "_t") == (group in groups), name
    assert execution.uses_weight("custom_projection.weight")


def test_keyframe_weight_requires_a_used_video_stream_and_mask():
    args = {
        "video_present": True,
        "audio_present": True,
        "video_enabled": False,
        "audio_enabled": True,
    }
    assert not TransformerExecution.resolve(**args).uses_weight("keyframes_abs_pos_embedding")
    assert TransformerExecution.resolve(**args, video_keyframes=True).uses_weight(
        "keyframes_abs_pos_embedding"
    )
