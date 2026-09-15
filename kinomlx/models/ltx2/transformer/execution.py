"""Metadata-only modality branches and their consumed transformer weight groups."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerExecution:
    video: bool
    audio: bool
    audio_to_video: bool
    video_to_audio: bool
    video_keyframes: bool = False

    @classmethod
    def resolve(
        cls,
        *,
        video_present: bool,
        audio_present: bool,
        video_enabled: bool,
        audio_enabled: bool,
        video_keyframes: bool = False,
    ) -> TransformerExecution:
        video = video_present and video_enabled
        audio = audio_present and audio_enabled
        return cls(video, audio, video and audio_present, audio and video_present, video_keyframes)

    def uses_weight(self, name: str) -> bool:
        """Select logical targets, including context-only streams and baked layouts.

        Unknown targets stay checked for custom ports or new parameter families;
        known inactive groups are the only targets deferred.
        """
        name = name.removeprefix("__layout__.")
        parts = name.split(".")
        cross = self.audio_to_video or self.video_to_audio
        if parts[0] == "transformer_blocks" and len(parts) > 2:
            group = parts[2]
            if group in {"attn1", "attn2", "ff", "scale_shift_table", "prompt_scale_shift_table"}:
                return self.video
            if group in {
                "audio_attn1",
                "audio_attn2",
                "audio_ff",
                "audio_scale_shift_table",
                "audio_prompt_scale_shift_table",
            }:
                return self.audio
            if group == "audio_to_video_attn":
                return self.audio_to_video
            if group == "video_to_audio_attn":
                return self.video_to_audio
            if group in {"scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video"}:
                return cross
        else:
            group = parts[0]
            if group == "patchify_proj":
                return self.video or self.video_to_audio
            if group == "audio_patchify_proj":
                return self.audio or self.audio_to_video
            if group == "keyframes_abs_pos_embedding":
                return self.video_keyframes and (self.video or self.video_to_audio)
            if group in {
                "adaln_single",
                "prompt_adaln_single",
                "proj_out",
                "norm_out",
                "scale_shift_table",
            }:
                return self.video
            if group in {
                "audio_adaln_single",
                "audio_prompt_adaln_single",
                "audio_proj_out",
                "audio_norm_out",
                "audio_scale_shift_table",
            }:
                return self.audio
            if group in {
                "av_ca_video_scale_shift_adaln_single",
                "av_ca_audio_scale_shift_adaln_single",
            }:
                return cross
            if group == "av_ca_a2v_gate_adaln_single":
                return self.audio_to_video
            if group == "av_ca_v2a_gate_adaln_single":
                return self.video_to_audio
        return True
