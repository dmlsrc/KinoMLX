"""Pure LTX HDR output restrictions shared by the CLI and direct output planning."""

from typing import cast

from kinomlx.config_rules import RecordRule, RuleInputs


def _hdr_output(values: RuleInputs) -> None:
    hdr = cast(bool, values.context["hdr"])
    fps = cast(float, values.context["fps"])
    if values["save_hdr_heic_frames"] and not hdr:
        raise ValueError("save_hdr_heic_frames requires HDR generation")
    if not hdr:
        return
    if values["vsr_spatial_mode"] != "off":
        raise ValueError("vsr_spatial_mode is not yet supported for HDR")
    target = cast(float | None, values["target_fps"])
    if target is not None and abs(target - fps) > 1e-6:
        raise ValueError("target_fps is not yet supported for HDR")
    if values["save_vae_frames"]:
        raise ValueError(
            "save_vae_frames is an SDR PNG diagnostic; "
            "HDR already writes a lossless EXR frame sequence"
        )
    # Save-all's original-video implication is inert in the HDR delivery path;
    # a direct original-video request still has no HDR-safe VSR/FRC sink.
    if values["vsr_save_original"] and not values["save_all_sidecars"]:
        raise ValueError("vsr_save_original requires an HDR-safe VSR/FRC path")


HDR_OUTPUT = RecordRule(
    "hdr_output",
    frozenset(
        {
            "vsr_spatial_mode",
            "target_fps",
            "save_hdr_heic_frames",
            "save_vae_frames",
            "vsr_save_original",
            "save_all_sidecars",
        }
    ),
    _hdr_output,
    phase="capability",
    context_reads=frozenset({"hdr", "fps"}),
)


__all__ = ["HDR_OUTPUT"]
