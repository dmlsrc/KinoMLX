"""Pure final resource policies owned by LTX Settings fields."""

from collections.abc import Mapping
from typing import cast

from kinomlx.config_rules import RecordRule


def _text_sources(values: Mapping[str, object]) -> None:
    if values["gemma_path"] is not None and values["text_encoder_path"] is not None:
        raise ValueError("gemma_path and text_encoder_path cannot be combined")


def _retention(values: Mapping[str, object]) -> None:
    if (
        values["transformer_retain"]
        and not values["stream_transformer"]
        and values["transformer_resident_blocks"] is None
    ):
        raise ValueError("transformer_retain requires transformer streaming")


def _adapter_precision(values: Mapping[str, object]) -> None:
    if values["adapter_accumulation"] == "float16" and values["transformer_dtype"] != "float16":
        raise ValueError("adapter_accumulation=float16 requires transformer_dtype=float16")


def _attention_probe(values: Mapping[str, object]) -> None:
    if values["steel_attention_probe"] and not values["steel_attention"]:
        raise ValueError("steel_attention_probe requires steel_attention=true")


def _audio_layout(values: Mapping[str, object]) -> None:
    if values["audio_layout_mirror"] and any(
        value for name, value in values.items() if name != "audio_layout_mirror"
    ):
        raise ValueError("explicit audio layout settings require audio_layout_mirror=false")


def _quantization(values: Mapping[str, object]) -> None:
    if values["transformer_quantize_exclude_layers"] and values["transformer_quantize"] == "off":
        raise ValueError("transformer_quantize_exclude_layers requires whole-block quantization")
    specs = values["video_ff_quantize_specs"]
    if values["transformer_quantize"] != "off" and specs:
        raise ValueError("whole-transformer and targeted FF quantization cannot be combined")
    if values["video_ff_quantize_group_size"] is not None and not specs:
        raise ValueError("video_ff_quantize_group_size requires video_ff_quantize_specs")
    if values["video_ff_quantize_bits"] is not None and not specs:
        raise ValueError("video_ff_quantize_bits requires video_ff_quantize_specs")


def _compile_groups(values: Mapping[str, object]) -> None:
    group = cast(int | None, values["transformer_compile_group_size"])
    resident = cast(int | None, values["transformer_resident_blocks"])
    if group is not None and resident is None:
        raise ValueError("transformer_compile_group_size requires transformer_resident_blocks")
    if group is not None and resident is not None and group > resident:
        raise ValueError("transformer_compile_group_size cannot exceed transformer_resident_blocks")


TEXT_SOURCES = RecordRule(
    "text_sources", frozenset({"gemma_path", "text_encoder_path"}), _text_sources
)
RETENTION = RecordRule(
    "retention",
    frozenset({"transformer_retain", "stream_transformer", "transformer_resident_blocks"}),
    _retention,
)
ADAPTER_PRECISION = RecordRule(
    "adapter_precision",
    frozenset({"adapter_accumulation", "transformer_dtype"}),
    _adapter_precision,
)
ATTENTION_PROBE = RecordRule(
    "attention_probe", frozenset({"steel_attention_probe", "steel_attention"}), _attention_probe
)
AUDIO_LAYOUT = RecordRule(
    "audio_layout",
    frozenset(
        {
            "audio_layout_mirror",
            "audio_ff_layout_specs",
            "audio_ff_layout_layers",
            "audio_attn_layout_specs",
            "audio_attn_layout_layers",
        }
    ),
    _audio_layout,
)
QUANTIZATION = RecordRule(
    "quantization",
    frozenset(
        {
            "transformer_quantize",
            "transformer_quantize_exclude_layers",
            "video_ff_quantize_specs",
            "video_ff_quantize_group_size",
            "video_ff_quantize_bits",
        }
    ),
    _quantization,
)
COMPILE_GROUPS = RecordRule(
    "compile_groups",
    frozenset({"transformer_compile_group_size", "transformer_resident_blocks"}),
    _compile_groups,
)
