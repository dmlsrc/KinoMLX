"""LTX field-owned syntax and final policy retain distinct admission phases."""

import re

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_rules import record_rules
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.settings import LTX2Settings


@pytest.mark.parametrize(
    ("invalid", "repair", "message"),
    [
        (
            {"transformer_retain": ("blocks.*",)},
            {"stream_transformer": True},
            "transformer_retain requires transformer streaming",
        ),
        (
            {"adapter_accumulation": "float16", "transformer_dtype": "bfloat16"},
            {"transformer_dtype": "float16"},
            "adapter_accumulation=float16 requires transformer_dtype=float16",
        ),
        (
            {"audio_ff_layout_specs": ("project_out:pretranspose",)},
            {"audio_layout_mirror": False},
            "explicit audio layout settings require audio_layout_mirror=false",
        ),
        (
            {"transformer_quantize_exclude_layers": (0,)},
            {"transformer_quantize": "mxfp8-blocks"},
            "transformer_quantize_exclude_layers requires whole-block quantization",
        ),
        (
            {"transformer_compile_group_size": 2},
            {"transformer_resident_blocks": 8},
            "transformer_compile_group_size requires transformer_resident_blocks",
        ),
    ],
)
def test_each_ltx_policy_runs_after_local_admission_and_can_be_repaired(invalid, repair, message):
    settings = LTX2Settings(**invalid)
    with pytest.raises(ValueError, match=rf"\[model_settings\]: {re.escape(message)}"):
        settings.validate()
    LTX2Settings(**{**invalid, **repair}).validate()


def test_all_ltx_settings_policies_have_one_declared_identity():
    assert {rule.name for rule in record_rules(LTX2Settings)} == {
        "retention",
        "adapter_precision",
        "audio_layout",
        "quantization",
        "compile_groups",
    }


@pytest.mark.parametrize(
    "specs",
    [
        ("unknown:pretranspose",),
        ("project_out:unknown",),
        ("project_out",),
        ("project_out:pretranspose", "project_out:pretranspose"),
    ],
)
def test_layout_syntax_is_a_local_constructor_constraint(specs):
    with pytest.raises(ValueError, match="video_ff_layout_specs"):
        LTX2Settings(video_ff_layout_specs=specs)


def test_malformed_lower_layer_spec_cannot_be_hidden(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[model_settings]\nvideo_ff_layout_specs = ["unknown:pretranspose"]\n')
    options = build_parser().parse_args(
        [
            "--config",
            str(path),
            "--video-ff-layout-specs",
            "project_out:pretranspose",
            "--print-config",
        ]
    )
    with pytest.raises(ConfigError, match="video_ff_layout_specs"):
        assemble(options)


def test_later_layer_can_repair_valid_but_conflicting_audio_layout(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[model_settings]\naudio_ff_layout_specs = ["project_out:pretranspose"]\n')
    options = build_parser().parse_args(
        ["--config", str(path), "--no-audio-layout-mirror", "--print-config"]
    )
    invocation = assemble(options)
    assert invocation.model_settings.audio_layout_mirror is False
    assert invocation.model_settings.audio_ff_layout_specs == ("project_out:pretranspose",)
