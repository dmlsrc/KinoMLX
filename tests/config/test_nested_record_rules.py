"""Final semantic validation descends through records and indexed collections."""

from dataclasses import dataclass, field

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_constraints import field_choices
from kinomlx.config_records import config_record, replace_config
from kinomlx.config_rules import RecordRule, validate_record
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.config_schema import CONFIG_SCHEMA
from kinomlx.models.ltx2.types import (
    LTX2ImageConditioningConfig,
    LTX2Request,
    LTX2VideoVAETilingConfig,
    TilingGridConstraint,
)


def ordered(values):
    if values["start"] > values["end"]:
        raise ValueError("start must not exceed end")


@config_record
@dataclass(frozen=True)
class Segment:
    start: int = field(
        default=0,
        metadata={"config_rules": (RecordRule("ordered", frozenset({"start", "end"}), ordered),)},
    )
    end: int = 1


@config_record
@dataclass(frozen=True)
class Timeline:
    segments: tuple[Segment, ...] = ()


def test_nested_collection_reports_exact_index_and_allows_repaired_final_record():
    timeline = Timeline(segments=[{}, {"start": 3, "end": 2}])
    with pytest.raises(ValueError, match=r"\[timeline\].segments\[1\]: "):
        validate_record(timeline, location="[timeline]")
    repaired = replace_config(timeline.segments[1], end=4)
    validate_record(
        replace_config(timeline, segments=(timeline.segments[0], repaired)),
        location="[timeline]",
    )


def test_ltx_nested_rule_runs_at_generation_boundary_before_file_checks():
    request = LTX2Request(
        text_conditioning="missing.safetensors",
        vae_tiling=LTX2VideoVAETilingConfig(mode="custom", temporal_overlap_frames=256),
    )
    with pytest.raises(ValueError, match=r"\[generate\].vae_tiling: .*overlap"):
        request.validate_for_generation()


def test_ltx_nested_rule_runs_during_configuration_print_assembly():
    options = build_parser().parse_args(
        [
            "--set",
            "generate.vae_tiling.mode=custom",
            "--set",
            "generate.vae_tiling.temporal_overlap_frames=256",
            "--print-config",
        ]
    )
    with pytest.raises(ConfigError, match="vae_tiling.*overlap"):
        assemble(options)


def test_later_layer_can_repair_semantic_tiling_conflict(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[generate.vae_tiling]\nmode = "custom"\ntemporal_tile_frames = 32\ntemporal_overlap_frames = 32\n'
    )
    options = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--set",
            "generate.vae_tiling.temporal_overlap_frames=24",
            "--print-config",
        ]
    )
    invocation = assemble(options)
    plan = invocation.request.vae_tiling.to_runtime_config()
    assert plan.temporal_config.chunk_size_in_frames == 32


@pytest.mark.parametrize(
    "values",
    [
        {"temporal_tile_frames": 8},
        {"temporal_tile_frames": 17},
        {"temporal_overlap_frames": 7},
        {"spatial_tile_pixels": 32},
        {"spatial_tile_pixels": 65},
        {"spatial_overlap_pixels": -32},
    ],
)
def test_tiling_grid_errors_are_local_admission_errors(values):
    with pytest.raises(ValueError, match=next(iter(values))):
        LTX2VideoVAETilingConfig(**values)


@pytest.mark.parametrize(
    "values", [{"grid": 0}, {"grid": True}, {"grid": 8, "nonzero_minimum": -1}]
)
def test_invalid_tiling_grid_declarations_fail(values):
    with pytest.raises(ValueError, match="must be"):
        TilingGridConstraint(**values)


@pytest.mark.parametrize(
    "name",
    ["noise_backend", "stage_1_sampler", "stage_2_sampler", "hdr", "vae_decode_dtype"],
)
def test_ltx_request_choices_are_owned_by_fields_and_bound_by_schema(name):
    assert CONFIG_SCHEMA.field(("generate", name)).choices == field_choices(LTX2Request, name)
    with pytest.raises(ValueError, match=name):
        LTX2Request(**{name: "unknown"})


def test_ltx_prompt_policy_can_be_repaired_before_final_validation():
    request = LTX2Request(prompt="hello", prompt_unconditional=True)
    with pytest.raises(ValueError, match=r"\[generate\]: prompt_unconditional cannot"):
        validate_record(request, location="[generate]")
    validate_record(replace_config(request, prompt_unconditional=False), location="[generate]")


def test_image_frame_bounds_can_be_repaired_before_final_validation():
    request = LTX2Request(
        frames=9, images=(LTX2ImageConditioningConfig("frame.png", frame_index=9),)
    )
    with pytest.raises(ValueError, match=r"\[generate\]: image frame_index"):
        validate_record(request, location="[generate]")
    validate_record(replace_config(request, frames=17), location="[generate]")


def test_higher_layer_repairs_image_frame_relationship(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[generate]\nframes = 9\nimages = [{path = "frame.png", frame_index = 9}]\n')
    options = build_parser().parse_args(
        ["--config", str(config), "--frames", "17", "--print-config"]
    )
    invocation = assemble(options)
    assert invocation.request.frames == 17
    assert invocation.request.images[0].frame_index == "9"


@pytest.mark.parametrize(
    "values",
    [
        {"prompt": "hello", "prompt_unconditional": True},
        {"frames": 9, "images": ({"path": "frame.png", "frame_index": 9},)},
    ],
)
def test_direct_restart_validates_final_request_before_artifact_checks(tmp_path, values):
    from types import SimpleNamespace

    from kinomlx.memory_policy import MLXMemoryPolicy
    from kinomlx.models.ltx2.pipelines.restart import restart_ltx2
    from kinomlx.models.ltx2.restart import LTX2Restart

    request = LTX2Request(**values)
    resources = SimpleNamespace(memory_policy=MLXMemoryPolicy())
    restart = LTX2Restart.decode(tmp_path / "missing.safetensors")
    with pytest.raises(
        ValueError, match=r"\[generate\]: (?:prompt_unconditional cannot|image frame_index)"
    ):
        restart_ltx2(request, resources, restart=restart)
