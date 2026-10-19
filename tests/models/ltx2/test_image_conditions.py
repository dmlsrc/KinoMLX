"""The ordered LTX-2 image-condition collection across every public surface."""

import contextlib
import io
import json
import tomllib
from pathlib import Path

import pytest

from kinomlx.cli.config_templates import config_template
from kinomlx.cli.main import main
from kinomlx.config import ConfigError
from kinomlx.config.inspection import configuration_inventory
from kinomlx.model_registry import config_registry
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.frame_position import (
    FramePositionConstraint,
    is_first_position,
    parse_frame_position,
    resolve_frame_position,
    resolve_frame_positions,
)
from kinomlx.models.ltx2.image_arguments import IMAGE_RECORD, ImageShorthand
from kinomlx.models.ltx2.pipelines.distilled import (
    condition_sources,
    requested_condition_families,
)
from kinomlx.models.ltx2.types import LTX2ImageConditioningConfig, LTX2Request
from kinomlx.settings import Settings
from tests.cli.test_restart import _assemble, _source_run

_SHORTHAND = ImageShorthand(IMAGE_RECORD)


def _images(*arguments: str):
    options = build_parser().parse_args([*arguments, "--print-config"])
    return assemble(options, base_settings=Settings()).request.images


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a.png", {"path": "a.png"}),
        ("a.png@8", {"path": "a.png", "frame_index": "8"}),
        ("a.png:0.8", {"path": "a.png", "strength": 0.8}),
        ("a.png@8:0.8", {"path": "a.png", "frame_index": "8", "strength": 0.8}),
        # Two numbers are the stage strengths alone; the common strength stays
        # at its default, exactly as on --adapter path.safetensors:S1:S2.
        ("a.png:0.9:0.5", {"path": "a.png", "stage1_strength": 0.9, "stage2_strength": 0.5}),
        (
            "a.png@16:0.9:0.5",
            {"path": "a.png", "frame_index": "16", "stage1_strength": 0.9, "stage2_strength": 0.5},
        ),
        # An ordinary path keeps every character it was given.
        ("photo@2x.png", {"path": "photo@2x.png"}),
        ("shots/take:2.png", {"path": "shots/take:2.png"}),
        # Range admission belongs to the field, so the shorthand hands 2 along.
        ("a.png@8:2", {"path": "a.png", "frame_index": "8", "strength": 2.0}),
    ],
)
def test_shorthand_decodes_a_frame_and_a_strength_without_touching_the_filesystem(text, expected):
    assert _SHORTHAND(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("@8", "requires a path"),
        (":0.5", "requires a path"),
        ("a.png:1:2:3", "one or two numeric suffix values"),
        ("a.png@8:0.9:0.7:0.5", "one or two numeric suffix values"),
    ],
)
def test_shorthand_refuses_a_missing_path_or_a_third_number(text, message):
    with pytest.raises(ValueError, match=message):
        _SHORTHAND(text)


def test_shorthand_does_not_depend_on_file_existence(tmp_path: Path):
    literal = tmp_path / "plate.png@8"
    literal.touch()
    assert _SHORTHAND(str(literal)) == {"path": str(tmp_path / "plate.png"), "frame_index": "8"}


def test_repeated_items_build_ordered_records_and_member_flags_extend_the_last_one():
    ordered = _images("--image", "first.png", "--image", "last.png@16:0.8", "--frames", "17")
    assert [(image.path, image.frame_index, image.strength) for image in ordered] == [
        (Path("first.png"), "first", 1.0),
        (Path("last.png"), "16", 0.8),
    ]
    extended = _images(
        "--image",
        "first.png",
        "--image",
        "last.png",
        "--image-frame-index",
        "16",
        "--image-strength",
        "0.5",
        "--frames",
        "17",
    )
    assert [(image.frame_index, image.strength) for image in extended] == [
        ("first", 1.0),
        ("16", 0.5),
    ]


def test_two_numbers_target_the_stage_members_and_leave_the_common_strength():
    ordered = _images(
        "--image", "first.png:1.0:0.6", "--image", "last.png@16:0.9:0.5", "--frames", "17"
    )
    assert [
        (image.strength, image.stage1_strength, image.stage2_strength) for image in ordered
    ] == [(1.0, 1.0, 0.6), (1.0, 0.9, 0.5)]
    assert [(image.strength_for_stage(1), image.strength_for_stage(2)) for image in ordered] == [
        (1.0, 0.6),
        (0.9, 0.5),
    ]


def test_the_stage_member_flags_extend_the_preceding_item():
    extended = _images(
        "--image",
        "first.png",
        "--image",
        "last.png@16",
        "--image-stage1-strength",
        "0.9",
        "--image-stage2-strength",
        "0.5",
        "--frames",
        "17",
    )
    assert [(image.stage1_strength, image.stage2_strength) for image in extended] == [
        (None, None),
        (0.9, 0.5),
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--image-frame-index", "8"], "requires a preceding --image item"),
        (["--image-strength", "0.5"], "requires a preceding --image item"),
        (["--image-stage1-strength", "0.5"], "requires a preceding --image item"),
        (["--image-stage2-strength", "0.5"], "requires a preceding --image item"),
        (["--image", "a.png", "--image-strength", "0.5", "--image-strength", "0.6"], "duplicate"),
        (
            ["--image", "a.png:0.9:0.5", "--image-stage2-strength", "0.6"],
            "duplicate",
        ),
    ],
)
def test_a_member_flag_without_its_item_is_refused(arguments, message, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argument", "member"),
    [
        ("a.png:2", "strength"),
        ("a.png:0.9:2", "stage2_strength"),
        ("a.png:2:0.5", "stage1_strength"),
    ],
)
def test_a_strength_outside_its_range_reaches_the_ordinary_member_error(argument, member, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--image", argument])
    assert f"generate.images[0].{member}: must be between 0 and 1" in capsys.readouterr().err


def test_two_conditions_on_one_frame_are_refused_by_name():
    with pytest.raises(ConfigError, match="image frame_index 0 is conditioned more than once"):
        _images("--image", "a.png", "--image", "b.png@0")


def test_the_collection_is_written_and_patched_on_every_structured_surface(tmp_path: Path):
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\n'
        'images = [{path = "first.png", stage2_strength = 0.6}, '
        '{path = "last.png", frame_index = 16}]\n',
        encoding="utf-8",
    )
    from_toml = _images("--config", str(config), "--frames", "17")
    assert [image.path.name for image in from_toml] == ["first.png", "last.png"]
    assert [image.stage2_strength for image in from_toml] == [0.6, None]

    patched = _images(
        "--config", str(config), "--frames", "17", "--set", "generate.images[1].strength=0.7"
    )
    assert [image.strength for image in patched] == [1.0, 0.7]
    staged = _images(
        "--config", str(config), "--frames", "17", "--set", "generate.images[0].stage2_strength=0.7"
    )
    assert [image.strength_for_stage(2) for image in staged] == [0.7, 1.0]
    assert _images("--config", str(config), "--set", "generate.images=[]") == ()


def test_the_environment_carries_the_same_array_of_records(monkeypatch):
    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "KINO_LTX2_IMAGES",
        '[{path="first.png"}, {path="last.png", frame_index=16, stage1_strength=0.9}]',
    )
    ordered = _images("--frames", "17")
    assert [image.frame_index for image in ordered] == ["first", "16"]
    assert [image.stage1_strength for image in ordered] == [None, 0.9]


def test_print_config_renders_the_ordered_array():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(
            [
                "--model",
                "ltx2",
                "--print-config",
                "--prompt",
                "a clip",
                "--frames",
                "17",
                "--image",
                "first.png",
                "--image",
                "last.png@16:0.9:0.5",
            ]
        )
    assert code == 0, err.getvalue()
    printed = tomllib.loads(out.getvalue())
    assert printed["generate"]["images"] == [
        {"path": "first.png"},
        {"path": "last.png", "frame_index": "16", "stage1_strength": 0.9, "stage2_strength": 0.5},
    ]


def test_the_starter_template_shows_a_first_and_last_frame_example():
    template = config_template("ltx2")
    assert "# Condition frames on ordered images." in template
    assert (
        '#   { path = "last.png", frame_index = "last", strength = 0.8, stage2_strength = 0.5 },'
        in template
    )
    assert "# Environment: KINO_LTX2_IMAGES." in template
    assert "[generate.image]" not in template


def test_config_inspect_lists_the_collection_and_every_member():
    schema = config_registry().model("ltx2")
    entry = configuration_inventory(schema, field="generate.images")["fields"][0]
    assert entry["item_schema"] is not None
    members = {member["name"]: member for member in entry["item_schema"]["members"]}
    assert list(members) == [
        "path",
        "frame_index",
        "strength",
        "stage1_strength",
        "stage2_strength",
    ]
    assert members["path"]["cli_flags"] == []
    assert members["frame_index"]["cli_flags"] == ["--image-frame-index"]
    assert members["strength"]["cli_flags"] == ["--image-strength"]
    assert members["stage1_strength"]["cli_flags"] == ["--image-stage1-strength"]
    assert members["stage2_strength"]["cli_flags"] == ["--image-stage2-strength"]
    assert entry["environment"] == ["KINO_LTX2_IMAGES"]


def _first_frame_mode(*arguments: str) -> str:
    options = build_parser().parse_args([*arguments, "--print-config"])
    return assemble(options, base_settings=Settings()).request.first_frame_mode


def test_the_frame_zero_mode_reaches_the_request_from_every_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert _first_frame_mode("--frames", "17") == "anchor"
    assert _first_frame_mode("--first-frame-mode", "guide", "--frames", "17") == "guide"

    config = tmp_path / "run.toml"
    config.write_text('[generate]\nprompt = "test"\nfirst_frame_mode = "guide"\n', encoding="utf-8")
    assert _first_frame_mode("--config", str(config), "--frames", "17") == "guide"
    assert (
        _first_frame_mode("--frames", "17", "--set", "generate.first_frame_mode=guide") == "guide"
    )

    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KINO_LTX2_FIRST_FRAME_MODE", "guide")
    assert _first_frame_mode("--frames", "17") == "guide"


def test_the_frame_zero_mode_is_admitted_without_a_frame_zero_condition():
    """Guide selects how frame 0 would be applied; it never refuses a run."""
    options = build_parser().parse_args(
        [
            "--first-frame-mode",
            "guide",
            "--image",
            "last.png@16",
            "--frames",
            "17",
            "--print-config",
        ]
    )
    request = assemble(options, base_settings=Settings()).request
    assert request.first_frame_mode == "guide"
    assert [image.frame_index for image in request.images] == ["16"]


def test_an_unknown_frame_zero_mode_is_refused_by_name(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--first-frame-mode", "hold"])
    assert "--first-frame-mode" in capsys.readouterr().err


def test_the_help_states_the_effect_of_each_frame_zero_mode():
    help_text = " ".join(build_parser().format_help().split())
    assert "anchor: Replace the first frame with the image and hold it fixed" in help_text
    assert "guide: Guide a denoised first frame toward the image" in help_text


def test_config_inspect_lists_the_frame_zero_mode_field():
    schema = config_registry().model("ltx2")
    entry = configuration_inventory(schema, field="generate.first_frame_mode")["fields"][0]
    assert entry["owner"].endswith("LTX2Request.first_frame_mode")
    assert entry["constraints"] == '"anchor" or "guide"'
    assert entry["local_constraints"][0]["choices"] == ["anchor", "guide"]
    assert entry["cli"]["flags"] == ["--first-frame-mode"]
    assert entry["environment"] == ["KINO_LTX2_FIRST_FRAME_MODE"]
    assert entry["default_toml"] == '"anchor"'


def test_a_run_record_that_carries_the_former_image_table_cannot_be_restarted(tmp_path: Path):
    run, _paths = _source_run(tmp_path)
    payload = json.loads(run.read_text())
    payload["configuration"]["values"]["generate"]["image"] = {
        "path": "frame.png",
        "frame_index": 0,
        "strength": 1.0,
    }
    run.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[generate\]\.image: unknown field 'image'"):
        _assemble(run)


@pytest.mark.parametrize(
    ("suffix", "stored"),
    [("first", "first"), ("last", "last"), ("LAST", "LAST"), ("0", "0"), ("120", "120")],
)
def test_the_shorthand_admits_a_named_position_in_the_case_it_was_typed(suffix, stored):
    assert _SHORTHAND(f"a.png@{suffix}") == {"path": "a.png", "frame_index": stored}


def test_an_unrecognized_suffix_still_belongs_to_the_path():
    """Only a position token is peeled, so photo@2x.png keeps every character."""
    assert _SHORTHAND("photo@2x.png") == {"path": "photo@2x.png"}
    assert _SHORTHAND("plate@lst") == {"path": "plate@lst"}


@pytest.mark.parametrize("position", ["first", "last", "0", "120"])
def test_every_position_reaches_the_request_from_every_surface(
    position: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert [image.frame_index for image in _images("--image", f"a.png@{position}")] == [position]

    config = tmp_path / "run.toml"
    config.write_text(
        f'[generate]\nprompt = "test"\nimages = [{{path = "a.png", frame_index = "{position}"}}]\n',
        encoding="utf-8",
    )
    assert [image.frame_index for image in _images("--config", str(config))] == [position]

    assert [
        image.frame_index
        for image in _images(
            "--image", "a.png", "--set", f"generate.images[0].frame_index={position}"
        )
    ] == [position]

    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KINO_LTX2_IMAGES", f'[{{path="a.png", frame_index="{position}"}}]')
    assert [image.frame_index for image in _images()] == [position]


def test_a_bare_frame_number_is_the_string_the_field_stores(tmp_path: Path):
    """TOML, --set and the API deliver a number; the field keeps its decimal string."""
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nimages = [{path = "a.png", frame_index = 120}]\n',
        encoding="utf-8",
    )
    assert [image.frame_index for image in _images("--config", str(config))] == ["120"]
    assert [
        image.frame_index
        for image in _images("--image", "a.png", "--set", "generate.images[0].frame_index=120")
    ] == ["120"]
    assert LTX2ImageConditioningConfig(Path("a.png"), frame_index=120).frame_index == "120"
    assert LTX2ImageConditioningConfig(Path("a.png"), frame_index=120.0).frame_index == "120"


def test_a_named_position_and_its_integer_build_the_same_condition_source():
    """first and 0, and last and 16 on 17 frames, are the same absolute frame."""
    named = LTX2Request(
        prompt="test",
        frames=17,
        images=(
            LTX2ImageConditioningConfig(Path("first.png"), frame_index="first"),
            LTX2ImageConditioningConfig(Path("last.png"), frame_index="last"),
        ),
    )
    numbered = LTX2Request(
        prompt="test",
        frames=17,
        images=(
            LTX2ImageConditioningConfig(Path("first.png"), frame_index=0),
            LTX2ImageConditioningConfig(Path("last.png"), frame_index=16),
        ),
    )
    assert condition_sources(named, frames=17) == condition_sources(numbered, frames=17)
    assert [source.frame_index for source in condition_sources(named, frames=17)] == [0, 16]
    assert [source.family for source in condition_sources(named, frames=17)] == [
        "image",
        "keyframe",
    ]
    assert requested_condition_families(named) == requested_condition_families(numbered)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ("--image", "a.png", "--set", "generate.images[0].frame_index=middle"),
            "frame position 'middle' must be 'first', 'last', or a non-negative integer",
        ),
        (
            ("--image", "a.png@40", "--frames", "17"),
            "image frame_index 40 is outside 17 frames",
        ),
        (
            ("--image", "a.png@last", "--image", "b.png@16", "--frames", "17"),
            "image frame_index 16 is conditioned more than once: 'last' and '16' "
            "both name frame 16",
        ),
        (
            ("--image", "a.png@last", "--image", "b.png@last", "--frames", "17"),
            "image frame_index 16 is conditioned more than once",
        ),
        (
            ("--auto-duration", "--image", "a.png@last", "--image", "b.png@last"),
            "image frame_index 'last' is conditioned more than once",
        ),
        (
            ("--auto-duration", "--image", "a.png@first", "--image", "b.png@0"),
            "image frame_index 0 is conditioned more than once: 'first' and '0' both name frame 0",
        ),
    ],
)
def test_a_position_the_grammar_or_the_timeline_refuses_names_the_problem(arguments, message):
    with pytest.raises(ConfigError) as refusal:
        _images(*arguments)
    assert message in str(refusal.value)


def test_an_unknown_name_stays_in_the_path_and_is_refused_as_a_missing_file(tmp_path: Path):
    """@lst is not a position token, so the refusal is the file it actually names."""
    request = LTX2Request(
        prompt="test",
        frames=17,
        images=(LTX2ImageConditioningConfig(tmp_path / "plate.png@lst"),),
    )
    with pytest.raises(ValueError, match=r"image does not exist: .*plate\.png@lst"):
        request.validate_for_generation()


def test_the_frame_position_constraint_and_config_inspect_show_the_grammar():
    described = FramePositionConstraint().describe()
    assert described["kind"] == "frame_position"
    assert described["named_positions"] == ["first", "last"]
    assert described["numeric_minimum"] == 0

    schema = config_registry().model("ltx2")
    entry = configuration_inventory(schema, field="generate.images")["fields"][0]
    member = next(item for item in entry["item_schema"]["members"] if item["name"] == "frame_index")
    assert member["constraints"] == (
        '"first", "last", or a non-negative integer, bare or quoted '
        '(in TOML, frame_index = 120 or "120"; saved as the string)'
    )
    assert member["local_constraints"][0]["named_positions"] == ["first", "last"]
    assert member["local_constraints"][0]["numeric_minimum"] == 0


def test_a_run_saves_the_position_spelling_the_user_wrote():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(
            [
                "--model",
                "ltx2",
                "--print-config",
                "--prompt",
                "a clip",
                "--frames",
                "17",
                "--image",
                "first.png@first",
                "--image",
                "last.png@last",
            ]
        )
    assert code == 0, err.getvalue()
    printed = tomllib.loads(out.getvalue())
    assert printed["generate"]["images"] == [
        {"path": "first.png", "frame_index": "first"},
        {"path": "last.png", "frame_index": "last"},
    ]


def test_the_grammar_resolves_names_and_integers_against_one_frame_count():
    assert parse_frame_position(" LAST ") == "last"
    assert parse_frame_position("016") == 16
    assert [is_first_position(spelling) for spelling in ("first", "0", "00", "last", "1")] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert resolve_frame_position("first", 17) == 0
    assert resolve_frame_position("last", 17) == 16
    assert resolve_frame_position("last", 1) == 0
    assert resolve_frame_positions(("first", "last", "8"), 17) == (0, 16, 8)
    with pytest.raises(ValueError, match="resolves against at least one frame"):
        resolve_frame_position("last", 0)
    with pytest.raises(ValueError, match="image frame_index 17 is outside 17 frames"):
        resolve_frame_position("17", 17)
    with pytest.raises(ValueError, match="outside predicted 17 frames"):
        resolve_frame_position("17", 17, predicted=True)
