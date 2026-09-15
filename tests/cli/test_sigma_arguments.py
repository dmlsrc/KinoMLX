"""The sigma spellings every model offers, and the message a bad one produces."""

import contextlib
import importlib
import io

import pytest


def _parse(model, *args):
    arguments = importlib.import_module(f"kinomlx.models.{model}.arguments")
    return arguments.build_parser().parse_args(["--prompt", "A landscape.", *args])


def _usage_error(model, *args) -> str:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        _parse(model, *args)
    return stderr.getvalue()


@pytest.mark.parametrize("model", ["krea2", "flux2"])
def test_image_models_take_one_schedule(model):
    assert _parse(model, "--sigmas", "1,0.75,0.5,0").sigmas == (1.0, 0.75, 0.5, 0.0)


@pytest.mark.parametrize("flag", ["--sigmas-stage-1", "--stage1-sigmas"])
def test_ltx2_takes_one_schedule_per_stage(flag):
    parsed = _parse("ltx2", flag, "1,0.5,0", "--sigmas-stage-2", "0.4,0")
    assert parsed.stage_1_sigmas == (1.0, 0.5, 0.0)
    assert parsed.stage_2_sigmas == (0.4, 0.0)


@pytest.mark.parametrize("model", ["krea2", "flux2", "ltx2"])
def test_omitting_the_flag_leaves_the_model_default(model):
    flag = "stage_1_sigmas" if model == "ltx2" else "sigmas"
    assert getattr(_parse(model), flag) is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0.5,0.9,0", "decrease strictly"),
        ("1.5,0", "between 0 and 1"),
        ("1", "at least two values"),
        ("1,low,0", "all be numbers"),
        ("1,,0", "comma-separated"),
    ],
)
def test_a_malformed_schedule_explains_itself_at_the_command_line(value, message):
    # argparse reports a bare type name unless the parser's own message surfaces.
    assert message in _usage_error("krea2", "--sigmas", value)


def test_a_malformed_stage_schedule_reports_the_reason_not_the_type_name():
    rendered = _usage_error("ltx2", "--sigmas-stage-2", "0.5,0.9,0")
    assert "--sigmas-stage-2" in rendered
    assert "decrease strictly" in rendered
    assert "invalid stage 2 sigmas value" not in rendered
