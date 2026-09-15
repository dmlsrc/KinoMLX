"""Image models share output format admission and exact replacement policy."""

import importlib
from pathlib import Path

import pytest

from kinomlx.config_records import replace_config
from kinomlx.config_rules import validate_record
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.models.flux2.output import plan_output
from kinomlx.models.gmnet.output import GMNetOutputError, plan_gmnet_output
from kinomlx.models.gmnet.types import GMNetOutputConfig, GMNetRequest
from kinomlx.models.krea2.output import Krea2StillOutputPlan, plan_krea2_output
from kinomlx.models.krea2.types import Krea2Request


@pytest.mark.parametrize(
    ("model", "suffix"), [("flux2", "png"), ("krea2", "tiff"), ("gmnet", "exr")]
)
def test_all_image_model_assemblies_require_an_exact_replacement_target(model, suffix):
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    assemble = importlib.import_module(f"kinomlx.models.{model}.config").assemble
    options = parser.parse_args(["--force", "--print-config"])
    with pytest.raises(ValueError, match="force requires an exact output path"):
        assemble(options)
    options = parser.parse_args(["--force", "--output", f"result.{suffix}", "--print-config"])
    invocation = assemble(options)
    assert invocation.output.force is True
    assert invocation.output.path == Path(f"result.{suffix}")


@pytest.mark.parametrize("record", [ImageOutputConfig, GMNetOutputConfig])
def test_explicit_target_resolves_provisional_replacement_conflict(record):
    config = record(force=True)
    with pytest.raises(ValueError, match=r"\[output\]: force requires an exact output path"):
        validate_record(config, location="[output]")
    suffix = "exr" if record is GMNetOutputConfig else "png"
    validate_record(replace_config(config, path=f"result.{suffix}"), location="[output]")


@pytest.mark.parametrize("suffix", [".png", ".tif", ".tiff", ".PNG", ".TIFF"])
def test_still_plan_and_configuration_accept_the_same_suffixes(suffix):
    path = Path(f"missing{suffix}")
    assert ImageOutputConfig(path=path).path == path
    assert Krea2StillOutputPlan(path).path == path


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".webp", ".mp4", ""])
def test_still_plan_and_configuration_reject_the_same_suffixes(suffix):
    path = Path(f"missing{suffix}")
    expected = "must end in .png or .tif or .tiff"
    with pytest.raises(ValueError, match="must end in") as configured:
        ImageOutputConfig(path=path)
    with pytest.raises(ValueError, match="must end in") as planned:
        Krea2StillOutputPlan(path)
    # One constraint object serves both, so the refusal reads the same after the
    # configuration path adds the field it was written at.
    assert str(configured.value).endswith(expected)
    assert str(planned.value) == expected


def test_direct_output_planners_all_enforce_exact_replacement():
    config = ImageOutputConfig(force=True)
    with pytest.raises(ValueError, match=r"\[output\]: force requires an exact output path"):
        plan_output(config, variant="distilled", seed=0, image_count=1)
    with pytest.raises(ValueError, match=r"\[output\]: force requires an exact output path"):
        plan_krea2_output(Krea2Request(prompt="hello"), "turbo", config)
    with pytest.raises(GMNetOutputError, match="force requires an exact output path"):
        plan_gmnet_output(GMNetRequest(Path("source.png")), GMNetOutputConfig(force=True))


@pytest.mark.parametrize(
    ("record", "path"),
    [(ImageOutputConfig, "image.jpg"), (GMNetOutputConfig, "still.png")],
)
def test_a_direct_construction_names_the_table_not_the_class(record, path):
    with pytest.raises(ValueError, match=r"^\[output\]\.path: must end in") as error:
        record(path=path)
    assert record.__name__ not in str(error.value)
