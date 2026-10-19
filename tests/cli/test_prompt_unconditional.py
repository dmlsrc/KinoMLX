"""The explicit empty-user-prompt contract is shared by image and video models."""

import importlib
from dataclasses import replace

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_rules import validate_record
from kinomlx.models.flux2.types import Flux2Request
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.models.ltx2.types import LTX2Request


@pytest.fixture(params=["ltx2", "krea2", "flux2"])
def model_cli(request):
    model = request.param
    arguments = importlib.import_module(f"kinomlx.models.{model}.arguments")
    config = importlib.import_module(f"kinomlx.models.{model}.config")

    def assemble(*args):
        invocation = config.assemble(arguments.build_parser().parse_args(list(args)))
        value = invocation.generate if model == "flux2" else invocation.request
        return invocation, value

    return model, assemble


def test_empty_prompt_selects_generation_and_roundtrips(model_cli, tmp_path):
    model, assemble = model_cli
    invocation, request = assemble("--prompt-unconditional")
    assert request.prompt_unconditional is True
    if model == "flux2":
        assert request.prompts() == ("",)
        assert not invocation.operation.dry_run
    elif model == "krea2":
        assert request.prompts == ("",)
    else:
        request.validate_for_generation()
        assert request.prompt == ""
    path = tmp_path / "config.toml"
    path.write_text(f'model = "{model}"\n[generate]\nprompt_unconditional = true\n')
    _, loaded = assemble("--config", str(path))
    assert loaded.prompt_unconditional
    _, overridden = assemble(
        "--config", str(path), "--no-prompt-unconditional", "--prompt", "a landscape"
    )
    assert not overridden.prompt_unconditional


def test_prompt_conflicts_respect_effective_precedence(model_cli, tmp_path):
    model, assemble = model_cli
    path = tmp_path / "prompt.txt"
    path.write_text("a landscape")
    for args in (("--prompt", "a landscape"), ("--prompt-file", str(path))):
        with pytest.raises((ConfigError, ValueError), match="prompt_unconditional"):
            assemble("--prompt-unconditional", *args)
    config = tmp_path / "config.toml"
    config.write_text(f'model = "{model}"\n[generate]\nprompt = "a landscape"\n')
    with pytest.raises((ConfigError, ValueError), match="prompt_unconditional"):
        assemble("--config", str(config), "--prompt-unconditional")


@pytest.mark.parametrize("factory", [Flux2Request, Krea2Request, LTX2Request])
def test_public_requests_preserve_intent_through_replace(factory):
    request = factory(prompt_unconditional=True)
    assert replace(request, seed=123).prompt_unconditional
    request = factory(prompt_unconditional=True, prompt="a landscape")
    with pytest.raises(ValueError, match="prompt_unconditional"):
        validate_record(request, location="[generate]")


def test_image_count_and_negative_prompt_stay_independent():
    flux = Flux2Request(prompt_unconditional=True, images_per_prompt=3, negative_prompt="blurry")
    flux.validate("base")
    assert flux.prompts() == ("",)
    assert flux.negatives() == ("blurry",)
    krea = Krea2Request(prompt_unconditional=True, images_per_prompt=3, system_prompt="")
    assert krea.expanded_prompts() == ("", "", "")
    assert krea.system_prompt == ""
