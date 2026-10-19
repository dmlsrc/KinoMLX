"""Resource-only dry run: one declaration, one refusal, one restart lock.

A dry run resolves and validates every source a generation would need, prints
a JSON inventory and exits without loading tensors. The cases here pin the
parts that must read the same on every family. Each printed report is
model-specific and is checked beside that family's own resource tests.
"""

import json
import logging
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from kinomlx.config.inspection import configuration_inventory
from kinomlx.model_catalog import model_choices
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.arguments import build_parser as flux_parser
from kinomlx.models.flux2.config import assemble as flux_assemble
from kinomlx.models.krea2.arguments import build_parser as krea_parser
from kinomlx.models.krea2.config import assemble as krea_assemble
from kinomlx.models.ltx2.arguments import build_parser as ltx_parser
from kinomlx.models.ltx2.config import assemble as ltx_assemble
from tests.config.saved_records import saved_run_configuration

_GENERATION_INPUT = {
    "flux2": ("--prompt", "a still"),
    "gmnet": ("--image", "source.png"),
    "krea2": ("--prompt", "a still"),
    "ltx2": ("--prompt", "a clip"),
}
_RESTART_MODELS = ("flux2", "krea2", "ltx2")
_RESTART_ASSEMBLY = {
    "flux2": (flux_parser, flux_assemble),
    "krea2": (krea_parser, krea_assemble),
    "ltx2": (ltx_parser, ltx_assemble),
}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient product variables so each declared layer stays explicit."""
    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)


def _generate_values(model: str) -> dict[str, object]:
    generate: dict[str, object] = {"prompt": "fixture prompt", "width": 64, "height": 64}
    if model == "ltx2":
        generate["frames"] = 9
    return {"model": model, "generate": generate, "output": {}, "operation": {"dry_run": True}}


def _generation(model: str) -> dict[str, object]:
    if model == "krea2":
        return {"variant": "turbo", "prompts": ["fixture prompt"]}
    if model == "flux2":
        return {
            "images": [{"seed": 1, "prompt": "fixture prompt", "negative_prompt": ""}],
            "resources": {"variant": "klein", "components": []},
        }
    return {"model_generation": "2.3"}


def _dry_run_source(directory: Path, model: str) -> Path:
    """Write a completed run whose saved configuration recorded ``dry_run``.

    Nothing produces such a run today, because a dry run never generates.
    Writing one anyway is how the stage lock is tested rather than assumed:
    the restart must force the flag false instead of inheriting it.
    """
    latents = directory / f"{model}_final_latents.safetensors"
    latents.write_bytes(b"latents")
    text = directory / f"{model}_text_conditioning.safetensors"
    text.write_bytes(b"text")
    payload: dict[str, object] = {
        "model": model,
        "status": "completed",
        **saved_run_configuration(model, _generate_values(model)),
        "outputs": {"final_latents": str(latents), "text_conditioning": str(text)},
        "generation": _generation(model),
    }
    run = directory / f"{model}_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


def test_every_family_declares_the_same_resource_only_operation() -> None:
    """A user who learns the flag on one model finds the same one on the rest."""
    registry = config_registry()
    assert set(model_choices()) == set(_GENERATION_INPUT)
    for model in _GENERATION_INPUT:
        schema = registry.model(model)
        field = schema.field(("operation", "dry_run"))
        assert field.persistence == "intent"
        (entry,) = configuration_inventory(schema, field="operation.dry_run")["fields"]
        assert entry["default_toml"] == "false"
        assert entry["cli"] == {
            "flags": ["--dry-run"],
            "destination": "dry_run",
            "item_member": None,
            "action": "store_true",
            "codec": "inferred",
        }
        assert entry["environment"] == [f"KINO_{model.upper()}_DRY_RUN"]
        # A restart executes a generation stage, so no stage may resume as a
        # dry run; GMNet has no stage to lock because it has no restart.
        assert {policy.stage: policy.mode for policy in field.stage_policies} == dict.fromkeys(
            schema.restart_stages, "locked"
        )


@pytest.mark.parametrize("model", tuple(_GENERATION_INPUT))
def test_inspecting_resources_is_the_flag_and_never_an_omission(model: str, caplog) -> None:
    """No family inspects resources because generation input was left out.

    Krea did, which made the flag and the omission two ways to reach one
    behavior, and made Krea the one family whose bare invocation did not say
    what it needed.
    """
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(["--model", model])

    assert exit_code == 2
    assert "config error:" in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("model", tuple(_GENERATION_INPUT))
def test_dry_run_refuses_the_generation_it_would_not_perform(model: str, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(["--model", model, "--dry-run", *_GENERATION_INPUT[model]])

    assert exit_code == 2
    assert "config error: Resource-only operations cannot be combined" in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("model", _RESTART_MODELS)
def test_dry_run_refuses_a_saved_stage_restart(model: str, tmp_path: Path, caplog) -> None:
    run = _dry_run_source(tmp_path, model)

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(["--model", model, "--restart", str(run), "--dry-run"])

    assert exit_code == 2
    assert "config error: Resource-only operations cannot be combined" in caplog.text


@pytest.mark.parametrize("model", _RESTART_MODELS)
def test_a_saved_stage_restart_forces_dry_run_false(model: str, tmp_path: Path) -> None:
    run = _dry_run_source(tmp_path, model)
    build_parser, assemble = _RESTART_ASSEMBLY[model]

    invocation = assemble(build_parser().parse_args(["--restart", str(run)]))

    # The inherited layer is forced false, so a recorded flag cannot make a
    # restart skip the generation stage it exists to repeat.
    assert invocation.operation.dry_run is False
    assert invocation.resolved_config["operation"] == {"dry_run": False}
