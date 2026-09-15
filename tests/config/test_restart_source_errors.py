"""Every failure of a restart source reaches the user as one located error.

Opening a saved run touches the filesystem, JSON, a persisted configuration and
the recorded artifacts. Each model converts that whole step at one place, so a
damaged or unusable source is a configuration error naming ``[restart]`` with
exit 2, never a traceback.
"""

import json
import logging
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from tests.config.saved_records import saved_run_configuration

MODELS = ["krea2", "flux2", "ltx2"]
# Klein resolves its final-latent artifact after preparing resources, so a
# deleted Klein artifact is a model failure rather than a configuration one.
ARTIFACT_CHECKING_MODELS = ["krea2", "ltx2"]


def _generate_values(model: str) -> dict[str, object]:
    generate: dict[str, object] = {"prompt": "fixture prompt", "width": 64, "height": 64}
    if model == "ltx2":
        generate["frames"] = 9
    return {"model": model, "generate": generate, "output": {}}


def _generation(model: str) -> dict[str, object]:
    if model == "krea2":
        return {"variant": "turbo", "prompts": ["fixture prompt"]}
    if model == "flux2":
        return {
            "images": [{"seed": 1, "prompt": "fixture prompt", "negative_prompt": ""}],
            "resources": {"variant": "klein", "components": []},
        }
    return {"model_generation": "2.3"}


def _write_run(
    directory: Path,
    model: str,
    *,
    name: str,
    status: str,
    outputs: dict[str, object],
) -> Path:
    payload: dict[str, object] = {
        "model": model,
        "status": status,
        **saved_run_configuration(model, _generate_values(model)),
        "outputs": outputs,
        "generation": _generation(model),
    }
    run = directory / f"{name}_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


def _restart(model: str, run: Path, caplog) -> int:
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        return main(["--model", model, "--restart", str(run), "--print-config"])


def _assert_located_config_error(exit_code: int, caplog) -> None:
    assert exit_code == 2
    assert "config error: [restart]:" in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("model", MODELS)
def test_a_failed_source_run_is_a_located_configuration_error(model, tmp_path, caplog) -> None:
    # A run that failed published none of the stations a restart consumes.
    run = _write_run(tmp_path, model, name=f"{model}_failed", status="failed", outputs={})

    _assert_located_config_error(_restart(model, run, caplog), caplog)


@pytest.mark.parametrize("model", ARTIFACT_CHECKING_MODELS)
def test_a_missing_recorded_artifact_is_a_located_configuration_error(
    model, tmp_path, caplog
) -> None:
    absent = tmp_path / f"{model}_deleted_final_latents.safetensors"
    run = _write_run(
        tmp_path,
        model,
        name=f"{model}_missing",
        status="completed",
        outputs={"final_latents": str(absent)},
    )

    _assert_located_config_error(_restart(model, run, caplog), caplog)
    assert "does not exist" in caplog.text


@pytest.mark.parametrize("model", MODELS)
def test_an_unreadable_restart_source_is_a_located_configuration_error(
    model, tmp_path, caplog
) -> None:
    # A directory cannot be read as a run log; the operating system error is a
    # problem with the [restart] table, not an internal fault.
    folder = tmp_path / f"{model}_folder_run.json"
    folder.mkdir()

    _assert_located_config_error(_restart(model, folder, caplog), caplog)


def test_a_klein_record_without_the_selected_image_row_is_located(tmp_path, caplog) -> None:
    latents = tmp_path / "klein_final_latents.safetensors"
    latents.write_bytes(b"latents")
    payload: dict[str, object] = {
        "model": "flux2",
        "status": "completed",
        **saved_run_configuration("flux2", _generate_values("flux2")),
        "outputs": {"final_latents": str(latents)},
        # The record claims two images but lists one row, so the selection has
        # nothing to narrow the inherited prompt batch with.
        "planned_image_count": 2,
        "generation": {
            "images": [{"seed": 1, "prompt": "fixture prompt", "negative_prompt": ""}],
            "resources": {"variant": "klein", "components": []},
        },
    }
    run = tmp_path / "klein_partial_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(
            ["--model", "flux2", "--restart", str(run), "--restart-index", "2", "--print-config"]
        )

    _assert_located_config_error(exit_code, caplog)
    assert "records no image 2" in caplog.text
