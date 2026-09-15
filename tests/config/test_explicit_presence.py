"""Sparse config preserves supplied intent rather than inferring it from equality."""

import tomllib

import pytest

from kinomlx.cli.main import main
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
@pytest.mark.parametrize("source", ["cli", "file", "set", "env"])
def test_explicit_default_survives_sparse_output(tmp_path, monkeypatch, capsys, model, source):
    monkeypatch.delenv("KINO_VERBOSE", raising=False)
    arguments = ["--model", model, "--print-config", "--only-non-defaults"]
    if source == "cli":
        arguments += ["--no-verbose"]
    elif source == "set":
        arguments += ["--set", "settings.verbose=false"]
    elif source == "file":
        path = tmp_path / "source.toml"
        path.write_text("[settings]\nverbose=false\n")
        arguments += ["--config", str(path)]
    else:
        monkeypatch.setenv("KINO_VERBOSE", "false")
    assert main(arguments) == 0
    sparse = tomllib.loads(capsys.readouterr().out)
    assert sparse["settings"]["verbose"] is False


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_absent_default_is_still_omitted(monkeypatch, capsys, model):
    monkeypatch.delenv("KINO_VERBOSE", raising=False)
    assert main(["--model", model, "--print-config", "--only-non-defaults"]) == 0
    sparse = tomllib.loads(capsys.readouterr().out)
    assert "verbose" not in sparse.get("settings", {})


def test_source_decoding_transfers_presence_but_implication_does_not(tmp_path):
    schema = config_registry().model("flux2")
    path = tmp_path / "prompt.txt"
    path.write_text("literal prompt")
    ((values, explicit),) = schema.prepare_layers_with_presence(
        (
            (
                "test",
                {
                    "generate": {"prompt_file": str(path)},
                    "output": {"save_all_sidecars": True},
                },
            ),
        )
    )
    assert ("generate", "prompt") in explicit
    assert ("generate", "prompt_file") not in explicit
    assert values["output"]["save_run_log"] is True
    assert ("output", "save_run_log") not in explicit


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_explicit_implication_target_remains_in_sparse_output(capsys, model):
    assert (
        main(
            [
                "--model",
                model,
                "--print-config",
                "--only-non-defaults",
                "--save-all-sidecars",
                "--save-run-log",
            ]
        )
        == 0
    )
    sparse = tomllib.loads(capsys.readouterr().out)
    assert sparse["output"]["save_all_sidecars"] is True
    assert sparse["output"]["save_run_log"] is True
