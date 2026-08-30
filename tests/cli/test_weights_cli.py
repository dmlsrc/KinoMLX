"""The generic weights host dispatches model-specific extensions."""

from __future__ import annotations

import json

import numpy as np

from kinomlx.cli.main import main
from kinomlx.weights.cli import run_weights_command
from tests.models.gmnet.test_convert import _install_fake_torch, _write_zip_checkpoint


def test_weights_help_lists_generic_and_gmnet_converters(capsys) -> None:
    assert main(["weights", "--help"]) == 0
    output = capsys.readouterr().out
    assert "convert INPUT" in output
    assert "convert gmnet INPUT" in output


def test_generic_convert_dispatch(monkeypatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(
        "kinomlx.weights.cli.run_generic_convert",
        lambda argv: received.append(argv) or 17,
    )

    assert run_weights_command(["convert", "source.pth", "--force"]) == 17
    assert received == [["source.pth", "--force"]]


def test_gmnet_convert_dispatch(monkeypatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(
        "kinomlx.models.gmnet.converter_cli.run_gmnet_convert_command",
        lambda argv: received.append(argv) or 19,
    )

    assert run_weights_command(["convert", "gmnet", "G_realworld.pth"]) == 19
    assert received == [["G_realworld.pth"]]


def test_root_main_dispatches_weights_before_model_routing(monkeypatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(
        "kinomlx.weights.cli.run_weights_command",
        lambda argv: received.append(argv) or 23,
    )

    assert main(["weights", "convert", "model.pth"]) == 23
    assert received == [["convert", "model.pth"]]


def test_generic_converter_command_emits_machine_receipt(tmp_path, monkeypatch, capsys) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "plain.pth"
    output = tmp_path / "plain.safetensors"
    _write_zip_checkpoint(source, {"module.weight": np.ones((2, 3), dtype=np.float32)})

    assert (
        run_weights_command(
            [
                "convert",
                str(source),
                "--output",
                str(output),
                "--json",
                "--quiet",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["converter"] == "generic"
    assert payload["tensor_count"] == 1
    assert payload["output"] == str(output)
    assert payload["source"] == str(source)
    assert payload["flagged_globals"] == []
