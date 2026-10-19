"""The ``--model gmnet`` command: routing, precedence, and refusals."""

import json
import logging
import subprocess
import sys
import tomllib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinomlx.cli.main import main
from kinomlx.config import ConfigError
from kinomlx.models.gmnet.arguments import build_parser
from kinomlx.models.gmnet.config import assemble
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.settings import Settings

_BLOCKED_HELP_PROBE = r"""
import sys

BLOCKED = ("tokenizers", "transformers", "diffusers", "torch", "PIL")

def is_blocked(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if is_blocked(name):
            raise ImportError(f"{name} blocked: gmnet help must use KinoMLX dependencies")
        return None

sys.meta_path.insert(0, Blocker())

from kinomlx.cli.main import main

for arguments in (
    ["--model", "gmnet", "--help"],
    ["weights", "convert", "--help"],
    ["weights", "convert", "gmnet", "--help"],
):
    try:
        main(arguments)
    except SystemExit as exit_info:
        assert exit_info.code == 0
offenders = sorted(name for name in sys.modules if is_blocked(name))
assert not offenders, f"optional external runtimes imported by gmnet help: {offenders}"
print("model and converter help ok")
"""


def _assemble(argv: list[str], *, model_settings: GMNetSettings | None = None):
    return assemble(
        build_parser().parse_args(argv),
        base_settings=Settings(),
        base_model_settings=model_settings,
    )


def test_default_output_directory_is_stable_without_environment(monkeypatch) -> None:
    monkeypatch.delenv("KINO_GMNET_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("KINO_OUTPUT_DIR", raising=False)

    invocation = _assemble([], model_settings=GMNetSettings())

    assert invocation.output.directory == Path("outputs")


def test_gmnet_help_avoids_optional_external_runtimes() -> None:
    process = subprocess.run(
        [sys.executable, "-c", _BLOCKED_HELP_PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "model and converter help ok" in process.stdout


def test_root_help_is_compact_and_model_neutral(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "Native multimodal MLX inference" in output
    assert "ltx2" in output
    assert "gmnet" in output
    assert "config init" in output
    assert "--transformer-resident-blocks" not in output


@pytest.mark.parametrize("spelling", [["--model", "gmnet"], ["--model=gmnet"]])
def test_main_routes_the_gmnet_model(spelling) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*spelling, "--help"])
    assert exit_info.value.code == 0


def test_config_only_model_selection_routes_to_gmnet(tmp_path, capsys) -> None:
    config = tmp_path / "run.toml"
    config.write_text('model = "gmnet"\n[model_settings]\nvariant = "synthetic"\n')
    assert main(["--config", str(config), "--print-config"]) == 0
    output = capsys.readouterr().out
    assert 'model = "gmnet"' in output
    assert 'variant = "synthetic"' in output


def test_set_only_model_selection_routes_to_gmnet(capsys) -> None:
    assert main(["--set", "model=gmnet", "--print-config"]) == 0
    assert 'model = "gmnet"' in capsys.readouterr().out


def test_gmnet_save_config_and_sparse_env_values_share_the_registry(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("KINO_GMNET_VARIANT", "synthetic")
    monkeypatch.setenv("KINO_GMNET_OUTPUT_DIR", str(tmp_path / "renders"))
    saved = tmp_path / "gmnet.toml"
    arguments = ["--model", "gmnet", "--image", "input.png"]

    assert main([*arguments, "--print-config"]) == 0
    full = tomllib.loads(capsys.readouterr().out)
    assert (
        main(
            [
                *arguments,
                "--save-config",
                str(saved),
                "--only-non-defaults",
            ]
        )
        == 0
    )
    sparse = tomllib.loads(saved.read_text(encoding="utf-8"))
    assert sparse["model"] == "gmnet"
    assert sparse["expand"] == {"image": "input.png"}
    assert sparse["model_settings"] == {"variant": "synthetic"}
    assert sparse["output"] == {"directory": str(tmp_path / "renders")}

    capsys.readouterr()
    assert main(["--config", str(saved), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == full


def test_gmnet_save_all_expands_every_applicable_sidecar() -> None:
    invocation = _assemble(["--save-all-sidecars"])

    assert invocation.output.save_all_sidecars is True
    assert invocation.output.save_gain_map is True
    assert invocation.output.save_run_log is True
    assert invocation.output.save_console_log is True
    assert invocation.output.save_effective_config is True


def test_gmnet_save_all_writes_execution_sidecars_before_model_loading(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("KINO_GMNET_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("KINO_OUTPUT_DIR", raising=False)
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "expanded.exr"

    class FakeReporter:
        def memory_checkpoint(self, _name):
            pass

        def memory_to_dict(self):
            return {}

        def to_dict(self):
            return {"expand": {"elapsed_seconds": 0.1}}

    def fake_run(_invocation, _reporter, *, plan, reservation):
        assert reservation.active
        for path in plan.artifacts.paths():
            path.write_bytes(b"artifact")
        return {
            "action": "expand",
            "outputs": plan.artifacts.to_dict(),
            "generation": {"variant": "realworld"},
            "warnings": [],
        }

    monkeypatch.setattr(
        "kinomlx.models.gmnet.command.command_reporter",
        lambda _stack, _settings: FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.gmnet.command._run_expand", fake_run)

    assert (
        main(
            [
                "--model",
                "gmnet",
                "--image",
                str(source),
                "--output",
                str(output),
                "--save-all-sidecars",
                "--json",
            ]
        )
        == 0
    )

    effective = tmp_path / "expanded_config.toml"
    execution = tmp_path / "expanded_console.log"
    run_log = tmp_path / "expanded_run.json"
    gain_map = tmp_path / "expanded.gain_map.safetensors"
    assert output.is_file()
    assert gain_map.is_file()
    assert effective.is_file()
    assert execution.is_file()
    assert run_log.is_file()
    assert tomllib.loads(effective.read_text(encoding="utf-8"))["output"] == {
        "directory": "outputs",
        "path": str(output),
        "save_all_sidecars": True,
        "save_console_log": True,
        "save_effective_config": True,
        "save_gain_map": True,
        "save_run_log": True,
        "force": False,
    }
    record = json.loads(run_log.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert set(record["outputs"]) == {
        "effective_config",
        "execution_log",
        "exr",
        "gain_map",
        "run_log",
    }
    execution_text = execution.read_text(encoding="utf-8")
    assert "--save-all-sidecars" in execution_text
    assert "Run:" in execution_text
    assert "Model: GMNet Realworld" in execution_text
    assert "Planned outputs:" in execution_text
    assert "Outputs:" in execution_text
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["model"] == "gmnet"
    assert result["action"] == "expand"
    assert result["generation"] == {"variant": "realworld"}
    assert result["warnings"] == []
    assert result["sidecar_errors"] == []
    assert set(result["outputs"]) == {
        "effective_config",
        "execution_log",
        "exr",
        "gain_map",
        "run_log",
    }


def test_gmnet_failure_finalizes_sidecars_and_releases_reservations(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KINO_GMNET_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("KINO_OUTPUT_DIR", raising=False)
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "expanded.exr"

    class FakeReporter:
        def memory_checkpoint(self, _name):
            pass

        def memory_to_dict(self):
            return {"peak_usage": 1}

        def to_dict(self):
            return {"expand": {"elapsed_seconds": 0.1}}

    def fail(_invocation, _reporter, *, plan, reservation):
        assert reservation.active
        assert (tmp_path / "expanded_config.toml").is_file()
        assert (tmp_path / "expanded_console.log").is_file()
        assert (tmp_path / "expanded_run.json").is_file()
        raise RuntimeError("synthetic expansion failure")

    monkeypatch.setattr(
        "kinomlx.models.gmnet.command.command_reporter",
        lambda _stack, _settings: FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.gmnet.command._run_expand", fail)

    assert (
        main(
            [
                "--model",
                "gmnet",
                "--image",
                str(source),
                "--output",
                str(output),
                "--save-all-sidecars",
            ]
        )
        == 2
    )

    run_log = tmp_path / "expanded_run.json"
    record = json.loads(run_log.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: synthetic expansion failure"
    assert record["diagnostics"] == {"memory": {"peak_usage": 1}}
    assert not output.exists()
    assert not (tmp_path / "expanded.gain_map.safetensors").exists()
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))


def test_set_model_selection_overrides_an_ordinary_model_flag(capsys) -> None:
    assert (
        main(
            [
                "--model",
                "gmnet",
                "--set",
                "model=ltx2",
                "--print-config",
            ]
        )
        == 0
    )
    assert 'model = "ltx2"' in capsys.readouterr().out

    assert (
        main(
            [
                "--model",
                "ltx2",
                "--set",
                "model=gmnet",
                "--print-config",
            ]
        )
        == 0
    )
    assert 'model = "gmnet"' in capsys.readouterr().out


def test_cli_model_selection_overrides_config_for_routing(tmp_path, capsys) -> None:
    config = tmp_path / "run.toml"
    config.write_text('model = "gmnet"\n')

    assert main(["--config", str(config), "--model", "ltx2", "--print-config"]) == 0

    assert 'model = "ltx2"' in capsys.readouterr().out


def test_unknown_model_error_names_the_winning_surface(capfd) -> None:
    assert main(["--model", "unknown-family"]) == 2
    assert "--model: unknown model 'unknown-family'" in capfd.readouterr().err


def test_help_uses_valid_config_selection(tmp_path, capsys) -> None:
    config = tmp_path / "run.toml"
    config.write_text('model = "gmnet"\n')

    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(config), "--help"])

    assert exit_info.value.code == 0
    assert "GMNet SDR-to-HDR still expansion" in capsys.readouterr().out


@pytest.mark.parametrize("contents", [None, "model = [\n"])
def test_help_survives_missing_or_malformed_config(tmp_path, capsys, contents) -> None:
    config = tmp_path / "bad.toml"
    if contents is not None:
        config.write_text(contents)

    assert main(["--config", str(config), "--help"]) == 0

    assert "Native multimodal MLX inference" in capsys.readouterr().out


def test_explicit_model_help_survives_a_missing_config(tmp_path, capsys) -> None:
    config = tmp_path / "missing.toml"

    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(config), "--model", "gmnet", "--help"])

    assert exit_info.value.code == 0
    assert "GMNet SDR-to-HDR still expansion" in capsys.readouterr().out


def test_routing_ignores_flag_order(tmp_path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(b"stub")
    (tmp_path / "photo.exr").write_bytes(b"existing")
    assert (
        main(
            [
                "--image",
                str(source),
                "--output-dir",
                str(tmp_path),
                "--model",
                "gmnet",
            ]
        )
        == 2
    )


def test_gmnet_without_an_action_is_a_usage_error() -> None:
    assert main(["--model", "gmnet"]) == 2


def test_expand_refuses_a_missing_input() -> None:
    assert main(["--model", "gmnet", "--image", "/nonexistent/image.png"]) == 2


def test_expand_refuses_exr_inputs(tmp_path) -> None:
    source = tmp_path / "frame.exr"
    source.write_bytes(b"stub")
    assert main(["--model", "gmnet", "--image", str(source)]) == 2


def test_expand_refuses_when_nothing_would_be_written(tmp_path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(b"stub")
    assert main(["--model", "gmnet", "--image", str(source), "--no-exr", "--no-heic"]) == 2


def test_expand_refuses_replacing_the_source_image(tmp_path) -> None:
    source = tmp_path / "photo.heic"
    source.write_bytes(b"stub")
    assert (
        main(
            [
                "--model",
                "gmnet",
                "--image",
                str(source),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_expand_refuses_existing_targets_without_force(tmp_path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(b"stub")
    (tmp_path / "photo.exr").write_bytes(b"existing")
    assert (
        main(
            [
                "--model",
                "gmnet",
                "--image",
                str(source),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_dry_run_reports_the_checkpoint_and_writes_nothing(tmp_path, capsys, caplog) -> None:
    override = tmp_path / "custom.safetensors"
    override.write_bytes(b"converted")
    outputs = tmp_path / "renders"
    arguments = [
        "--model",
        "gmnet",
        "--dry-run",
        "--weights-path",
        str(override),
        "--output-dir",
        str(outputs),
    ]

    with caplog.at_level(logging.INFO, logger="kinomlx"):
        assert main(arguments) == 0

    # Without --json the same run states the resolved checkpoint for a reader.
    assert "Resources" in caplog.text
    assert capsys.readouterr().out == ""

    assert main([*arguments, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "gmnet"
    assert report["weights"] == str(override.absolute())
    assert not outputs.exists()


def test_dry_run_refuses_an_expansion(tmp_path) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"stub")
    with pytest.raises(ConfigError, match="Resource-only operations"):
        _assemble(["--dry-run", "--image", str(source)])


def test_json_error_is_a_single_machine_record(capsys) -> None:
    assert main(["--model", "gmnet", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "--image" in payload["error"]


def test_json_environment_is_honored(monkeypatch, capsys) -> None:
    monkeypatch.setenv("KINO_JSON", "1")
    assert main(["--model", "gmnet"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_success_payload_names_achieved_peak_and_records_model_provenance(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    from kinomlx.models.gmnet.command import _run_expand

    weights = tmp_path / "gmnet_synthetic.safetensors"
    variant = SimpleNamespace(value="synthetic")
    spec = SimpleNamespace(
        variant=variant,
        network_scale=1,
        peak_over_sdr_white=8.0,
        sdr_reference_white_nits=100.0,
    )
    result = SimpleNamespace(
        qmax_normalized=0.75,
        peak_linear=6.0,
        spec=spec,
    )

    class FakePlan:
        def reserve(self):
            return nullcontext(SimpleNamespace())

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            self.resources = SimpleNamespace(spec=spec, weights_path=weights)

        def expand(self, _request):
            return result

    class FakeSink:
        def __init__(self, *_args, **_kwargs):
            pass

        def write(self, *_args, **_kwargs):
            heic = tmp_path / "out.heic"
            return SimpleNamespace(
                heic=heic,
                to_dict=lambda: {"heic": str(heic)},
            )

    monkeypatch.setattr(
        "kinomlx.models.gmnet.command.plan_gmnet_output",
        lambda *_args: FakePlan(),
    )
    monkeypatch.setattr("kinomlx.models.gmnet.command.GMNetOutputSink", FakeSink)
    monkeypatch.setattr("kinomlx.models.gmnet.command.GMNetRunner", FakeRunner)
    invocation = SimpleNamespace(
        request=SimpleNamespace(),
        output=SimpleNamespace(),
        model_settings=SimpleNamespace(),
        settings=Settings(),
    )
    reporter = SimpleNamespace(memory_checkpoint=lambda _name: None)

    with caplog.at_level("INFO", logger="kinomlx.models.gmnet.command"):
        payload = _run_expand(invocation, reporter)

    generation = payload["generation"]
    assert generation["variant"] == "synthetic"
    assert generation["weights_path"] == str(weights)
    assert generation["model_sdr_reference_white_nits"] == 100.0
    assert generation["achieved_peak_over_sdr_white"] == 6.0
    assert generation["achieved_model_contract_peak_nits"] == 600.0
    assert generation["pq_delivery_reference_white_nits"] == 203.0
    assert generation["achieved_pq_delivery_peak_nits"] == 1218.0
    assert "peak_over_sdr_white" not in generation
    assert "Resources:" in caplog.text
    assert f"Weights: {weights}" in caplog.text


class TestSettingsEquivalence:
    """defaults < env < TOML tables < CLI < --set."""

    def test_default_variant(self, monkeypatch):
        monkeypatch.delenv("KINO_GMNET_VARIANT", raising=False)
        invocation = _assemble([], model_settings=GMNetSettings())
        assert invocation.model_settings.variant == "realworld"
        assert invocation.model_settings.weights_path is None

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("KINO_GMNET_VARIANT", "synthetic")
        assert _assemble([]).model_settings.variant == "synthetic"

    def test_toml_overrides_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KINO_GMNET_VARIANT", "synthetic")
        config = tmp_path / "run.toml"
        config.write_text('model = "gmnet"\n[model_settings]\nvariant = "realworld"\n')
        assert _assemble(["--config", str(config)]).model_settings.variant == "realworld"

    def test_toml_resolves_an_explicit_weights_path(self, tmp_path):
        config = tmp_path / "run.toml"
        weights = tmp_path / "custom.safetensors"
        config.write_text(f"[model_settings]\nweights_path = {json.dumps(str(weights))}\n")

        invocation = _assemble(["--config", str(config)])

        assert invocation.model_settings.weights_path == weights

    def test_cli_overrides_toml(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('[model_settings]\nvariant = "realworld"\n')
        invocation = _assemble(
            ["--config", str(config), "--variant", "synthetic"],
            model_settings=GMNetSettings(),
        )
        assert invocation.model_settings.variant == "synthetic"

    def test_set_overrides_cli(self):
        invocation = _assemble(
            ["--variant", "realworld", "--set", "model_settings.variant=synthetic"],
            model_settings=GMNetSettings(),
        )
        assert invocation.model_settings.variant == "synthetic"

    def test_infrastructure_settings_share_the_same_precedence(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text("[settings]\nquiet = true\nmlx_cache_limit_gb = 0.25\n")
        invocation = _assemble(["--config", str(config), "--no-quiet"])
        assert invocation.settings.quiet is False
        assert invocation.settings.mlx_cache_limit_gb == 0.25

    def test_cli_directory_replaces_a_config_exact_output(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('[output]\npath = "from-config.exr"\n')
        directory = tmp_path / "hdr"
        invocation = _assemble(["--config", str(config), "--output-dir", str(directory)])
        assert invocation.output.path is None
        assert invocation.output.directory == directory

    def test_set_directory_replaces_a_cli_exact_output(self, tmp_path):
        exact = tmp_path / "from-cli.exr"
        directory = tmp_path / "hdr"
        invocation = _assemble(
            [
                "--output",
                str(exact),
                "--set",
                f"output.directory={json.dumps(str(directory))}",
            ]
        )
        assert invocation.output.path is None
        assert invocation.output.directory == directory

    def test_unknown_settings_key_is_rejected(self):
        with pytest.raises(ConfigError, match="model_settings"):
            _assemble(["--set", "model_settings.nonsense=1"])

    def test_generation_tables_are_rejected(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('[generate]\nprompt = "hello"\n')
        with pytest.raises(ConfigError):
            _assemble(["--config", str(config)])

    def test_foreign_model_scalar_is_rejected(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('model = "ltx2"\n')
        with pytest.raises(ConfigError, match="does not select"):
            _assemble(["--config", str(config)])

    def test_invalid_variant_is_a_config_error(self):
        with pytest.raises(ConfigError, match="variant"):
            _assemble(["--set", "model_settings.variant=cinematic"])
