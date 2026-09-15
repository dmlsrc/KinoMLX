"""Global cross-surface, restart precedence and source-policy acceptance.

These cases use independent expected values rather than comparing one generated
structure against another, so a coordinated mistake in the declarations cannot
make every surface consistently wrong.
"""

import ast
import contextlib
import io
import json
import tomllib
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from kinomlx.config.inspection import configuration_inventory
from kinomlx.model_registry import config_registry, model_choices
from kinomlx.models.ltx2.arguments import build_parser as ltx_parser
from kinomlx.models.ltx2.config import assemble as ltx_assemble
from kinomlx.settings import Settings
from tests.config.saved_records import saved_run_configuration

_REQUIRED_INPUT = {
    "flux2": ("--prompt", "a still"),
    "gmnet": ("--image", "source.png"),
    "krea2": ("--prompt", "a still"),
    "ltx2": ("--prompt", "a clip"),
}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_config(model: str, *arguments: str) -> dict[str, object]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["--model", model, "--print-config", *_REQUIRED_INPUT[model], *arguments])
    assert code == 0, err.getvalue()
    return tomllib.loads(out.getvalue())


@pytest.fixture
def clean_environment(monkeypatch):
    """Remove ambient product variables so each layer under test is explicit."""
    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    return monkeypatch


def test_every_model_classifies_every_registered_configuration_field():
    """AUTH-04 globally: no model reaches the registry with an unexplained gap."""
    registry = config_registry()
    unclassified: list[str] = []
    for model in model_choices():
        schema = registry.model(model)
        inventory = configuration_inventory(schema)
        for entry in inventory["fields"]:
            coverage = entry["station_coverage"]
            classified = coverage == "registered" and bool(entry["station_policies"])
            if coverage in {"registered", "not_registered"} and not classified:
                unclassified.append(f"{model}:{entry['path']}")
            if coverage == "no_restart":
                assert schema.restart_stations == ()
                assert inventory["restart"]["unavailable_reason"]
        for entry in inventory["fields"]:
            for policy in entry["station_policies"]:
                assert policy["reason"].strip(), (model, entry["path"])
    assert unclassified == []
    assert configuration_inventory(registry.model("gmnet"))["restart"]["stations"] == []


@pytest.mark.parametrize("model", tuple(_REQUIRED_INPUT))
def test_shared_output_concepts_keep_one_meaning_on_every_model(model, clean_environment):
    """UX-01: shared spellings admit the same units, presence and precedence."""
    clean_environment.setenv(f"KINO_{model.upper()}_OUTPUT_PREFIX", "from-environment")
    inherited = _print_config(model)
    assert inherited["output"]["prefix"] == "from-environment"
    assert "save_run_log" not in inherited["output"]

    explicit = _print_config(
        model,
        "--output-prefix",
        "from-flag",
        "--no-save-run-log",
        "--set",
        "output.save_effective_config=true",
    )
    assert explicit["output"]["prefix"] == "from-flag"
    assert explicit["output"]["save_run_log"] is False
    assert explicit["output"]["save_effective_config"] is True

    overridden = _print_config(
        model, "--output-prefix", "from-flag", "--set", "output.prefix=from-override"
    )
    assert overridden["output"]["prefix"] == "from-override"


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_shared_generation_concepts_use_the_same_units_and_precedence(model, clean_environment):
    """UX-01: seed and geometry mean the same thing wherever they are applicable."""
    clean_environment.setenv(f"KINO_{model.upper()}_SEED", "5")
    assert _print_config(model)["generate"]["seed"] == 5
    assert _print_config(model, "--seed", "9")["generate"]["seed"] == 9
    assert (
        _print_config(model, "--seed", "9", "--set", "generate.seed=11")["generate"]["seed"] == 11
    )
    geometry = _print_config(model, "--width", "512", "--height", "768")["generate"]
    assert (geometry["width"], geometry["height"]) == (512, 768)


def test_model_specific_conventions_are_visible_and_reject_unsupported_input(clean_environment):
    """UX-02: the generated reference states each distinction; bad input fails clearly."""
    registry = config_registry()
    declared = {
        model: registry.model(model).field(("model_settings", "transformer_dtype")).valid_text
        for model in ("flux2", "krea2", "ltx2")
    }
    assert declared["flux2"] == '"bfloat16" or "float16"'
    assert declared["krea2"] == declared["ltx2"] == '"bfloat16", "float16", or "float32"'

    inventory = {
        model: {
            entry["path"]: entry
            for entry in configuration_inventory(registry.model(model))["fields"]
        }
        for model in ("flux2", "krea2")
    }
    assert inventory["flux2"]["model_settings.variant"]["constraints"] == (
        '"base", "distilled", or "kv"'
    )
    assert inventory["krea2"]["model_settings.variant"]["constraints"] == '"raw" or "turbo"'

    assert (
        _print_config("krea2", "--transformer-dtype", "float32")["model_settings"][
            "transformer_dtype"
        ]
        == "float32"
    )
    out, err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
        pytest.raises(SystemExit) as failure,
    ):
        main(
            [
                "--model",
                "flux2",
                "--print-config",
                "--prompt",
                "a still",
                "--transformer-dtype",
                "float32",
            ]
        )
    assert failure.value.code != 0
    assert "float32" in err.getvalue()
    assert "--transformer-dtype" in err.getvalue()


def test_each_surface_reports_its_own_active_spelling(clean_environment):
    """UX-03: the same mistake names the flag, the variable or the config path."""
    clean_environment.setenv("KINO_KREA2_STEPS", "not-a-number")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert main(["--model", "krea2", "--print-config", "--prompt", "a still"]) != 0
    assert "KINO_KREA2_STEPS" in err.getvalue()

    clean_environment.delenv("KINO_KREA2_STEPS")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert (
            main(
                [
                    "--model",
                    "krea2",
                    "--print-config",
                    "--prompt",
                    "a still",
                    "--set",
                    "generate.steps=not-a-number",
                ]
            )
            != 0
        )
    assert "[generate].steps" in err.getvalue()


def _restart_run(tmp_path: Path, *, cache_limit: float | None) -> Path:
    latents = tmp_path / "parent.safetensors"
    latents.write_bytes(b"final")
    video = tmp_path / "parent.mp4"
    video.write_bytes(b"video")
    payload = {
        "status": "completed",
        "model": "ltx2",
        **saved_run_configuration(
            "ltx2",
            {
                "model": "ltx2",
                "generate": {
                    "prompt": "parent prompt",
                    "width": 768,
                    "height": 448,
                    "frames": 121,
                    "fps": 24.0,
                    "seed": 42,
                },
                "output": {"save_run_log": True},
                "settings": {} if cache_limit is None else {"mlx_cache_limit_gb": cache_limit},
                "model_settings": {"model_generation": "2.5"},
            },
        ),
        "outputs": {"final_latents": str(latents), "video": str(video)},
        "generation": {"model_generation": "2.5", "video_shape": [1, 3, 121, 448, 768]},
    }
    run = tmp_path / "parent_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


def _restart_cache_limit(run: Path, *arguments: str) -> float | None:
    invocation = ltx_assemble(
        ltx_parser().parse_args(["--restart", str(run), *arguments]),
        base_settings=Settings(),
    )
    return invocation.settings.mlx_cache_limit_gb


def test_restart_precedence_orders_environment_saved_file_flag_and_override(
    tmp_path, clean_environment
):
    """LAYER-01 on a restart run: five distinct values resolve in the declared order."""
    run = _restart_run(tmp_path, cache_limit=3.0)
    config = tmp_path / "restart.toml"
    config.write_text("[settings]\nmlx_cache_limit_gb = 5.0\n", encoding="utf-8")
    clean_environment.setenv("KINO_MLX_CACHE_LIMIT_GB", "1.0")

    assert (
        _restart_cache_limit(
            run,
            "--config",
            str(config),
            "--mlx-cache-limit-gb",
            "7.0",
            "--set",
            "settings.mlx_cache_limit_gb=9.0",
        )
        == 9.0
    )
    assert _restart_cache_limit(run, "--config", str(config), "--mlx-cache-limit-gb", "7.0") == 7.0
    assert _restart_cache_limit(run, "--config", str(config)) == 5.0
    assert _restart_cache_limit(run) == 3.0


def test_environment_supplies_a_restart_value_the_saved_run_omitted(tmp_path, clean_environment):
    """LAYER-01: the saved layer outranks the environment but is not required."""
    run = _restart_run(tmp_path, cache_limit=None)
    clean_environment.setenv("KINO_MLX_CACHE_LIMIT_GB", "1.0")
    assert _restart_cache_limit(run) == 1.0


def test_product_and_example_python_keep_one_runtime_import_graph(parsed_project_sources):
    """PY-01: no deferred-annotation shortcuts, and examples resolve at runtime."""
    examples = sorted((_PROJECT_ROOT / "examples").glob("*.py"))
    assert [path.name for path in examples] == [
        "flux2_generate.py",
        "gmnet_expand.py",
        "krea2_generate.py",
        "ltx2_distilled.py",
    ]
    trees = dict(parsed_project_sources)
    trees.update({path: ast.parse(path.read_text(encoding="utf-8")) for path in examples})
    offenders: list[str] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                offenders.extend(f"{path}: future {alias.name}" for alias in node.names)
            if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
                offenders.append(f"{path}: TYPE_CHECKING")
    assert offenders == []
    for boundary in config_registry().model("ltx2").api_boundaries:
        assert boundary.describe()["parameters"]


@pytest.mark.parametrize(
    ("model", "variable", "value", "expected"),
    [
        (
            "ltx2",
            "KINO_LTX2_SAMPLER",
            "bogus",
            "config error: environment KINO_LTX2_SAMPLER [generate].sampler: "
            "expected one of 'auto', 'deterministic', 'ancestral', got 'bogus'",
        ),
        (
            "krea2",
            "KINO_KREA2_STEPS",
            "0",
            "config error: environment KINO_KREA2_STEPS [generate].steps: "
            "must be greater than or equal to 1",
        ),
        (
            "krea2",
            "KINO_KREA2_STEPS",
            "abc",
            "config error: environment KINO_KREA2_STEPS [generate].steps: "
            "invalid literal for int() with base 10: 'abc'",
        ),
    ],
)
def test_environment_failures_carry_their_location_once(
    clean_environment, model, variable, value, expected
):
    """UX-03: a choice, a bound and a plain type all read the same way."""
    clean_environment.setenv(variable, value)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert main(["--model", model, "--print-config", *_REQUIRED_INPUT[model]]) != 0
    rendered = " ".join(err.getvalue().split())
    assert expected in rendered
    assert rendered.count(f"environment {variable}") == 1
