"""Krea stage, source, and public recipe configuration decisions."""

import json
from pathlib import Path

import pytest

from kinomlx.api_boundaries import ConfigurationOwner
from kinomlx.cli.resource_operation import Operation
from kinomlx.config import ConfigError
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.config_schema import CONFIG_SCHEMA
from kinomlx.models.krea2.restart import RestartConfig
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.settings import Settings

from .test_restart import _write_latents, _write_run, _write_text_conditioning


def _stage_records(
    *, model_settings: Krea2Settings | None = None, restart: RestartConfig | None = None
) -> dict[str, object]:
    return {
        "settings": Settings(),
        "model_settings": model_settings or Krea2Settings(),
        "operation": Operation(),
        "generate": Krea2Request(prompt="scene"),
        "output": ImageOutputConfig(),
        "restart": restart or RestartConfig(run=Path("source.json")),
    }


@pytest.mark.parametrize(
    ("stage", "model_settings", "expected"),
    [
        ("denoise", Krea2Settings(source_path=Path("other")), (("model_settings", "source_path"),)),
        ("decode", Krea2Settings(source_path=Path("other")), (("model_settings", "source_path"),)),
        ("denoise", Krea2Settings(transformer_dtype="float16"), ()),
        (
            "decode",
            Krea2Settings(transformer_dtype="float16"),
            (("model_settings", "transformer_dtype"),),
        ),
        ("denoise", Krea2Settings(vae_dtype="float16"), ()),
        ("decode", Krea2Settings(vae_dtype="float16"), ()),
    ],
)
def test_model_settings_follow_the_consumer_stage(
    stage: str, model_settings: Krea2Settings, expected: tuple[tuple[str, str], ...]
) -> None:
    changed = CONFIG_SCHEMA.stage_changes(
        _stage_records(model_settings=model_settings),
        _stage_records(),
        stage=stage,
        active_explicit=frozenset({("model_settings", field) for _, field in expected}),
    )
    assert changed == expected


def test_restart_latent_selector_rejects_same_valued_explicit_denoise_input() -> None:
    parent = _stage_records(restart=RestartConfig(run=Path("source.json"), stage="denoise"))
    assert (
        CONFIG_SCHEMA.stage_changes(parent, parent, stage="denoise", active_explicit=frozenset())
        == ()
    )
    assert CONFIG_SCHEMA.stage_changes(
        parent,
        parent,
        stage="denoise",
        active_explicit=frozenset({("restart", "latents")}),
    ) == (("restart", "latents"),)


def _source_run(tmp_path: Path) -> Path:
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    run = tmp_path / "run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    return run


def _assemble_restart(run: Path, stage: str, *args: str):
    return assemble(
        build_parser().parse_args(["--restart", str(run), "--restart-from", stage, *args]),
        base_settings=Settings(),
    )


def test_transformer_precision_changes_only_when_denoising_repeats(tmp_path: Path) -> None:
    run = _source_run(tmp_path)
    denoise = _assemble_restart(run, "denoise", "--transformer-dtype", "float16")
    assert denoise.model_settings.transformer_dtype == "float16"
    with pytest.raises(ConfigError, match="model_settings.*transformer_dtype"):
        _assemble_restart(run, "decode", "--transformer-dtype", "float16")
    decode = _assemble_restart(run, "decode", "--vae-dtype", "float16")
    assert decode.model_settings.vae_dtype == "float16"


def test_current_vae_path_rebases_inherited_decoder_but_not_current_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _source_run(tmp_path)
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["configuration"]["values"]["model_settings"].update(
        vae_path="saved/vae", vae_decoder="qwen"
    )
    run.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("KINO_KREA2_VAE_PATH", "hidden/env/vae")
    inherited = _assemble_restart(run, "decode")
    assert inherited.model_settings.vae_path == Path("saved/vae")
    assert inherited.model_settings.vae_decoder == "qwen"
    replaced = _assemble_restart(run, "decode", "--vae-path", "current/vae")
    assert replaced.model_settings.vae_decoder is None
    explicit = _assemble_restart(
        run, "decode", "--vae-path", "current/vae", "--vae-decoder", "upscale2x"
    )
    assert explicit.model_settings.vae_decoder == "upscale2x"


def test_krea_public_boundaries_include_direct_recipes_and_resource_selection() -> None:
    boundaries = {
        boundary.operation.__name__: boundary for boundary in CONFIG_SCHEMA.api_boundaries
    }
    assert {"prepare_resources", "generate_krea2", "generate_krea2_edit", "restart_krea2"} <= set(
        boundaries
    )
    for name in ("generate_krea2", "generate_krea2_edit", "restart_krea2"):
        request = next(role for role in boundaries[name].parameters if role.name == "request")
        assert isinstance(request.owner, ConfigurationOwner)
        assert request.owner.table == ("generate",)
        assert request.owner.projection
