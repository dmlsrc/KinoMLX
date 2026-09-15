"""Independent LTX station, output-context and public API ownership cases."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_rules import validate_capabilities
from kinomlx.models.ltx2 import config as ltx2_config
from kinomlx.models.ltx2 import resources
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble, validate_for_execution
from kinomlx.models.ltx2.config_schema import CONFIG_SCHEMA
from kinomlx.models.ltx2.output import write_generation
from kinomlx.models.ltx2.output_config import OutputConfig
from kinomlx.models.ltx2.restart_rules import RESTART_PHASE_CHOICES
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import Settings
from tests.cli.test_restart import _assemble, _source_run


def test_all_ltx_fields_have_explicit_reasons_and_direct_api_owners() -> None:
    # The restart phase choices are the station names; nothing restates them.
    assert CONFIG_SCHEMA.restart_stations is RESTART_PHASE_CHOICES
    assert set(CONFIG_SCHEMA.restart_stations) == {"stage-2", "decode"}
    for table in CONFIG_SCHEMA.tables:
        for field in table.fields:
            if field.virtual:
                continue
            assert {policy.station for policy in field.station_policies} == {
                "stage-2",
                "decode",
            }, (*table.path, field.name)
            assert all(policy.reason.strip() for policy in field.station_policies)
    operations = {
        boundary.operation.__qualname__: boundary.describe()
        for boundary in CONFIG_SCHEMA.api_boundaries
    }
    assert {
        "prepare_resources",
        "generate_distilled",
        "restart_distilled",
    } <= operations.keys()
    generated = operations["generate_distilled"]["parameters"]
    conditioning = next(item for item in generated if item["name"] == "conditioning")
    assert conditioning["classification"] == "runtime"
    assert conditioning["serializable"] is False
    assert next(item for item in generated if item["name"] == "request")["table"] == "generate"
    restart = operations["restart_distilled"]["parameters"]
    assert next(item for item in restart if item["name"] == "restart")["classification"] == (
        "runtime"
    )


def test_decode_keeps_equal_explicit_transformer_policy_but_rejects_change(
    tmp_path: Path,
) -> None:
    run, _ = _source_run(tmp_path)
    same = _assemble(run, "--transformer-dtype", "bfloat16")
    assert same.model_settings.transformer_dtype == "bfloat16"
    with pytest.raises(ConfigError, match=r"\[model_settings\]\.transformer_dtype"):
        _assemble(run, "--transformer-dtype", "float16")
    with pytest.raises(ConfigError, match=r"\[model_settings\]\.transformer_dtype"):
        _assemble(run, "--set", 'model_settings.transformer_dtype="float16"')


def test_stage_2_can_change_transformer_policy_but_not_parent_generation(
    tmp_path: Path,
) -> None:
    run, _ = _source_run(tmp_path)
    resumed = _assemble(run, "--restart-from", "stage-2", "--transformer-dtype", "float16")
    assert resumed.model_settings.transformer_dtype == "float16"
    with pytest.raises(ConfigError, match=r"\[model_settings\]\.model_generation"):
        _assemble(run, "--restart-from", "stage-2", "--model-generation", "2.3")


def test_stage_2_accepts_new_transformer_source_but_decode_locks_it(tmp_path: Path) -> None:
    run, _ = _source_run(tmp_path)
    alternative = tmp_path / "alternative-transformer.safetensors"
    stage_2 = _assemble(run, "--restart-from", "stage-2", "--transformer-path", str(alternative))
    assert stage_2.model_settings.transformer_path == alternative
    with pytest.raises(ConfigError, match=r"\[model_settings\]\.transformer_path"):
        _assemble(run, "--transformer-path", str(alternative))


def test_whole_record_toml_change_reaches_the_same_decode_lock(tmp_path: Path) -> None:
    run, _ = _source_run(tmp_path)
    config = tmp_path / "replay.toml"
    config.write_text(
        f'[restart]\nrun = "{run}"\nphase = "decode"\n'
        '[model_settings]\ntransformer_dtype = "float16"\n',
        encoding="utf-8",
    )
    from kinomlx.models.ltx2.arguments import build_parser
    from kinomlx.models.ltx2.config import assemble

    with pytest.raises(ConfigError, match=r"\[model_settings\]\.transformer_dtype"):
        assemble(build_parser().parse_args(["--config", str(config)]), base_settings=Settings())


def test_unavailable_tensor_artifacts_reject_true_but_allow_false(tmp_path: Path) -> None:
    run, _ = _source_run(tmp_path)
    with pytest.raises(
        ConfigError, match="save_text_conditioning is not produced by a decode restart"
    ):
        validate_for_execution(_assemble(run, "--save-text-conditioning"))
    with pytest.raises(ConfigError, match="save_media_conditioning is not produced"):
        validate_for_execution(_assemble(run, "--save-media-conditioning"))
    allowed = _assemble(run, "--no-save-latents")
    validate_for_execution(allowed)


@pytest.mark.parametrize(
    ("config", "hdr", "error"),
    [
        (OutputConfig(save_hdr_heic_frames=True), False, "save_hdr_heic_frames requires HDR"),
        (OutputConfig(vsr_spatial_mode="balanced"), True, "vsr_spatial_mode"),
        (OutputConfig(target_fps=60), True, "target_fps"),
        (OutputConfig(save_vae_frames=True), True, "save_vae_frames"),
        (OutputConfig(vsr_save_original=True), True, "vsr_save_original"),
    ],
)
def test_hdr_output_context_rejects_unsupported_delivery(
    config: OutputConfig, hdr: bool, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        validate_capabilities(config, {"hdr": hdr, "fps": 24.0}, location="[output]")


def test_hdr_output_context_accepts_heic_and_save_all_original_implication() -> None:
    validate_capabilities(
        OutputConfig(save_hdr_heic_frames=True), {"hdr": True, "fps": 24.0}, location="[output]"
    )
    validate_capabilities(
        OutputConfig(vsr_save_original=True, save_all_sidecars=True),
        {"hdr": True, "fps": 24.0},
        location="[output]",
    )


def test_direct_output_checks_hdr_context_and_closes_failed_generation(tmp_path: Path) -> None:
    generation = Mock()
    with pytest.raises(ValueError, match="save_hdr_heic_frames requires HDR"):
        write_generation(
            generation,
            OutputConfig(path=tmp_path / "result.mp4", save_hdr_heic_frames=True),
            fps=24.0,
        )
    generation.close.assert_called_once_with()


def _counted_model_settings_coercions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the table label of every LTX2Settings coercion one assemble runs."""
    tables: list[str] = []
    original = ltx2_config.coerce_dataclass_fields

    def counted(cls, mapping, *, table):
        if cls is LTX2Settings:
            tables.append(table)
        return original(cls, mapping, table=table)

    monkeypatch.setattr(ltx2_config, "coerce_dataclass_fields", counted)
    return tables


def test_one_invocation_coerces_the_model_settings_table_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _counted_model_settings_coercions(monkeypatch)

    assemble(
        build_parser().parse_args(["--prompt", "a probe prompt", "--transformer-dtype", "float16"]),
        base_settings=Settings(),
    )

    assert tables == ["[model_settings]"]


def test_a_restart_coerces_each_side_of_the_comparison_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run, _paths = _source_run(tmp_path)
    tables = _counted_model_settings_coercions(monkeypatch)

    _assemble(run)

    # The parent side coerces its own table under its own label; neither side
    # coerces twice.
    assert tables == ["[model_settings]", "[restart source model_settings]"]


def test_direct_resource_record_rule_runs_before_source_file_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resources,
        "_discover_generation_selection",
        lambda *_args: pytest.fail("invalid record reached source discovery"),
    )
    with pytest.raises(ValueError, match="gemma_path and text_encoder_path cannot be combined"):
        resources.prepare_resources(
            LTX2Settings(gemma_path=Path("gemma"), text_encoder_path=Path("text.safetensors")),
            infrastructure=Settings(),
        )
