"""PEFT configuration discovery for selected adapter files."""

import json
from pathlib import Path

import pytest

from kinomlx.adapters.peft import companion_config
from kinomlx.adapters.source import adapter_configuration
from tests.adapters._snapshot import link_as_snapshot

CONFIG = {"peft_type": "LORA", "r": 32, "lora_alpha": 64}
PEFT_KEYS = (
    "base_model.model.blocks.0.to_q.lora_A.weight",
    "base_model.model.blocks.0.to_q.lora_B.weight",
)


def _export(folder: Path, *, config: dict[str, object] | None = CONFIG) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    if config is not None:
        (folder / "adapter_config.json").write_text(json.dumps(config))
    return weights


def test_snapshot_links_keep_the_companion_configuration(tmp_path: Path) -> None:
    source = _export(tmp_path / "export")
    weights, _ = link_as_snapshot(
        tmp_path / "hub", source, source.with_name("adapter_config.json"), folder="RL-Lora"
    )
    resolved = weights.resolve()
    assert resolved.name != "adapter_model.safetensors"

    found = adapter_configuration(weights, {}, PEFT_KEYS)
    assert found.values == CONFIG
    assert found.source == "adapter_config.json"
    assert not found.scale_unknown
    # The blob alone has lost the names its configuration was found by.
    assert companion_config(resolved) == {}
    assert adapter_configuration(resolved, {}, PEFT_KEYS).scale_unknown


def test_a_readable_link_to_the_snapshot_file_still_finds_it(tmp_path: Path) -> None:
    source = _export(tmp_path / "export")
    weights, _ = link_as_snapshot(tmp_path / "hub", source, source.with_name("adapter_config.json"))
    readable = tmp_path / "style.safetensors"
    readable.symlink_to(weights)
    assert companion_config(readable) == CONFIG


def test_a_plain_export_folder_reads_its_configuration(tmp_path: Path) -> None:
    weights = _export(tmp_path / "export")
    assert adapter_configuration(weights, {}, PEFT_KEYS).values == CONFIG


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "renamed.safetensors"])
def test_a_peft_export_without_configuration_has_an_unknown_scale(
    tmp_path: Path, name: str
) -> None:
    weights = _export(tmp_path / "export", config=None).rename(tmp_path / "export" / name)
    found = adapter_configuration(weights, {}, PEFT_KEYS)
    assert found.values == {}
    assert found.source is None
    assert found.scale_unknown


def test_the_standard_name_alone_marks_an_export_without_configuration(tmp_path: Path) -> None:
    weights = _export(tmp_path / "export", config=None)
    assert adapter_configuration(weights, {}, ("blocks.0.to_q.lora_A.weight",)).scale_unknown


def test_a_community_file_without_configuration_is_not_flagged(tmp_path: Path) -> None:
    weights = tmp_path / "community.safetensors"
    weights.write_bytes(b"weights")
    names = ("diffusion_model.blocks.0.to_q.lora_A.weight", "diffusion_model.blocks.0.to_q.alpha")
    found = adapter_configuration(weights, {"lora_alpha": "384", "lora_rank": "384"}, names)
    assert found == adapter_configuration(weights, {}, names)
    assert found.source is None
    assert not found.scale_unknown


def test_embedded_configuration_is_named_as_its_source(tmp_path: Path) -> None:
    weights = tmp_path / "pytorch_lora_weights.safetensors"
    weights.write_bytes(b"weights")
    metadata = {"lora_adapter_metadata": json.dumps(CONFIG)}
    found = adapter_configuration(weights, metadata, ("transformer.blocks.0.lora_A.weight",))
    assert found.values == CONFIG
    assert found.source == "lora_adapter_metadata"


def test_disagreeing_embedded_and_companion_configurations_are_rejected(tmp_path: Path) -> None:
    weights = _export(tmp_path / "export")
    metadata = {"lora_adapter_metadata": json.dumps({**CONFIG, "lora_alpha": 32})}
    with pytest.raises(ValueError, match="disagree"):
        adapter_configuration(weights, metadata, PEFT_KEYS)
