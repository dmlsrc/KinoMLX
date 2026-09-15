"""Local admission, deterministic source identities, and tokenizer preparation."""

import itertools
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.modeling.artifacts import inspect_weight_source
from kinomlx.models.flux2 import resources
from kinomlx.models.flux2.command import run_flux2_command
from kinomlx.models.flux2.metadata import EXPECTED, validate_config, validate_model_index
from kinomlx.models.flux2.resources import Flux2Binding, Flux2Component
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.settings import Settings


def _component(root: Path, kind: str = "transformer") -> Flux2Component:
    source_path = root / f"source-{kind}"
    source_path.mkdir()
    (source_path / "config.json").write_text("{}")
    weight_path = source_path / "model.safetensors"
    save_weights(
        weight_path,
        {
            "weight": mx.arange(12).reshape(3, 4).astype(mx.bfloat16),
            "unused": mx.ones(2, dtype=mx.bfloat16),
        },
    )
    source = inspect_weight_source(source_path, filenames=("model.safetensors",))
    bindings = (
        Flux2Binding(weight_path, "weight", (3, 4), "runtime"),
        Flux2Binding(weight_path, "unused", (2,), "unused-language-output"),
    )
    return Flux2Component(kind, source, "graph-v1", source.source_fingerprint, bindings)


def test_incomplete_shard_index_is_rejected(tmp_path):
    component = _component(tmp_path)
    root = component.source.source_path
    index = {"weight_map": {"weight": "absent.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError, match="absent"):
        inspect_weight_source(root, filenames=("model.safetensors",))


def test_truncated_source_is_rejected_without_loading(tmp_path):
    component = _component(tmp_path)
    path = component.bindings[0].source_file
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload|offset|size|truncated"):
        inspect_weight_source(path.parent, filenames=(path.name,))


@pytest.mark.parametrize("value", [True, 2, None])
def test_incompatible_config_rejected(tmp_path, value):
    config = dict(EXPECTED["transformer"], patch_size=value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="patch_size"):
        validate_config("transformer", path)


@pytest.mark.parametrize("kind", ["Flux2KleinPipeline", "Flux2KleinKVPipeline"])
@pytest.mark.parametrize("distilled", [False, True])
def test_model_index_admission_without_weights(tmp_path, kind, distilled):
    path = tmp_path / "model_index.json"
    for payload in ({"_class_name": kind}, {"_class_name": kind, "is_distilled": distilled}):
        path.write_text(json.dumps(payload))
        validate_model_index(path, is_distilled=distilled)
    for payload in (
        {"_class_name": "OtherPipeline"},
        {"_class_name": kind, "is_distilled": not distilled},
        {"_class_name": kind, "is_distilled": int(distilled)},
    ):
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Klein"):
            validate_model_index(path, is_distilled=distilled)


def test_override_failure_precedes_default_discovery(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("default snapshot was consulted before override validation")

    monkeypatch.setattr(resources, "resolve_cached_snapshot", forbidden)
    with pytest.raises((OSError, ValueError)):
        resources.inspect_resources(
            Flux2Settings(vae_path=tmp_path / "absent"),
            infrastructure=Settings(),
        )


def test_command_refuses_generation_and_conflicting_operations(capsys):
    assert run_flux2_command(["--steps", "4"]) == 2
    with pytest.raises(SystemExit) as exc:
        run_flux2_command(["--prepare"])
    assert exc.value.code == 2
    assert run_flux2_command(["--only-non-defaults"]) == 2


def test_command_config_resolution_and_exclusive_save(tmp_path):
    path = tmp_path / "config.toml"
    args = [
        "--variant",
        "base",
        "--set",
        "model_settings.variant=kv",
        "--save-config",
        str(path),
        "--only-non-defaults",
    ]
    assert run_flux2_command(args) == 0
    assert 'variant = "kv"' in path.read_text()
    assert run_flux2_command(args) == 2


@pytest.mark.heavy
def test_real_profiles_inventory_and_component_deduplication(tmp_path):
    infrastructure = Settings.from_env().with_overrides()
    graphs = []
    for variant in ("base", "distilled", "kv"):
        try:
            graph = resources.inspect_resources(
                Flux2Settings(variant=variant), infrastructure=infrastructure
            )
        except FileNotFoundError:
            pytest.skip("bundled Klein checkpoints are not cached")
        assert [len(c.bindings) for c in graph.components] == [0, 399, 233, 251]
        assert [len(c.active_bindings) for c in graph.components] == [0, 298, 233, 250]
        again = resources.inspect_resources(
            Flux2Settings(variant=variant), infrastructure=infrastructure
        )
        assert graph == again
        graphs.append(graph)
    kv = graphs[-1]
    overrides = {f"{c.kind}_path": c.source.source_path for c in kv.components}
    local = resources.inspect_resources(
        Flux2Settings(variant="kv", **overrides),
        infrastructure=replace(infrastructure, hf_home=tmp_path / "absent-hub"),
    )
    assert local.source_root is None
    assert [c.identity for c in local.components] == [c.identity for c in kv.components]
    assert len({g.component("transformer").identity for g in graphs}) == 3
    assert len({g.recipe_identity for g in graphs}) == 3
    # Sharing is driven by component identity, never the profile label.
    for kind in ("tokenizer", "text_encoder", "vae"):
        components = [g.component(kind) for g in graphs]
        for left, right in itertools.pairwise(components):
            if left.source.source_fingerprint == right.source.source_fingerprint:
                assert left.identity == right.identity


def test_tokenizer_tables_are_derived_once_for_validation_and_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kinomlx.modeling.artifacts import TokenizerSource
    from kinomlx.models.flux2.loading import load_tokenizer

    source = TokenizerSource(
        tmp_path, tmp_path / "tokenizer.json", tmp_path / "tokenizer_config.json", "source", 151643
    )
    tables = SimpleNamespace(pad_token_id=151643)
    calls = []
    monkeypatch.setattr(
        resources, "load_tokenizer_tables", lambda *args: calls.append(args) or tables
    )
    monkeypatch.setattr("kinomlx.models.flux2.loading.QwenByteBPETokenizer", lambda tables: tables)
    component = resources.tokenizer_component(source)
    assert load_tokenizer(component) is tables
    assert load_tokenizer(component) is tables
    assert len(calls) == 1


@pytest.mark.parametrize("kind", resources.KINDS)
def test_single_override_keeps_default_discovery_for_other_flux_components(
    tmp_path, monkeypatch, kind
):
    from types import SimpleNamespace

    snapshot = tmp_path / "snapshot"
    override = tmp_path / "replacement"
    components = {}
    for name in resources.KINDS:
        directory = tmp_path / name
        directory.mkdir()
        components[name] = _component(directory, name)
    calls = {}

    def inspect(name, path):
        calls[name] = path
        return components[name].source

    monkeypatch.setattr(resources, "inspect_component_source", inspect)
    monkeypatch.setattr(resources, "validate_config", lambda *a, **k: None)
    monkeypatch.setattr(resources, "validate_model_index", lambda *a, **k: None)
    monkeypatch.setattr(resources, "_validate_weights", lambda *a, **k: ())
    monkeypatch.setattr(resources, "weight_component", lambda name, source, **k: components[name])
    monkeypatch.setattr(
        resources,
        "resolve_cached_snapshot",
        lambda *a: SimpleNamespace(path=snapshot, revision="test"),
    )
    result = resources.inspect_resources(
        Flux2Settings(**{f"{kind}_path": override}),
        infrastructure=Settings(hf_home=tmp_path),
    )
    assert result.source_root == snapshot
    assert calls == {
        name: override if name == kind else snapshot / name for name in resources.KINDS
    }
