"""Packaging-neutral Krea 2 resource inspection and source preparation."""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.models.krea2.catalog import (
    UPSCALE2X_VAE_FILENAME,
    UPSCALE2X_VAE_REPO_ID,
    Krea2VAEDecoder,
    Krea2Variant,
    repository_id,
)
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.settings import Settings

from ._fixtures import make_snapshot


def _infrastructure(
    tmp_path: Path,
) -> Settings:
    return Settings(
        hf_home=tmp_path / "hf",
    )


def _install_cached_snapshot(
    hf_home: Path,
    source: Path,
    variant: Krea2Variant,
    revision: str,
) -> Path:
    cache = hf_home / "hub" / f"models--{repository_id(variant).replace('/', '--')}"
    snapshot = cache / "snapshots" / revision
    shutil.copytree(source, snapshot)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    return snapshot


def _upscale2x_candidate(source: Path, destination: Path) -> Path:
    weights = mx.load(str(source / "vae" / "diffusion_pytorch_model.safetensors"))
    weights["decoder.conv_out.weight"] = mx.zeros((12, 1, 3, 1, 1), dtype=mx.bfloat16)
    weights["decoder.conv_out.bias"] = mx.zeros((12,), dtype=mx.bfloat16)
    save_weights(destination, weights)
    return destination


def _install_cached_upscale2x_vae(
    hf_home: Path,
    candidate: Path,
    revision: str,
) -> Path:
    cache = hf_home / "hub" / f"models--{UPSCALE2X_VAE_REPO_ID.replace('/', '--')}"
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    selected = snapshot / UPSCALE2X_VAE_FILENAME
    shutil.copy2(candidate, selected)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    return selected


def test_non_hf_source_root_resolves_without_network_or_hub_assumptions(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "community", variant="turbo")
    resources = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=_infrastructure(tmp_path),
    )

    assert resources.variant is Krea2Variant.TURBO
    assert resources.source_root == source.resolve()
    assert resources.transformer.layout == "diffusers-sharded"
    assert resources.transformer.tensor_count == 430
    assert resources.text_encoder.tensor_count > resources.text_encoder.consumed_tensor_count
    assert resources.text_encoder.consumed_tensor_count == 386
    assert resources.vae.consumed_tensor_count == 104
    assert resources.capabilities.recipe_families == ("turbo",)
    assert resources.capabilities.negative_prompt is False


@pytest.mark.parametrize("cached", [False, True])
def test_component_overrides_replace_only_their_component(tmp_path: Path, cached: bool) -> None:
    base = make_snapshot(tmp_path / "base", variant="turbo")
    replacement = make_snapshot(
        tmp_path / "replacement",
        variant="turbo",
        transformer_marker=9.0,
    )
    if cached:
        base = _install_cached_snapshot(
            _infrastructure(tmp_path).hf_home, base, Krea2Variant.TURBO, "a" * 40
        )
    resources = prepare_resources(
        Krea2Settings(
            source_path=None if cached else base,
            transformer_path=replacement / "transformer",
            vae_path=replacement / "vae",
        ),
        infrastructure=_infrastructure(tmp_path),
    )

    assert resources.tokenizer.source_path == (base / "tokenizer").resolve()
    assert resources.text_encoder.source_path == (base / "text_encoder").resolve()
    assert resources.transformer.source_path == (replacement / "transformer").resolve()
    assert resources.vae.source_path == (replacement / "vae").resolve()


def test_path_free_selection_uses_exact_cached_main_ref(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "download", variant="raw")
    revision = "a" * 40
    cached = _install_cached_snapshot(
        _infrastructure(tmp_path).hf_home,
        source,
        Krea2Variant.RAW,
        revision,
    )

    resources = prepare_resources(
        Krea2Settings(variant="raw"),
        infrastructure=_infrastructure(tmp_path),
    )

    assert resources.source_root == cached
    assert resources.revision == revision


def test_upscale2x_selector_finds_its_one_file_cached_snapshot(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    infrastructure = _infrastructure(tmp_path)
    candidate = _upscale2x_candidate(source, tmp_path / "upscale2x.safetensors")
    cached = _install_cached_upscale2x_vae(
        infrastructure.hf_home,
        candidate,
        "c" * 40,
    )

    resources = prepare_resources(
        Krea2Settings(source_path=source, vae_decoder="upscale2x"),
        infrastructure=infrastructure,
    )

    assert resources.vae_decoder is Krea2VAEDecoder.UPSCALE2X
    assert resources.vae_decoder.output_scale == 2
    assert resources.vae.source_path == cached.resolve()
    assert resources.vae.consumed_tensor_count == 104
    assert resources.describe()["vae_decoder"] == {"name": "upscale2x", "output_scale": 2}


def test_explicit_vae_replacement_autodetects_upscale2x_and_checks_selector(
    tmp_path: Path,
) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    candidate = _upscale2x_candidate(source, tmp_path / "upscale2x.safetensors")
    infrastructure = _infrastructure(tmp_path)

    resources = prepare_resources(
        Krea2Settings(source_path=source, vae_path=candidate),
        infrastructure=infrastructure,
    )
    assert resources.vae_decoder is Krea2VAEDecoder.UPSCALE2X

    with pytest.raises(ValueError, match="contradicts.*upscale2x"):
        prepare_resources(
            Krea2Settings(source_path=source, vae_path=candidate, vae_decoder="qwen"),
            infrastructure=infrastructure,
        )


def test_missing_upscale2x_cache_reports_the_exact_one_file_download(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    expected = (
        r"hf download spacepxl/Wan2\.1-VAE-upscale2x "
        r"Wan2\.1_VAE_upscale2x_imageonly_real_v1\.safetensors"
    )
    with pytest.raises(FileNotFoundError, match=expected):
        prepare_resources(
            Krea2Settings(source_path=source, vae_decoder="upscale2x"),
            infrastructure=_infrastructure(tmp_path),
        )


def test_every_acquisition_hint_renders_one_copyable_command(tmp_path: Path) -> None:
    # The shared hint already wraps its command in backticks, so a caller that
    # wraps it again hands the user a literal backtick to copy.
    infrastructure = _infrastructure(tmp_path)
    messages = []
    with pytest.raises(FileNotFoundError, match="hf download") as missing_main:
        prepare_resources(Krea2Settings(), infrastructure=infrastructure)
    messages.append(str(missing_main.value))

    source = make_snapshot(tmp_path / "source", variant="turbo")
    with pytest.raises(FileNotFoundError, match="hf download") as missing_snapshot:
        prepare_resources(
            Krea2Settings(source_path=source, vae_decoder="upscale2x"),
            infrastructure=infrastructure,
        )
    messages.append(str(missing_snapshot.value))

    cache = infrastructure.hf_home / "hub" / f"models--{UPSCALE2X_VAE_REPO_ID.replace('/', '--')}"
    (cache / "snapshots" / ("d" * 40)).mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("d" * 40, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="hf download") as empty_snapshot:
        prepare_resources(
            Krea2Settings(source_path=source, vae_decoder="upscale2x"),
            infrastructure=infrastructure,
        )
    messages.append(str(empty_snapshot.value))

    assert f"is missing {UPSCALE2X_VAE_FILENAME}" in messages[-1]
    for message in messages:
        assert message.count("`") == 2
        assert "``" not in message


def test_missing_and_stale_hub_sources_report_exact_download_guidance(
    tmp_path: Path,
) -> None:
    infrastructure = _infrastructure(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"hf download krea/Krea-2-Turbo"):
        prepare_resources(Krea2Settings(), infrastructure=infrastructure)

    cache = infrastructure.hf_home / "hub" / "models--krea--Krea-2-Turbo"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("b" * 40, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"hf download krea/Krea-2-Turbo"):
        prepare_resources(Krea2Settings(), infrastructure=infrastructure)


def test_community_components_without_model_index_require_explicit_variant(
    tmp_path: Path,
) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    with pytest.raises(ValueError, match="explicit Krea 2 variant"):
        prepare_resources(
            Krea2Settings(
                tokenizer_path=source / "tokenizer",
                text_encoder_path=source / "text_encoder",
                transformer_path=source / "turbo.safetensors",
                vae_path=source / "vae",
            ),
            infrastructure=_infrastructure(tmp_path),
        )


def test_cross_component_dimensions_fail_during_source_inspection(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    config_path = source / "text_encoder" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["hidden_size"] = 2048
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="text hidden dimension"):
        prepare_resources(
            Krea2Settings(source_path=source),
            infrastructure=_infrastructure(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("attention_bias", True, "attention_bias"),
        ("hidden_act", "gelu", "hidden_act"),
        ("rope_type", "dynamic", "rope_type"),
    ],
)
def test_qwen_math_switches_fail_during_source_inspection(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    config_path = source / "text_encoder" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "rope_type":
        config["text_config"]["rope_parameters"][field] = value
    else:
        config["text_config"][field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        prepare_resources(
            Krea2Settings(source_path=source),
            infrastructure=_infrastructure(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dropout", 0.1),
        ("temperal_downsample", [True, True, False]),
    ],
)
def test_vae_math_switches_fail_during_source_inspection(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    config_path = source / "vae" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        prepare_resources(
            Krea2Settings(source_path=source),
            infrastructure=_infrastructure(tmp_path),
        )


def test_tokenizer_ids_must_fit_the_qwen_embedding_table(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    tokenizer_path = source / "tokenizer" / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"][0]["id"] = 151936
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds the text embedding vocabulary"):
        prepare_resources(
            Krea2Settings(source_path=source),
            infrastructure=_infrastructure(tmp_path),
        )


def test_incomplete_source_shards_fail_during_header_preflight(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source", variant="turbo")
    missing = source / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="shard"):
        prepare_resources(
            Krea2Settings(source_path=source),
            infrastructure=_infrastructure(tmp_path),
        )


def test_dry_run_report_names_every_source_a_generation_would_consume(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "community", variant="turbo")

    report = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=_infrastructure(tmp_path),
    ).describe()

    assert report["model"] == "krea2"
    assert report["variant"] == "turbo"
    assert report["source_root"] == str(source.resolve())
    assert report["is_distilled"] is True
    assert (report["steps"], report["guidance"]) == (8, 0.0)
    assert report["vae_decoder"] == {"name": "qwen", "output_scale": 1}
    assert report["check_adapter_weights"] is True
    components = report["components"]
    assert isinstance(components, list)
    # Every component states its resolved path, both identities, its binding
    # policy and its tensor accounting, with no key omitted for any kind.
    expected_keys = {
        "kind",
        "dtype",
        "source",
        "source_identity",
        "graph_identity",
        "layout",
        "source_tensors",
        "runtime_tensors",
        "excluded_tensors",
    }
    assert [entry["kind"] for entry in components] == [
        "tokenizer",
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert all(set(entry) == expected_keys for entry in components)
    assert [entry["dtype"] for entry in components] == [
        "json",
        "source",
        "bfloat16",
        "bfloat16",
    ]
    text_encoder = components[1]
    assert (text_encoder["source_tensors"], text_encoder["runtime_tensors"]) == (400, 386)
    assert text_encoder["excluded_tensors"] == 400 - 386
    assert text_encoder["source"] == str((source / "text_encoder").resolve())
    assert text_encoder["source_identity"].startswith("sha256:")
