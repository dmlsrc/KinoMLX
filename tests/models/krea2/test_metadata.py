"""Krea 2 graph metadata diagnostics and finite-value validation."""

import json
from pathlib import Path

import pytest

from kinomlx.models.krea2.metadata import (
    load_json_object,
    parse_scheduler_config,
    parse_vae_config,
)

from ._fixtures import make_snapshot


def _rewrite(path: Path, **updates: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_json_read_error_names_the_os_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(OSError, match=r"cannot read JSON object.*missing\.json"):
        load_json_object(missing)


def test_vae_rejects_nonfinite_latent_statistics(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source")
    config = source / "vae" / "config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["latents_std"][0] = float("nan")
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"latents_std\[0\] must be finite"):
        parse_vae_config(config)


def test_vae_graph_error_names_the_field(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source")
    config = source / "vae" / "config.json"
    _rewrite(config, base_dim=95)

    with pytest.raises(ValueError, match=r"VAE base_dim: 95"):
        parse_vae_config(config)


def test_vae_accepts_the_compatible_diffusers_wan_config_shape(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source")
    config = source / "vae" / "config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["_class_name"] = "AutoencoderKLWan"
    payload["in_channels"] = payload.pop("input_channels")
    payload["out_channels"] = 12
    payload["scale_factor_spatial"] = 8
    config.write_text(json.dumps(payload), encoding="utf-8")

    parsed = parse_vae_config(config)

    assert parsed.input_channels == 3
    assert parsed.spatial_compression == 8


def test_scheduler_error_names_sampling_field(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "source")
    config = source / "scheduler" / "scheduler_config.json"
    _rewrite(config, use_karras_sigmas=True)

    with pytest.raises(ValueError, match=r"scheduler use_karras_sigmas: True"):
        parse_scheduler_config(config)
