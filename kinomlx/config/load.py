"""TOML loading and ordered config composition."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .merge import merge_configs
from .validate import ConfigError


def load_config(path: str | Path) -> dict[str, object]:
    """Load one TOML file and include its path in every parse error."""
    try:
        config_path = Path(path).expanduser()
    except RuntimeError as exc:
        raise ConfigError(f"cannot resolve config path {path}: {exc}") from exc
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    try:
        config: dict[str, object] = tomllib.loads(raw.decode("utf-8"))
        return config
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc


def compose_config(
    base_paths: list[str | Path] | None,
    specific_path: str | Path | None = None,
) -> dict[str, object]:
    """Merge optional base files followed by one invocation-specific file."""
    layers = [load_config(path) for path in (base_paths or [])]
    if specific_path is not None:
        layers.append(load_config(specific_path))
    return merge_configs(*layers)
