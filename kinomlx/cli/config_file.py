"""KinoMLX invocation-file schema and loader."""

from __future__ import annotations

from pathlib import Path

from kinomlx.config import load_config, validate_top_level

from ._registry import config_registry


def load_invocation_file(path: str | Path, *, model: str = "ltx2") -> dict[str, object]:
    """Load and validate the top-level shape of one invocation file."""
    config = load_config(path)
    schema = config_registry().model(model)
    validate_top_level(
        config,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
        source=str(Path(path).expanduser()),
    )
    return config


__all__ = ["load_invocation_file"]
