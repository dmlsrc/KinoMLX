"""Recursive JSON value types."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | Sequence[JsonValue] | Mapping[str, JsonValue]
type JsonObject = dict[str, JsonValue]

__all__ = ["JsonObject", "JsonScalar", "JsonValue"]


def load_json_object(path: Path | str) -> JsonObject:
    """Read one required JSON object with a path-labeled error."""
    source = Path(path).expanduser().resolve()
    try:
        serialized = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid JSON object at {source}") from exc
    except OSError as exc:
        raise OSError(f"cannot read JSON object at {source}: {exc}") from exc
    try:
        value = json.loads(serialized)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON object at {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {source}")
    return cast(JsonObject, value)
