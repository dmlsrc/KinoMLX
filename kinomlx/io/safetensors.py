"""Safetensors I/O helpers - typed, opinionated wrappers.

Wraps ``mlx.core.load`` / ``mlx.core.save_safetensors`` for the
project's two safetensors use cases:

- **Weight loading** - read model checkpoints and LoRA files into a
  ``dict[str, mx.array]`` for assignment into MLX modules.
- **Cache writing** - bake derived weights (with metadata such as
  schema version) into a single file the loader can re-read fast.

For ``.npy`` / ``.npz`` / ``.gguf`` use ``mlx.core.load`` directly;
these helpers assume safetensors throughout.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx._typing import JsonObject

from .atomic import atomic_output_path


def read_header(path: Path | str) -> JsonObject:
    """Read the complete safetensors JSON header without loading tensors."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(0)
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            raise ValueError(
                f"{path}: truncated safetensors header (declared length exceeds file size)"
            )
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        parsed = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{path}: invalid safetensors header") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    return cast(JsonObject, parsed)


def read_dtypes(path: Path | str) -> dict[str, str]:
    """Return raw safetensors dtype names without materializing tensor data."""
    header = read_header(path)
    return {
        key: str(entry["dtype"])
        for key, entry in header.items()
        if key != "__metadata__" and isinstance(entry, dict) and "dtype" in entry
    }


def read_u8_tensor(path: Path | str, name: str) -> bytes:
    """Seek-read one one-dimensional U8 tensor without loading neighboring weights."""
    source = Path(path)
    header = read_header(source)
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"{source}: no tensor named {name!r}")
    if entry.get("dtype") != "U8":
        raise ValueError(f"{source}: tensor {name!r} is not U8")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 0
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
    ):
        raise ValueError(f"{source}: tensor {name!r} has an invalid U8 asset layout")
    start, end = offsets
    if start < 0 or end < start or end - start != shape[0]:
        raise ValueError(f"{source}: tensor {name!r} has inconsistent U8 asset offsets")
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{source}: truncated safetensors length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        data_start = 8 + header_length
        if end > file_size - data_start:
            raise ValueError(f"{source}: tensor {name!r} extends beyond the file payload")
        stream.seek(8 + header_length + start)
        payload = stream.read(end - start)
    if len(payload) != end - start:
        raise ValueError(f"{source}: tensor {name!r} has a truncated U8 payload")
    return payload


def load_weights(path: Path | str) -> dict[str, mx.array]:
    """Load a safetensors file's tensors into a name -> array dict.

    Discards metadata.  Use :func:`load_weights_with_metadata` when
    you need both.
    """
    weights, _metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        mx.load(
            str(path),
            return_metadata=True,
            format="safetensors",
        ),
    )
    return weights


def load_weights_with_metadata(
    path: Path | str,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Load weights and string-keyed metadata together."""
    weights, metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        mx.load(
            str(path),
            return_metadata=True,
            format="safetensors",
        ),
    )
    return weights, metadata


def save_weights(
    path: Path | str,
    weights: Mapping[str, mx.array],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Save weights (and optional metadata) to a safetensors file.

    ``metadata`` keys and values must both be strings - that's the
    safetensors format constraint, not ours.  Encode numeric or
    structured data (e.g. schema versions, layout flags) as strings
    at this boundary; decode on the read side.
    """
    with atomic_output_path(path, temp_suffix=".tmp.safetensors") as temporary:
        mx.save_safetensors(
            str(temporary),
            dict(weights),
            metadata=dict(metadata) if metadata else None,
        )


def read_metadata(path: Path | str) -> dict[str, str]:
    """Read metadata from a safetensors file without materializing tensors.

    Useful for cache schema-version checks: open, read the version
    string, decide whether to bail and re-bake - without paying the
    cost of loading gigabytes of weights into memory.

    Reads the safetensors header natively - an 8-byte little-endian length
    prefix followed by a JSON header whose ``__metadata__`` key holds the
    string map - so this stays free of the safetensors and numpy packages.
    ``mx.load`` handles the tensor path and doesn't need them either.
    """
    header = read_header(path)
    raw_metadata = header.get("__metadata__")
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{path}: safetensors metadata must be an object")
    return {str(key): str(value) for key, value in raw_metadata.items()}


__all__ = [
    "load_weights",
    "load_weights_with_metadata",
    "read_dtypes",
    "read_header",
    "read_metadata",
    "read_u8_tensor",
    "save_weights",
]
