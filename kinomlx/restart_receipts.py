"""Observational receipts for the artifacts a saved-station restart consumes.

Identity here is evidence, not a gate: a missing fingerprint, a changed file, or
advisory metadata that disagrees with the recorded run is reported and carried
into the receipt, and structural validation still decides whether the tensors can
be used. Only a missing file stops the restart before loading.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kinomlx.config import ConfigError
from kinomlx.config.persistence import read_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import read_metadata

_LOG = logging.getLogger(__name__)


def restart_object(value: object, *, label: str, source: Path) -> dict[str, object]:
    """Admit one recorded object from a run log, naming the field that failed."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"restart run {source}: {label} must be an object")
    return dict(cast(Mapping[str, object], value))


@dataclass(frozen=True)
class RestartRunLog:
    """One decoded run log, with the objects every restart reads from it."""

    source: Path
    payload: dict[str, object]
    configuration: PreparedConfig
    effective: dict[str, object]
    outputs: dict[str, object]


def read_restart_run(run: Path, *, model: str) -> RestartRunLog:
    """Decode a saved run log and admit the objects a restart resolves against."""
    source = run.expanduser().absolute()
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read restart run {source}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid restart run JSON {source}: {exc}") from exc

    payload = restart_object(decoded, label="root", source=source)
    return RestartRunLog(
        source=source,
        payload=payload,
        configuration=read_configuration(payload, model=model, source=str(source)),
        effective=restart_object(
            payload.get("effective_config"), label="effective_config", source=source
        ),
        outputs=restart_object(payload.get("outputs"), label="outputs", source=source),
    )


def input_receipt(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None,
    expected_metadata: Mapping[str, str],
) -> dict[str, object]:
    """Inspect one consumed sidecar without turning identity into a load gate."""
    source = path.expanduser().absolute()
    if not source.is_file():
        raise ConfigError(f"restart {name} artifact does not exist: {source}")

    hash_error = None
    try:
        actual_sha256 = file_sha256(source)
    except OSError as exc:
        actual_sha256 = None
        hash_error = f"{type(exc).__name__}: {exc}"
        _LOG.warning(
            "Could not fingerprint restart input %s: %s; continuing to structural loading",
            source,
            exc,
        )

    matches_parent = None
    if expected_sha256 is not None and actual_sha256 is not None:
        matches_parent = actual_sha256 == expected_sha256
        if not matches_parent:
            _LOG.warning(
                "Restart input %s differs from the parent receipt "
                "(expected %s, got %s); continuing because identity is observational",
                source,
                expected_sha256,
                actual_sha256,
            )

    declared_metadata: dict[str, str] = {}
    metadata_mismatches: dict[str, dict[str, str | None]] = {}
    metadata_error = None
    try:
        metadata = read_metadata(source)
    except (OSError, ValueError) as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"
    else:
        declared_metadata = {key: metadata[key] for key in expected_metadata if key in metadata}
        metadata_mismatches = {
            key: {"expected": expected, "declared": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if metadata_mismatches:
            _LOG.warning(
                "Restart input %s has advisory metadata differences %s; continuing to "
                "structural tensor validation",
                source,
                ", ".join(sorted(metadata_mismatches)),
            )

    return {
        "path": str(source),
        "sha256": actual_sha256,
        "parent_sha256": expected_sha256,
        "matches_parent": matches_parent,
        "hash_error": hash_error,
        "declared_metadata": declared_metadata,
        "metadata_mismatches": metadata_mismatches,
        "metadata_error": metadata_error,
    }


__all__ = ["RestartRunLog", "input_receipt", "read_restart_run", "restart_object"]
