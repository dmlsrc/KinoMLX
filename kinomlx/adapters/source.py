"""Shared local adapter identity, metadata, and selection admission."""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kinomlx._json import JsonObject
from kinomlx.io.identity import file_signature, payload_digest
from kinomlx.io.safetensors import inspect_safetensors

from .peft import PEFT_CONFIG_NAME, PEFT_WEIGHTS_NAME, companion_config, selected_link_chain
from .selection import AdapterSelection

MODEL_METADATA_KEYS = (
    "base_model",
    "base_model_name_or_path",
    "ss_sd_model_name",
    "ss_base_model_version",
    "model_version",
)
EMBEDDED_CONFIG_KEY = "lora_adapter_metadata"
# PEFT saves its tuner state under this prefix. Diffusers and community exports
# rename their keys, so a tensor spelled this way comes from a PEFT export.
PEFT_STATE_PREFIX = "base_model.model."


@dataclass(frozen=True)
class AdapterConfiguration:
    """The PEFT configuration one selected adapter file declares, if any."""

    values: JsonObject
    source: str | None = None
    # A PEFT export without its configuration, whose LoRA scale cannot be known.
    scale_unknown: bool = False


@dataclass(frozen=True)
class AdapterSource:
    path: Path
    tensor_names: tuple[str, ...]
    metadata: Mapping[str, str]
    config: JsonObject
    source_identity: str
    config_source: str | None = None
    scale_unknown: bool = False


def adapter_configuration(
    selected: Path, metadata: Mapping[str, str], tensor_names: Iterable[str]
) -> AdapterConfiguration:
    """Read the configuration beside a selected file or embedded in its metadata.

    ``selected`` is the path as chosen, before resolution, so a configuration
    beside a linked ``adapter_model.safetensors`` is still found. When both
    sources exist they must agree.
    """
    config = companion_config(selected)
    source = PEFT_CONFIG_NAME if config else None
    if EMBEDDED_CONFIG_KEY in metadata:
        embedded = metadata[EMBEDDED_CONFIG_KEY]
        if len(embedded.encode("utf-8")) > 1024 * 1024:
            raise ValueError("embedded adapter configuration exceeds the 1 MiB limit")
        try:
            value = json.loads(embedded)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid embedded Diffusers adapter configuration") from exc
        if isinstance(value, dict) and value and "peft_type" not in value:
            prefixes = {key.split(".", 1)[0] for key in value if "." in key}
            if len(prefixes) == 1 and all("." in key for key in value):
                value = {key.split(".", 1)[1]: item for key, item in value.items()}
        if not isinstance(value, dict) or not isinstance(value.get("peft_type"), str):
            raise ValueError("embedded adapter configuration must declare peft_type")
        if config and config != value:
            raise ValueError("embedded and companion adapter configurations disagree")
        config = cast(JsonObject, value)
        source = source or EMBEDDED_CONFIG_KEY
    scale_unknown = not config and (
        any(hop.name == PEFT_WEIGHTS_NAME for hop in selected_link_chain(selected))
        or any(name.startswith(PEFT_STATE_PREFIX) for name in tensor_names)
    )
    return AdapterConfiguration(config, source, scale_unknown)


def inspect_adapter_source(
    selected: AdapterSelection, *, model: str, categories: frozenset[str]
) -> AdapterSource:
    if not math.isfinite(selected.strength):
        raise ValueError("adapter strength must be finite")
    if unknown := set(selected.exclude) - categories:
        raise ValueError(f"unknown {model} adapter exclusions: {sorted(unknown)}")
    chosen = selected.path.expanduser()
    path = chosen.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{model} adapter does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{model} adapter is not a file: {path}")
    try:
        source = inspect_safetensors(path)
    except OSError as exc:
        raise OSError(f"cannot inspect {model} adapter {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid {model} adapter {path}: {exc}") from exc
    metadata = dict(source.metadata)
    configuration = adapter_configuration(chosen, metadata, source.tensor_names)
    config = configuration.values
    for key in MODEL_METADATA_KEYS:
        if isinstance(declared := config.get(key), str):
            metadata.setdefault(key, declared)
    return AdapterSource(
        path,
        source.tensor_names,
        metadata,
        config,
        f"source:{payload_digest({'source': file_signature(path), 'config': config})}",
        configuration.source,
        configuration.scale_unknown,
    )
