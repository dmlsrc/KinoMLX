"""Generic torch-free conversion of plain PyTorch state dicts to safetensors."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import mlx.core as mx

from kinomlx.errors import KinoMLXError
from kinomlx.io.safetensors import load_weights, save_weights
from kinomlx.reporting import NullReporter, Reporter

from .output import WeightOutputError, reserved_weight_output
from .torch_checkpoint import RestrictedCheckpointError, load_restricted_checkpoint

_log = logging.getLogger(__name__)
SOURCE_COLLECTION = Path("weights-src")


class WeightConversionError(KinoMLXError, RuntimeError):
    """A generic checkpoint cannot be converted without guessing."""


@dataclass(frozen=True)
class GenericConversionReceipt:
    """Facts from one verified generic conversion."""

    output: Path
    source: Path
    source_sha256: str
    tensor_count: int
    parameter_count: int
    stripped_keys: int
    filtered_keys: int
    dropped_entries: tuple[str, ...]
    flagged_globals: tuple[str, ...]


def resolve_source(spec: Path | str) -> Path:
    """Resolve a literal path, collection-relative path, or unique basename."""
    literal = Path(spec)
    if literal.is_file():
        return literal
    direct = SOURCE_COLLECTION / spec
    if direct.is_file():
        _log.info("Source resolved from the source collection: %s", direct)
        return direct
    if SOURCE_COLLECTION.is_dir() and "/" not in str(spec):
        matches = sorted(
            candidate for candidate in SOURCE_COLLECTION.rglob(str(spec)) if candidate.is_file()
        )
        if len(matches) == 1:
            _log.info("Source resolved from the source collection: %s", matches[0])
            return matches[0]
        if matches:
            raise WeightConversionError(
                f"{str(spec)!r} is ambiguous in {SOURCE_COLLECTION}/: "
                + ", ".join(str(match) for match in matches)
            )
    raise WeightConversionError(
        f"no such checkpoint: {spec} (also looked under {SOURCE_COLLECTION}/)"
    )


def _is_tensor(value: object) -> TypeGuard[mx.array]:
    return isinstance(value, mx.array)


def _select_state_dict(
    tree: object,
    param_key: str | None,
) -> Mapping[object, object]:
    if param_key is not None:
        if not isinstance(tree, Mapping) or not isinstance(
            tree.get(param_key),
            Mapping,
        ):
            have = list(tree) if isinstance(tree, Mapping) else type(tree).__name__
            raise WeightConversionError(
                f"--param-key {param_key!r} is unavailable (checkpoint has {have})"
            )
        selected = tree[param_key]
        assert isinstance(selected, Mapping)
        return selected
    if (
        isinstance(tree, Mapping)
        and "params" in tree
        and "params_ema" in tree
        and isinstance(tree["params"], Mapping)
        and isinstance(tree["params_ema"], Mapping)
    ):
        raise WeightConversionError(
            "checkpoint carries both 'params' and 'params_ema'; pass --param-key "
            "to select the weights used by the model's inference path"
        )
    if not isinstance(tree, Mapping):
        raise WeightConversionError("checkpoint carries no recognizable tensor state dict")
    candidate_names = tuple(
        key
        for key in ("state_dict", "model", "net", "weights", "params", "params_ema")
        if isinstance(tree.get(key), Mapping)
    )
    has_direct_tensors = any(_is_tensor(value) for value in tree.values())
    if has_direct_tensors and candidate_names:
        raise WeightConversionError(
            "checkpoint carries root tensors and nested parameter mappings "
            f"{list(candidate_names)}; pass --param-key to select a nested mapping"
        )
    if has_direct_tensors:
        return tree
    if len(candidate_names) > 1:
        raise WeightConversionError(
            f"checkpoint carries multiple parameter mappings {list(candidate_names)}; "
            "pass --param-key to select the weights used by inference"
        )
    if candidate_names:
        selected = tree[candidate_names[0]]
        assert isinstance(selected, Mapping)
        return selected
    raise WeightConversionError("checkpoint carries no recognizable tensor state dict")


def _default_output(source: Path) -> Path:
    collection = (Path.cwd() / SOURCE_COLLECTION).absolute()
    if source.absolute().is_relative_to(collection):
        raise WeightConversionError(
            "state an output when converting from weights-src/; converted artifacts "
            "must not land inside the source collection"
        )
    return source.with_suffix(".safetensors")


def convert_checkpoint(
    source: Path | str,
    output: Path | str | None = None,
    *,
    param_key: str | None = None,
    strip_prefix: str = "module.",
    only_prefix: str = "",
    allow_suspicious: bool = False,
    force: bool = False,
    reporter: Reporter | None = None,
) -> GenericConversionReceipt:
    """Convert a plain tensor state dict without model-specific layout changes.

    Use a model-owned converter when inference requires exact key validation,
    tensor transposition, normalization folding, or checkpoint metadata. The
    generic path strips/filters names only and otherwise preserves tensor
    values and layouts.
    """
    sink = reporter if reporter is not None else NullReporter()
    phase = "convert generic checkpoint"
    sink.phase_start(phase, total=4, unit="step")
    try:
        source_path = resolve_source(source)
        target = _default_output(source_path) if output is None else Path(output).expanduser()
        if target.suffix.lower() != ".safetensors":
            raise WeightConversionError("conversion output must end in .safetensors")
        with source_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sink.phase_advance(phase)

        try:
            with reserved_weight_output(
                target,
                source=source_path,
                force=force,
            ) as temporary:
                checkpoint = load_restricted_checkpoint(
                    source_path,
                    allow_suspicious=allow_suspicious,
                )
                state = _select_state_dict(checkpoint.tree, param_key)
                sink.phase_advance(phase)

                tensors: dict[str, mx.array] = {}
                dropped: list[str] = []
                stripped = 0
                filtered = 0
                for raw_key, value in state.items():
                    key = str(raw_key)
                    if not _is_tensor(value):
                        dropped.append(key)
                        continue
                    if only_prefix and not key.startswith(only_prefix):
                        filtered += 1
                        continue
                    converted_key = (
                        key[len(strip_prefix) :]
                        if strip_prefix and key.startswith(strip_prefix)
                        else key
                    )
                    stripped += int(converted_key != key)
                    if converted_key in tensors:
                        raise WeightConversionError(
                            f"key conversion maps more than one tensor to {converted_key!r}"
                        )
                    tensors[converted_key] = mx.contiguous(value)
                if not tensors:
                    raise WeightConversionError("no tensors remain after checkpoint filtering")
                mx.eval(list(tensors.values()))
                sink.phase_advance(phase)

                metadata = {
                    "format": "generic-torch-state-dict",
                    "source_file": source_path.name,
                    "source_sha256": digest,
                    "converter": "kinomlx weights convert",
                    "strip_prefix": strip_prefix,
                    "only_prefix": only_prefix,
                    "param_key": param_key or "",
                }
                save_weights(temporary, tensors, metadata)
                loaded = load_weights(temporary)
                if set(loaded) != set(tensors):
                    raise WeightConversionError(
                        "saved safetensors key set differs from the converted state dict"
                    )
                sink.phase_advance(phase)
        except WeightOutputError as exc:
            raise WeightConversionError(str(exc)) from exc
        return GenericConversionReceipt(
            output=target,
            source=source_path,
            source_sha256=digest,
            tensor_count=len(tensors),
            parameter_count=sum(tensor.size for tensor in tensors.values()),
            stripped_keys=stripped,
            filtered_keys=filtered,
            dropped_entries=tuple(dropped),
            flagged_globals=checkpoint.flagged_globals,
        )
    except RestrictedCheckpointError as exc:
        raise WeightConversionError(str(exc)) from exc
    finally:
        sink.phase_end(phase)


__all__ = [
    "GenericConversionReceipt",
    "WeightConversionError",
    "convert_checkpoint",
    "resolve_source",
]
