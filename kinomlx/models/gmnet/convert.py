"""Model-specific GMNet checkpoint conversion.

The generic :mod:`kinomlx.weights` layer safely reconstructs tensor-only
PyTorch checkpoint trees and reserves outputs. This module owns the GMNet
contract layered on top: exact generator keys, variant identification,
provenance metadata, and end-to-end validation through the GMNet loader.

Source checkpoints live in the repo-root ``weights-src/`` collection
(one subfolder per weights owner, upstream filenames, sha256-verified
at collection time - see its README). An input that does not exist as
given resolves against that collection relative to the working
directory, so documented commands work with bare filenames.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.errors import KinoMLXError
from kinomlx.io.safetensors import save_weights
from kinomlx.models.gmnet.catalog import (
    GMNetVariant,
    variant_for_source_sha256,
    variant_spec,
    variant_weights_path,
)
from kinomlx.models.gmnet.net import EXPECTED_CHECKPOINT_KEYS, load_gmnet_weights
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.weights.convert import WeightConversionError
from kinomlx.weights.convert import resolve_source as resolve_generic_source
from kinomlx.weights.output import WeightOutputError, reserved_weight_output
from kinomlx.weights.torch_checkpoint import (
    RestrictedCheckpointError,
    load_restricted_checkpoint,
    scan_pickle_globals,
    suspicious_globals,
)


class CheckpointConversionError(KinoMLXError, RuntimeError):
    """A typed refusal or failure while converting a source checkpoint."""


def resolve_source(spec: Path | str) -> Path:
    """Resolve through the model-neutral source-collection convention."""
    try:
        return resolve_generic_source(spec)
    except WeightConversionError as exc:
        raise CheckpointConversionError(str(exc)) from exc


def _state_dict_from_tree(tree: object, source: Path) -> dict[str, mx.array]:
    if not isinstance(tree, Mapping):
        raise CheckpointConversionError(
            f"{source} does not unpickle to a state dict (got {type(tree).__name__})"
        )
    tensors: dict[str, mx.array] = {}
    for key, value in tree.items():
        if not isinstance(value, mx.array):
            raise CheckpointConversionError(
                f"{source} entry {key!r} is not a tensor ({type(value).__name__}); "
                "GMNet checkpoints carry only the generator state dict"
            )
        if value.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise CheckpointConversionError(
                f"{source} entry {key!r} has unsupported dtype {value.dtype}; "
                "GMNet generator weights must be floating-point"
            )
        name = str(key)
        name = name.removeprefix("module.")
        tensors[name] = mx.contiguous(value)
    return tensors


@dataclass(frozen=True)
class ConversionReceipt:
    """What one completed conversion produced and where it came from."""

    output: Path
    variant: GMNetVariant | None
    source: Path
    source_sha256: str
    tensor_count: int
    parameter_count: int
    flagged_globals: tuple[str, ...]


def convert_checkpoint(
    source: Path | str,
    output: Path | str | None = None,
    *,
    declared_variant: GMNetVariant | None = None,
    allow_suspicious: bool = False,
    force: bool = False,
    cache_dir: Path | str | None = None,
    reporter: Reporter | None = None,
) -> ConversionReceipt:
    """Convert one GMNet ``.pth`` checkpoint into a verified safetensors file.

    The variant is identified by the source file's sha256 when it is one of
    the two published checkpoints; otherwise ``declared_variant`` must name
    it (a retrain still converts, but its numeric contract is the caller's
    claim). A source that does not exist as given resolves against the
    ``weights-src/`` collection. Without an explicit ``output``, the file
    lands in the editable checkout's model weights directory, or below the
    configured KinoMLX cache for an installed package.
    """
    sink = reporter if reporter is not None else NullReporter()
    phase = "convert GMNet checkpoint"
    sink.phase_start(phase, total=4, unit="step")
    try:
        source_path = resolve_source(source)
        with source_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sink.phase_advance(phase)

        detected = variant_for_source_sha256(digest)
        variant = detected if detected is not None else declared_variant
        if detected is not None and declared_variant is not None and detected != declared_variant:
            raise CheckpointConversionError(
                f"{source_path} is the published {detected.value} checkpoint "
                f"(sha256 {digest[:12]}...), not {declared_variant.value}"
            )

        if output is not None:
            target = Path(output).expanduser()
        elif variant is not None:
            if cache_dir is None:
                from kinomlx.settings import Settings

                cache_dir = Settings.from_env_fields("cache_dir").cache_dir
            target = variant_weights_path(variant, cache_dir)
        else:
            raise CheckpointConversionError(
                f"{source_path} (sha256 {digest[:12]}...) is not a published GMNet "
                "checkpoint; pass --declare-variant and an explicit --output path"
            )
        if target.suffix.lower() != ".safetensors":
            raise CheckpointConversionError("conversion output must end in .safetensors")
        try:
            with reserved_weight_output(
                target,
                source=source_path,
                force=force,
            ) as temporary:
                try:
                    checkpoint = load_restricted_checkpoint(
                        source_path,
                        allow_suspicious=allow_suspicious,
                    )
                except RestrictedCheckpointError as exc:
                    raise CheckpointConversionError(str(exc)) from exc
                tensors = _state_dict_from_tree(checkpoint.tree, source_path)

                provided = frozenset(tensors)
                missing = sorted(EXPECTED_CHECKPOINT_KEYS - provided)
                unexpected = sorted(provided - EXPECTED_CHECKPOINT_KEYS)
                if missing or unexpected:
                    raise CheckpointConversionError(
                        f"{source_path} is not a GMNet generator state dict: "
                        f"missing {missing[:4]}{'...' if len(missing) > 4 else ''}, "
                        f"unexpected {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}"
                    )
                mx.eval(list(tensors.values()))
                sink.phase_advance(phase)

                metadata = {
                    "format": "gmnet-generator-state-dict",
                    "model": "gmnet",
                    "variant": variant.value if variant is not None else "unknown",
                    "source_file": source_path.name,
                    "source_sha256": digest,
                    "license": "MIT",
                    "converter": "kinomlx weights convert gmnet",
                    "tensor_layout": (
                        "upstream torch OIHW conv weights; transposed to NHWC at load"
                    ),
                }
                if detected is not None:
                    metadata["source_url"] = variant_spec(detected).source_url
                save_weights(temporary, tensors, metadata)
                sink.phase_advance(phase)

                # Verify the artifact end to end: it must load back through the model
                # loader, not merely reopen as safetensors.
                load_gmnet_weights(temporary)
                sink.phase_advance(phase)
        except WeightOutputError as exc:
            raise CheckpointConversionError(str(exc)) from exc

        return ConversionReceipt(
            output=target,
            variant=variant,
            source=source_path,
            source_sha256=digest,
            tensor_count=len(tensors),
            parameter_count=sum(tensor.size for tensor in tensors.values()),
            flagged_globals=checkpoint.flagged_globals,
        )
    finally:
        sink.phase_end(phase)


__all__ = [
    "CheckpointConversionError",
    "ConversionReceipt",
    "convert_checkpoint",
    "resolve_source",
    "scan_pickle_globals",
    "suspicious_globals",
]
