"""Packaging-neutral Model tokenizer and safetensors source inspection."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kinomlx._json import load_json_object
from kinomlx.io.identity import artifact_set_fingerprint
from kinomlx.io.safetensors import inspect_safetensors


class SourceLayout(StrEnum):
    """Physical source layouts normalized before weight preparation."""

    TOKENIZER = "tokenizer-json"
    SINGLE = "single-safetensors"
    SHARDED = "sharded-safetensors"


@dataclass(frozen=True)
class TensorFile:
    """One validated source file and its header-only tensor names."""

    path: Path
    keys: tuple[str, ...]
    shapes: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class WeightSource:
    """One component source independent of model-specific target bindings."""

    source_path: Path
    config_path: Path
    layout: SourceLayout
    tensor_files: tuple[TensorFile, ...]
    index_path: Path | None
    source_fingerprint: str

    @property
    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(*source.keys for source in self.tensor_files)

    @property
    def tensor_count(self) -> int:
        return sum(len(source.keys) for source in self.tensor_files)

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return header-only source shapes keyed by tensor name."""
        return {name: shape for source in self.tensor_files for name, shape in source.shapes}


@dataclass(frozen=True)
class TokenizerSource:
    """The two tokenizer assets consumed by the native tokenizer derivation."""

    source_path: Path
    tokenizer_json: Path
    tokenizer_config: Path
    source_fingerprint: str
    maximum_token_id: int


def _required_file(path: Path, *, label: str) -> Path:
    logical = path.expanduser().absolute()
    if not logical.is_file():
        raise FileNotFoundError(f"Model {label} is missing: {logical}")
    return logical


def inspect_tokenizer_source(path: Path | str) -> TokenizerSource:
    """Resolve a tokenizer directory or tokenizer.json file."""
    selected = Path(path).expanduser().absolute()
    if selected.is_dir():
        root = selected
        tokenizer_json = root / "tokenizer.json"
    elif selected.is_file() and selected.name == "tokenizer.json":
        root = selected.parent
        tokenizer_json = selected
    else:
        raise FileNotFoundError(
            f"Model tokenizer source must be a directory or tokenizer.json: {selected}"
        )
    tokenizer_json = _required_file(tokenizer_json, label="tokenizer.json")
    tokenizer_config = _required_file(
        root / "tokenizer_config.json",
        label="tokenizer_config.json",
    )
    # Syntax, top-level shape, and embedding-bound IDs are validated before any
    # runtime import. Full tokenizer semantics remain tokenizer-derivation policy.
    document = load_json_object(tokenizer_json)
    load_json_object(tokenizer_config)
    model = document.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("Model tokenizer model.vocab must be a JSON object")
    raw_ids = list(model["vocab"].values())
    added_tokens = document.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise ValueError("Model tokenizer added_tokens must be an array")
    for index, token in enumerate(added_tokens):
        if not isinstance(token, dict):
            raise ValueError(f"Model tokenizer added_tokens[{index}] must be a JSON object")
        raw_ids.append(token.get("id"))
    if not raw_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in raw_ids
    ):
        raise ValueError("Model tokenizer IDs must be non-negative integers")
    maximum_token_id = max(raw_ids)
    fingerprint = artifact_set_fingerprint(
        (
            ("tokenizer.json", tokenizer_json),
            ("tokenizer_config.json", tokenizer_config),
        )
    )
    return TokenizerSource(
        source_path=root,
        tokenizer_json=tokenizer_json,
        tokenizer_config=tokenizer_config,
        source_fingerprint=fingerprint,
        maximum_token_id=maximum_token_id,
    )


def _config_path(selected: Path, fallback: Path | None) -> Path:
    local = selected / "config.json" if selected.is_dir() else selected.parent / "config.json"
    if local.is_file():
        return local.absolute()
    if fallback is not None and fallback.is_file():
        return fallback.expanduser().absolute()
    raise FileNotFoundError(f"Model component has no config.json beside {selected}")


def _index_files(index_path: Path) -> tuple[Path, ...]:
    index = load_json_object(index_path)
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError(f"Model safetensors index has no weight_map: {index_path}")
    weight_map: dict[str, str] = {}
    for key, value in raw_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Model safetensors index contains a non-string mapping: {index_path}")
        relative = Path(value)
        if relative.is_absolute() or relative.name != value:
            raise ValueError(f"Model safetensors index contains an unsafe shard path: {value!r}")
        weight_map[key] = value
    names = tuple(sorted(set(weight_map.values())))
    files = tuple(_required_file(index_path.parent / name, label=f"shard {name}") for name in names)
    actual_by_key: dict[str, Path] = {}
    for shard in files:
        inventory = inspect_safetensors(shard)
        for key in inventory.tensor_names:
            previous = actual_by_key.get(key)
            if previous is not None:
                raise ValueError(
                    f"Model shards repeat tensor {key!r} in {previous.name} and {shard.name}"
                )
            actual_by_key[key] = shard
    for key, name in weight_map.items():
        actual = actual_by_key.get(key)
        if actual is None:
            raise ValueError(f"Model index maps missing tensor {key!r} to {name!r}")
        if actual.name != name:
            raise ValueError(
                f"Model index maps tensor {key!r} to {name!r}, found it in {actual.name!r}"
            )
    return files


def _select_weight_files(
    selected: Path,
    *,
    filenames: tuple[str, ...],
) -> tuple[SourceLayout, Path | None, tuple[Path, ...]]:
    if selected.is_file():
        if selected.name.endswith(".safetensors.index.json"):
            return SourceLayout.SHARDED, selected, _index_files(selected)
        if selected.suffix == ".safetensors":
            return SourceLayout.SINGLE, None, (selected,)
        raise ValueError(f"unsupported Model component file: {selected}")
    if not selected.is_dir():
        raise FileNotFoundError(f"Model component source does not exist: {selected}")
    indexes = tuple(sorted(selected.glob("*.safetensors.index.json")))
    if len(indexes) > 1:
        raise ValueError(f"Model component directory has multiple safetensors indexes: {selected}")
    if indexes:
        return SourceLayout.SHARDED, indexes[0], _index_files(indexes[0])
    candidates = tuple(selected / name for name in filenames if (selected / name).is_file())
    if not candidates:
        generic = tuple(sorted(selected.glob("*.safetensors")))
        if len(generic) == 1:
            candidates = generic
    if len(candidates) != 1:
        raise ValueError(
            f"Model component directory needs one recognized safetensors source: {selected}"
        )
    return SourceLayout.SINGLE, None, (candidates[0],)


def inspect_weight_source(
    path: Path | str,
    *,
    filenames: tuple[str, ...],
    config_fallback: Path | None = None,
) -> WeightSource:
    """Resolve and header-inspect one single-file or indexed component source."""
    selected = Path(path).expanduser().absolute()
    config_path = _config_path(selected, config_fallback)
    layout, index_path, files = _select_weight_files(selected, filenames=filenames)
    inspected = tuple(inspect_safetensors(file) for file in files)
    tensor_files = tuple(
        TensorFile(
            path=inventory.path,
            keys=inventory.tensor_names,
            shapes=tuple((tensor.name, tensor.shape) for tensor in inventory.tensors),
        )
        for inventory in inspected
    )
    seen: set[str] = set()
    for source in tensor_files:
        duplicates = seen & set(source.keys)
        if duplicates:
            sample = min(duplicates)
            raise ValueError(f"Model component repeats tensor {sample!r} across source files")
        seen.update(source.keys)
    artifacts: list[tuple[str, Path]] = [("config.json", config_path)]
    if index_path is not None:
        artifacts.append(("weights.index.json", index_path))
    artifacts.extend((f"weight:{file.name}", file) for file in files)
    return WeightSource(
        source_path=selected,
        config_path=config_path,
        layout=layout,
        tensor_files=tensor_files,
        index_path=index_path,
        source_fingerprint=artifact_set_fingerprint(artifacts),
    )


__all__ = [
    "SourceLayout",
    "TensorFile",
    "TokenizerSource",
    "WeightSource",
    "inspect_tokenizer_source",
    "inspect_weight_source",
]
