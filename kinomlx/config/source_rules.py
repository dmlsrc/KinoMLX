"""Declared source decoding and filesystem reads before pure normalization."""

from collections.abc import Mapping
from pathlib import Path

from kinomlx.config_values import ConfigError

from .transforms import DELETE_FIELD, FieldPath, LayerTransform


def _prompt_file(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    value = values.get(("generate", "prompt_file"))
    if value is None:
        return {}
    if values.get(("generate", "prompt")) is not None:
        raise ConfigError("[generate]: prompt and prompt_file cannot both be set in the same layer")
    if not isinstance(value, str | Path):
        raise ConfigError("[generate].prompt_file must be a file path")
    path = Path(value).expanduser()
    if not path.is_file():
        raise ConfigError(f"[generate].prompt_file is not a file: {path}")
    try:
        prompt = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read UTF-8 prompt file {path}: {exc}") from exc
    if not prompt or "\x00" in prompt:
        raise ConfigError(f"prompt file must contain nonempty text without NUL characters: {path}")
    return {("generate", "prompt_file"): DELETE_FIELD, ("generate", "prompt"): prompt}


PROMPT_FILE = LayerTransform(
    name="prompt_file",
    reads=frozenset({("generate", "prompt"), ("generate", "prompt_file")}),
    writes=frozenset({("generate", "prompt"), ("generate", "prompt_file")}),
    run=_prompt_file,
    phase="source",
    effects=frozenset({"read_files"}),
)
