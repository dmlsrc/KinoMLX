"""Focused static contracts for the model-neutral configuration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kinomlx.config import (
    apply_set_overrides,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    merge_configs,
)


@dataclass(frozen=True)
class ExampleConfig:
    count: int = 1
    path: Path | None = None


nested_literal = {"sample": {"count": 2}}
merged: dict[str, object] = merge_configs(nested_literal)
overridden: dict[str, object] = apply_set_overrides(nested_literal, ["sample.count=3"])
coerced: dict[str, object] = coerce_dataclass_fields(
    ExampleConfig,
    {"path": "~/sample"},
    table="[sample]",
)
typed: ExampleConfig = dataclass_from_mapping(
    ExampleConfig,
    {"count": 4},
    table="[sample]",
)
serialized: dict[str, object] = dataclass_to_mapping(typed)
