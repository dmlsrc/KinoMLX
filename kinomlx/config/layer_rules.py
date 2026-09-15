"""Shared selection semantics declared once for model schema contributions."""

from collections.abc import Mapping

from kinomlx.config_values import ConfigError

from .transforms import FieldPath, LayerTransform

_SEED = ("generate", "seed")
_SEEDS = ("generate", "seeds")
_OUTPUT_PATH = ("output", "path")
_OUTPUT_DIRECTORY = ("output", "directory")
_OUTPUT_PREFIX = ("output", "prefix")


def _seed_selection(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    scalar = values.get(_SEED) is not None
    multiple = values.get(_SEEDS) is not None
    if scalar and multiple:
        raise ConfigError("[generate]: seed and seeds are mutually exclusive")
    if scalar or multiple:
        return {_SEEDS if scalar else _SEED: None}
    return {}


def _output_selection(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    if values.get(_OUTPUT_PATH) is None and (
        _OUTPUT_DIRECTORY in values or _OUTPUT_PREFIX in values
    ):
        return {_OUTPUT_PATH: None}
    return {}


SEED_SELECTION = LayerTransform(
    name="seed_selection",
    reads=frozenset({_SEED, _SEEDS}),
    writes=frozenset({_SEED, _SEEDS}),
    run=_seed_selection,
    phase="before_implications",
)

OUTPUT_SELECTION = LayerTransform(
    name="output_selection",
    reads=frozenset({_OUTPUT_PATH, _OUTPUT_DIRECTORY, _OUTPUT_PREFIX}),
    writes=frozenset({_OUTPUT_PATH}),
    run=_output_selection,
)
