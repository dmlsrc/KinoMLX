"""LTX-specific pure rules contributed through the shared configuration runner."""

from collections.abc import Mapping
from dataclasses import dataclass

from kinomlx.config.registry import ModelConfigSpec
from kinomlx.config.transforms import DELETE_FIELD, FieldPath, LayerTransform
from kinomlx.config_values import ConfigError

from .config_groups import LTX2ConfigGroup

_FRAMES = ("generate", "frames")
_DURATION = ("generate", "duration")
_AUTO_DURATION = ("generate", "auto_duration")


def _duration_selection(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    patch: dict[FieldPath, object] = {}
    if _AUTO_DURATION in values:
        automatic = values[_AUTO_DURATION]
        if not isinstance(automatic, bool):
            raise ConfigError("generate.auto_duration must be a boolean")
        patch[_AUTO_DURATION] = DELETE_FIELD
        if automatic:
            conflicts = tuple(path for path in (_FRAMES, _DURATION) if path in values)
            if conflicts:
                joined = " and ".join(".".join(path) for path in conflicts)
                raise ConfigError(
                    f"generate.auto_duration cannot be combined with {joined} "
                    "in the same configuration layer"
                )
            return {**patch, _FRAMES: None, _DURATION: None}
    if values.get(_FRAMES) is not None and values.get(_DURATION) is not None:
        raise ConfigError(
            "generate.frames and generate.duration cannot be combined "
            "in the same configuration layer"
        )
    if _FRAMES in values and _DURATION not in values:
        # An explicit frame count replaces a lower-precedence duration.
        patch[_DURATION] = None
    return patch


DURATION_SELECTION = LayerTransform(
    name="ltx_duration_selection",
    reads=frozenset({_FRAMES, _DURATION, _AUTO_DURATION}),
    writes=frozenset({_FRAMES, _DURATION, _AUTO_DURATION}),
    run=_duration_selection,
)


def _duration_serialization(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    if (
        _FRAMES in values
        and _DURATION in values
        and all(values[path] is None for path in (_FRAMES, _DURATION))
    ):
        return {_FRAMES: DELETE_FIELD, _DURATION: DELETE_FIELD, _AUTO_DURATION: True}
    return {}


DURATION_SERIALIZATION = LayerTransform(
    name="ltx_duration_serialization",
    reads=frozenset({_FRAMES, _DURATION}),
    writes=frozenset({_FRAMES, _DURATION, _AUTO_DURATION}),
    run=_duration_serialization,
    phase="serialization",
)


_TILING_MODE = ("generate", "vae_tiling", "mode")
_TILING_GEOMETRY = frozenset(
    ("generate", "vae_tiling", name)
    for name in (
        "temporal_tile_frames",
        "temporal_overlap_frames",
        "spatial_tile_pixels",
        "spatial_overlap_pixels",
    )
)
_TILING_SIZES = frozenset(
    ("generate", "vae_tiling", name) for name in ("temporal_tile_frames", "spatial_tile_pixels")
)


def _inactive_tiling(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    if values.get(_TILING_MODE) in {"auto", "single"}:
        return {path: DELETE_FIELD for path in _TILING_GEOMETRY if path in values}
    return {}


def _tiling_input(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    """Admit one authored layer's tiling intent, refusing a contradiction inside it.

    Only decoded input layers reach the source phase. A lower layer's stale
    geometry, including a restart parent's, is still dropped rather than
    refused, because that is precedence rather than a contradiction.

    A tile size selects custom tiling on its own, so naming one beside a mode
    that does not tile says two different things. An overlap only refines a tile
    size and carries a built-in default that earlier whole-configuration dumps
    wrote out beside an inactive mode, so it stays a redundancy to drop.
    """
    if not _TILING_GEOMETRY & values.keys():
        return {}
    mode = values.get(_TILING_MODE)
    if mode is None:
        # A tile value on its own can only mean custom tiling.
        return {_TILING_MODE: "custom"}
    if mode in {"auto", "single"} and _TILING_SIZES & values.keys():
        raise ConfigError('[generate].vae_tiling: tile values require mode = "custom"')
    return {}


def _tiling_selection(values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
    if values.get(_TILING_MODE) is None and _TILING_GEOMETRY & values.keys():
        return {_TILING_MODE: "custom"}
    return _inactive_tiling(values)


VAE_TILING_INPUT = LayerTransform(
    name="ltx_vae_tiling_input",
    reads=_TILING_GEOMETRY | {_TILING_MODE},
    writes=frozenset({_TILING_MODE}),
    run=_tiling_input,
    phase="source",
)

VAE_TILING_SELECTION = LayerTransform(
    name="ltx_vae_tiling_selection",
    reads=_TILING_GEOMETRY | {_TILING_MODE},
    writes=_TILING_GEOMETRY | {_TILING_MODE},
    run=_tiling_selection,
)

VAE_TILING_EFFECTIVE = LayerTransform(
    name="ltx_vae_tiling_effective",
    reads=_TILING_GEOMETRY | {_TILING_MODE},
    writes=_TILING_GEOMETRY,
    run=_inactive_tiling,
    phase="effective",
)


@dataclass(frozen=True)
class _SourceSelection:
    monolithic: tuple[FieldPath, ...]
    split: tuple[FieldPath, ...]
    sources: tuple[FieldPath, ...]
    generation: FieldPath
    video_vae: FieldPath
    video_vae_source: FieldPath
    gemma: FieldPath
    text_encoder: FieldPath

    def __call__(self, values: Mapping[FieldPath, object]) -> Mapping[FieldPath, object]:
        patch: dict[FieldPath, object] = {}
        if values.get(self.video_vae) is not None and self.video_vae_source not in values:
            patch[self.video_vae_source] = None
        if values.get(self.generation) is not None:
            patch.update({path: None for path in self.sources if path not in values})
        monolithic = any(values.get(path) is not None for path in self.monolithic)
        split = any(values.get(path) is not None for path in self.split)
        if monolithic != split:
            patch.update(dict.fromkeys(self.split if monolithic else self.monolithic))
        gemma = values.get(self.gemma) is not None
        text = values.get(self.text_encoder) is not None
        if gemma != text:
            patch[self.text_encoder if gemma else self.gemma] = None
        return patch


def source_selection_rule(schema: ModelConfigSpec) -> LayerTransform:
    """Compile source membership from the existing owned field groups."""
    groups = (
        LTX2ConfigGroup.MODEL_MONOLITHIC_SOURCE,
        LTX2ConfigGroup.MODEL_SPLIT_SOURCE,
        LTX2ConfigGroup.MODEL_SOURCE,
        LTX2ConfigGroup.MODEL_GENERATION_SELECTOR,
        LTX2ConfigGroup.MODEL_VIDEO_VAE_SELECTOR,
        LTX2ConfigGroup.MODEL_VIDEO_VAE_SOURCE,
        LTX2ConfigGroup.MODEL_GEMMA_SOURCE,
        LTX2ConfigGroup.MODEL_TEXT_ENCODER_SOURCE,
    )
    paths = tuple(schema.paths_in_group(group) for group in groups)
    if any(len(members) != 1 for members in paths[3:]):
        raise ValueError("LTX source selector groups must each identify exactly one field")
    selection = _SourceSelection(
        paths[0],
        paths[1],
        paths[2],
        paths[3][0],
        paths[4][0],
        paths[5][0],
        paths[6][0],
        paths[7][0],
    )
    return LayerTransform(
        name="ltx_source_selection",
        reads=frozenset(path for members in paths for path in members),
        writes=frozenset(
            (
                *selection.sources,
                *selection.monolithic,
                *selection.split,
                selection.video_vae_source,
                selection.gemma,
                selection.text_encoder,
            )
        ),
        run=selection,
        group_paths=tuple(zip(groups, paths, strict=True)),
    )
