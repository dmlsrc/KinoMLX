"""Resolve TOML, environment, and CLI input into typed runtime config."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from kinomlx.config import (
    ConfigError,
    ConfigGroup,
    apply_set_overrides,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    merge_configs,
    normalize_output_selection,
    validate_top_level,
)
from kinomlx.models.ltx2.artifacts import LTX2ArtifactConfig
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import (
    NATIVE_FPS,
    DistilledRequest,
    resolved_frame_count_for_duration,
)
from kinomlx.output import (
    default_hdr_exr_directory,
    default_hdr_heic_directory,
    default_vae_frame_directory,
)
from kinomlx.settings import Settings

from ._registry import MODAL_MODEL_SPECS, MODEL_SPECS, config_registry
from .config_file import load_invocation_file
from .config_records import DEFAULT_OUTPUT_PREFIX, OutputConfig
from .restart import RestartConfig, RestartManifest, load_restart_manifest

_VAE_TILING_GEOMETRY_FIELDS = frozenset(
    {
        "temporal_tile_frames",
        "temporal_overlap_frames",
        "spatial_tile_pixels",
        "spatial_overlap_pixels",
    }
)


def _normalize_inactive_vae_tiling(
    generation: Mapping[str, object],
) -> dict[str, object]:
    """Remove geometry that is inactive under an automatic or single-tile policy."""
    normalized = dict(generation)
    tiling = normalized.get("vae_tiling")
    if not isinstance(tiling, Mapping) or tiling.get("mode") not in {"auto", "single"}:
        return normalized
    normalized["vae_tiling"] = {
        name: value for name, value in tiling.items() if name not in _VAE_TILING_GEOMETRY_FIELDS
    }
    return normalized


def _is_resolved_duration_pair(generation: Mapping[str, object]) -> bool:
    """Return whether frames are the exact lattice resolution of duration and fps."""
    frames = generation.get("frames")
    duration = generation.get("duration")
    fps = generation.get("fps", NATIVE_FPS)
    if isinstance(frames, bool) or not isinstance(frames, int):
        return False
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return False
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        return False
    if not math.isfinite(duration) or duration <= 0:
        return False
    if not math.isfinite(fps) or fps <= 0:
        return False
    return frames == resolved_frame_count_for_duration(float(duration), float(fps))


def sanitize_output_prefix(prefix: str | None) -> str:
    """Return the shell-friendly filename prefix used for generated paths."""
    value = (prefix or DEFAULT_OUTPUT_PREFIX).strip() or DEFAULT_OUTPUT_PREFIX
    sanitized = [
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ]
    return "".join(sanitized).strip("._") or DEFAULT_OUTPUT_PREFIX


def build_timestamped_output_path(
    directory: Path | str,
    prefix: str | None,
    *,
    now: datetime | None = None,
) -> Path:
    """Build ``<directory>/<prefix>_<local timestamp>.mp4`` without writing it."""
    timestamp = (now if now is not None else datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_directory = Path(directory).expanduser()
    output_prefix = sanitize_output_prefix(prefix)
    return output_directory / f"{output_prefix}_{timestamp}.mp4"


def _output_environment_layer() -> dict[str, object]:
    """Resolve output-directory environment fallbacks below TOML and CLI."""
    overrides = OutputConfig.overrides_from_env_fields("directory")
    return {"output": overrides} if overrides else {}


def _prepare_layer(layer: dict[str, object]) -> dict[str, object]:
    """Apply output shorthands that must remain local to one precedence layer."""
    schema = config_registry().model("ltx2")
    prepared = _normalize_model_selection(
        normalize_output_selection(schema.expand_layer_implications(layer))
    )
    generation = prepared.get("generate")
    if isinstance(generation, dict) and "auto_duration" in generation:
        prepared = dict(prepared)
        generation = dict(generation)
        auto_duration = generation.pop("auto_duration")
        if not isinstance(auto_duration, bool):
            raise ConfigError("generate.auto_duration must be a boolean")
        if auto_duration is True:
            conflicts = tuple(name for name in ("frames", "duration") if name in generation)
            if conflicts:
                joined = " and ".join(f"generate.{name}" for name in conflicts)
                raise ConfigError(
                    f"generate.auto_duration cannot be combined with {joined} "
                    "in the same configuration layer"
                )
            generation["frames"] = None
            generation["duration"] = None
        prepared["generate"] = generation
    if (
        isinstance(generation, dict)
        and generation.get("frames") is not None
        and generation.get("duration") is not None
    ):
        if not _is_resolved_duration_pair(generation):
            raise ConfigError(
                "generate.frames and generate.duration cannot be combined "
                "in the same configuration layer unless frames are the exact "
                "resolved count for duration"
            )
        prepared = dict(prepared)
        generation = dict(generation)
        generation.pop("duration")
        prepared["generate"] = generation
    if isinstance(generation, dict):
        tiling = generation.get("vae_tiling")
        if isinstance(tiling, dict):
            has_custom_fields = any(name in tiling for name in _VAE_TILING_GEOMETRY_FIELDS)
            mode = tiling.get("mode")
            if has_custom_fields and mode is None:
                prepared = dict(prepared)
                generation = dict(generation)
                generation["vae_tiling"] = {**tiling, "mode": "custom"}
                prepared["generate"] = generation
            elif mode in {"auto", "single"}:
                prepared = dict(prepared)
                generation = _normalize_inactive_vae_tiling(generation)
                prepared["generate"] = generation
    if isinstance(generation, dict) and "frames" in generation and "duration" not in generation:
        # Frames and duration select the same quantity. A higher-precedence
        # explicit frame count must clear a lower-precedence duration.
        prepared = dict(prepared)
        prepared["generate"] = {**generation, "duration": None}
    return prepared


def _normalize_model_selection(config: dict[str, object]) -> dict[str, object]:
    """Normalize independent primary-layout and text-source selections."""
    table = config.get("model_settings")
    if not isinstance(table, dict):
        return config
    schema = config_registry().model("ltx2")
    monolithic_selectors = schema.cli_destinations_in_group(ConfigGroup.MODEL_MONOLITHIC_SOURCE)
    split_selectors = schema.cli_destinations_in_group(ConfigGroup.MODEL_SPLIT_SOURCE)
    model_sources = schema.cli_destinations_in_group(ConfigGroup.MODEL_SOURCE)
    generation_selector = schema.only_cli_destination_in_group(
        ConfigGroup.MODEL_GENERATION_SELECTOR
    )
    video_vae_selector = schema.only_cli_destination_in_group(ConfigGroup.MODEL_VIDEO_VAE_SELECTOR)
    video_vae_source = schema.only_cli_destination_in_group(ConfigGroup.MODEL_VIDEO_VAE_SOURCE)
    gemma_selector = schema.only_cli_destination_in_group(ConfigGroup.MODEL_GEMMA_SOURCE)
    text_encoder_selector = schema.only_cli_destination_in_group(
        ConfigGroup.MODEL_TEXT_ENCODER_SOURCE
    )
    monolithic = any(table.get(name) is not None for name in monolithic_selectors)
    split = any(table.get(name) is not None for name in split_selectors)
    normalized = dict(table)
    if table.get(video_vae_selector) is not None:
        # A named cached-VAE choice replaces a lower-precedence explicit path.
        # An explicit path in this same layer still wins.
        normalized.setdefault(video_vae_source, None)
    if table.get(generation_selector) is not None:
        # A higher-precedence generation choice selects that generation's
        # discoverable cache set. Clear lower-precedence checkpoint paths while
        # preserving any path explicitly supplied beside this selector.
        for name in model_sources:
            normalized.setdefault(name, None)
    if monolithic != split:
        cleared = split_selectors if monolithic else monolithic_selectors
        normalized.update(dict.fromkeys(cleared))
    gemma_selected = table.get(gemma_selector) is not None
    text_selected = table.get(text_encoder_selector) is not None
    if gemma_selected != text_selected:
        normalized[text_encoder_selector if gemma_selected else gemma_selector] = None
    if normalized == table:
        return config
    result = dict(config)
    result["model_settings"] = normalized
    return result


@dataclass(frozen=True)
class Invocation:
    """One fully typed CLI or TOML invocation."""

    model: str
    request: DistilledRequest
    output: OutputConfig
    settings: Settings
    model_settings: LTX2Settings
    model_artifacts: LTX2ArtifactConfig
    resolved_config: dict[str, object]
    options: argparse.Namespace
    restart: RestartManifest | None = None
    generated_output: bool = False


def _cli_layer(options: argparse.Namespace) -> dict[str, object]:
    return config_registry().model("ltx2").cli_layer(options)


def _table(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"[{name}] keys must be strings")
    return cast(Mapping[str, object], value)


def _restart_selection(
    file_config: Mapping[str, object],
    cli_config: Mapping[str, object],
    set_config: Mapping[str, object],
    *,
    source: str,
) -> RestartConfig | None:
    """Resolve restart selection before applying its prior invocation as a base."""
    merged = merge_configs(
        _table(file_config, "restart"),
        _table(cli_config, "restart"),
        _table(set_config, "restart"),
    )
    if not merged:
        return None
    if "run" not in merged:
        raise ConfigError(f"{source}: [restart].run is required")
    try:
        return dataclass_from_mapping(RestartConfig, merged, table="[restart]")
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[restart]: {exc}") from exc


def _validate_settings(settings: Settings) -> None:
    try:
        settings.validate()
    except ValueError as exc:
        raise ConfigError(f"[settings].{exc}") from exc


def _validate_model_settings(settings: LTX2Settings) -> None:
    try:
        settings.validate()
    except ValueError as exc:
        raise ConfigError(f"[model_settings].{exc}") from exc


def _validate_restart_request_changes(
    restart: RestartManifest,
    request: DistilledRequest,
    parent_request: DistilledRequest,
) -> None:
    group = (
        ConfigGroup.RESTART_DECODE_LOCKED
        if restart.config.phase == "decode"
        else ConfigGroup.RESTART_STAGE2_LOCKED
    )
    locked = config_registry().model("ltx2").cli_destinations_in_group(group)
    changed = sorted(
        name for name in locked if getattr(request, name) != getattr(parent_request, name)
    )
    if not changed:
        return
    fields = ", ".join(f"[generate].{name}" for name in changed)
    raise ConfigError(
        f"[restart]: {restart.config.phase} starts after {fields}; "
        "those earlier-station values cannot be changed"
    )


def assemble(
    options: argparse.Namespace,
    *,
    base_settings: Settings | None = None,
    base_model_settings: LTX2Settings | None = None,
) -> Invocation:
    """Resolve defaults < env < TOML < CLI < ``--set`` exactly once."""
    source = str(options.config) if options.config is not None else "invocation"
    file_config = load_invocation_file(options.config) if options.config is not None else {}
    file_config = _prepare_layer(file_config)
    cli_config = _prepare_layer(_cli_layer(options))
    set_config = apply_set_overrides({}, options.set_overrides)
    set_config = _prepare_layer(set_config)
    restart_config = _restart_selection(
        file_config,
        cli_config,
        set_config,
        source=source,
    )
    restart = None if restart_config is None else load_restart_manifest(restart_config)
    restart_base = {} if restart is None else _prepare_layer(restart.base_config())
    merged = merge_configs(
        _output_environment_layer(),
        restart_base,
        file_config,
        cli_config,
        set_config,
    )
    generation = merged.get("generate")
    if isinstance(generation, Mapping):
        # The final policy owns whether geometry is active. Normalizing after
        # precedence merging lets a higher-precedence auto/single selection
        # clear lower-precedence custom geometry and keeps legacy full dumps
        # reloadable.
        merged["generate"] = _normalize_inactive_vae_tiling(generation)
    schema = config_registry().model("ltx2")
    merged = schema.normalize_config(merged)
    validate_top_level(
        merged,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
        source=source,
    )

    model = merged.get("model", "ltx2")
    if not isinstance(model, str):
        raise ConfigError(f"{source}: model must be a string")
    if model not in MODEL_SPECS:
        if model in MODAL_MODEL_SPECS:
            raise ConfigError(
                f"{source}: model {model!r} has its own argument set; "
                "this generation/restart invocation cannot be assembled by the "
                "LTX-2 configuration loader"
            )
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ConfigError(f"{source}: unknown model {model!r}; expected one of {supported}")
    if restart is not None and model != restart.source_model:
        raise ConfigError(
            f"[restart]: source model {restart.source_model!r} cannot restart as {model!r}"
        )

    settings_table = merged.get("settings", {})
    settings_overrides = coerce_dataclass_fields(
        Settings,
        settings_table,
        table="[settings]",
    )
    if base_settings is None:
        try:
            base_settings = Settings.from_env()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment settings: {exc}") from exc
    settings = base_settings.with_overrides(**settings_overrides)
    _validate_settings(settings)

    model_settings_table = merged.get("model_settings", {})
    model_settings_overrides = coerce_dataclass_fields(
        LTX2Settings,
        model_settings_table,
        table="[model_settings]",
    )
    if base_model_settings is None:
        try:
            base_model_settings = LTX2Settings.from_env()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment model settings: {exc}") from exc
    model_sources = schema.cli_destinations_in_group(ConfigGroup.MODEL_SOURCE)
    generation_selector = schema.only_cli_destination_in_group(
        ConfigGroup.MODEL_GENERATION_SELECTOR
    )
    monolithic_selectors = schema.cli_destinations_in_group(ConfigGroup.MODEL_MONOLITHIC_SOURCE)
    split_selectors = schema.cli_destinations_in_group(ConfigGroup.MODEL_SPLIT_SOURCE)
    gemma_selector = schema.only_cli_destination_in_group(ConfigGroup.MODEL_GEMMA_SOURCE)
    text_encoder_selector = schema.only_cli_destination_in_group(
        ConfigGroup.MODEL_TEXT_ENCODER_SOURCE
    )
    if model_settings_overrides.get(generation_selector) is not None:
        # These keyword names are registry-owned dataclass field destinations,
        # but typeshed cannot validate dynamically constructed dataclass kwargs.
        base_model_settings = replace(
            base_model_settings,
            **{  # type: ignore[arg-type]
                name: None for name in model_sources if model_settings_overrides.get(name) is None
            },
        )
    selected_monolithic = any(
        model_settings_overrides.get(name) is not None for name in monolithic_selectors
    )
    selected_split = any(model_settings_overrides.get(name) is not None for name in split_selectors)
    if selected_split and not selected_monolithic:
        # A real LTX-2.5 pack directory remains the source for every component
        # except a higher-precedence transformer override. A monolithic file is
        # still cleared so the established split-primary override keeps working.
        clear_monolithic = {
            name: None
            for name in monolithic_selectors
            if not (
                name == "weights_path"
                and base_model_settings.weights_path is not None
                and base_model_settings.weights_path.expanduser().is_dir()
            )
        }
        base_model_settings = replace(
            base_model_settings,
            **clear_monolithic,  # type: ignore[arg-type]
        )
    elif selected_monolithic and not selected_split:
        base_model_settings = replace(
            base_model_settings,
            **dict.fromkeys(split_selectors),  # type: ignore[arg-type]
        )
    selected_gemma = model_settings_overrides.get(gemma_selector) is not None
    selected_text = model_settings_overrides.get(text_encoder_selector) is not None
    if selected_gemma and not selected_text:
        base_model_settings = replace(
            base_model_settings,
            **{text_encoder_selector: None},  # type: ignore[arg-type]
        )
    elif selected_text and not selected_gemma:
        base_model_settings = replace(
            base_model_settings,
            **{gemma_selector: None},  # type: ignore[arg-type]
        )
    model_settings = base_model_settings.with_overrides(
        **model_settings_overrides
    ).resolve_presets()
    _validate_model_settings(model_settings)

    model_artifacts = dataclass_from_mapping(
        LTX2ArtifactConfig,
        merged.get("model_artifacts", {}),
        table="[model_artifacts]",
    )

    request = dataclass_from_mapping(
        DistilledRequest,
        merged.get("generate", {}),
        table="[generate]",
    )
    if restart is not None:
        parent_request = dataclass_from_mapping(
            DistilledRequest,
            restart_base.get("generate", {}),
            table="[restart source generate]",
        )
        _validate_restart_request_changes(restart, request, parent_request)
    if restart is None or restart.config.phase == "stage-2":
        try:
            request.validate_with_settings(model_settings)
        except ValueError as exc:
            raise ConfigError(f"[generate]: {exc}") from exc
    output = dataclass_from_mapping(
        OutputConfig,
        merged.get("output", {}),
        table="[output]",
    )

    generate_config = _normalize_inactive_vae_tiling(dataclass_to_mapping(request))
    if request.duration is not None:
        # DistilledRequest resolves duration to the exact valid frame count.
        # Persist that authoritative count alone so the machine-written layer
        # cannot conflict with itself when it is reloaded.
        generate_config.pop("duration", None)

    resolved: dict[str, object] = {
        "model": model,
        "generate": generate_config,
        "output": dataclass_to_mapping(output),
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "model_artifacts": dataclass_to_mapping(model_artifacts),
    }
    if request.frames is None and request.duration is None:
        generate = cast(dict[str, object], resolved["generate"])
        generate["auto_duration"] = True
    if restart is not None:
        resolved["restart"] = dataclass_to_mapping(restart.config)
    return Invocation(
        model=model,
        request=request,
        output=output,
        settings=settings,
        model_settings=model_settings,
        model_artifacts=model_artifacts,
        resolved_config=resolved,
        options=options,
        restart=restart,
    )


def validate_for_execution(invocation: Invocation) -> None:
    """Validate fields allowed to stay absent for ``--print-config``."""
    output = invocation.output
    if output.save_hdr_heic_frames and invocation.request.hdr is None:
        raise ConfigError("[output].save_hdr_heic_frames requires HDR generation")
    if invocation.request.hdr is not None:
        if output.vsr_spatial_mode != "off":
            raise ConfigError("[output].vsr_spatial_mode is not yet supported for HDR")
        if output.target_fps is not None and abs(output.target_fps - invocation.request.fps) > 1e-6:
            raise ConfigError("[output].target_fps is not yet supported for HDR")
        if output.save_vae_frames:
            raise ConfigError(
                "[output].save_vae_frames is an SDR PNG diagnostic; "
                "HDR already writes a lossless EXR frame sequence"
            )
        # Save-all requests every applicable sidecar category. Its original-video
        # category is inert when HDR disables the VSR/FRC path, just as its audio
        # category is inert when audio generation is disabled. Keep rejecting a
        # direct request for an HDR original because no HDR-safe postprocessing
        # path currently exists.
        if output.vsr_save_original and not output.save_all_sidecars:
            raise ConfigError("[output].vsr_save_original requires an HDR-safe VSR/FRC path")
    restart = invocation.restart
    if restart is not None and restart.config.phase == "decode":
        if invocation.model_artifacts.save_latents is True:
            raise ConfigError("[model_artifacts].save_latents is not produced by a decode restart")
        if invocation.model_artifacts.save_text_conditioning is True:
            raise ConfigError(
                "[model_artifacts].save_text_conditioning is not produced by a decode restart"
            )
        if invocation.model_artifacts.save_media_conditioning is True:
            raise ConfigError(
                "[model_artifacts].save_media_conditioning is not produced by a decode restart"
            )
        return
    if (
        restart is not None
        and restart.config.phase == "stage-2"
        and invocation.request.text_conditioning is None
    ):
        raise ConfigError(
            "[restart]: stage-2 requires the parent text_conditioning output or --text-conditioning"
        )
    if (
        restart is not None
        and restart.config.phase == "stage-2"
        and invocation.model_artifacts.save_text_conditioning is True
    ):
        raise ConfigError(
            "[model_artifacts].save_text_conditioning is consumed, not produced, "
            "by a stage-2 restart"
        )
    if invocation.model_artifacts.save_media_conditioning is True and (
        invocation.request.image is None and invocation.request.hdr_reference is None
    ):
        raise ConfigError(
            "[model_artifacts].save_media_conditioning requires --image or --hdr-reference"
        )
    try:
        invocation.request.validate_for_generation()
    except ValueError as exc:
        raise ConfigError(f"[generate]: {exc}") from exc


def resolve_for_execution(
    invocation: Invocation,
    *,
    now: datetime | None = None,
) -> Invocation:
    """Validate an invocation and reserve its one authoritative output path."""
    validate_for_execution(invocation)
    if invocation.output.path is not None:
        if (
            invocation.restart is not None
            and invocation.restart.source_video is not None
            and invocation.output.path.expanduser().absolute()
            == invocation.restart.source_video.expanduser().absolute()
        ):
            raise ConfigError("[output].path must not overwrite the restart source video")
        explicit_path = invocation.output.path.expanduser()
        if explicit_path.exists():
            raise ConfigError(
                f"[output].path already exists: {explicit_path}; "
                "choose another path or remove the existing file"
            )
        if invocation.output.save_vae_frames:
            frame_directory = default_vae_frame_directory(invocation.output.path)
            if frame_directory.exists():
                raise ConfigError(
                    f"[output].save_vae_frames would overwrite existing path {frame_directory}"
                )
        if invocation.request.hdr is not None:
            exr_directory = default_hdr_exr_directory(invocation.output.path)
            if exr_directory.exists():
                raise ConfigError(f"[generate].hdr would overwrite existing path {exr_directory}")
            if invocation.output.save_hdr_heic_frames:
                heic_directory = default_hdr_heic_directory(invocation.output.path)
                if heic_directory.exists():
                    raise ConfigError(
                        "[output].save_hdr_heic_frames would overwrite existing path "
                        f"{heic_directory}"
                    )
        return invocation
    base_path = build_timestamped_output_path(
        invocation.output.directory,
        invocation.output.prefix,
        now=now,
    )
    try:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        collision = 1
        while True:
            path = (
                base_path
                if collision == 1
                else base_path.with_name(f"{base_path.stem}_{collision}{base_path.suffix}")
            )
            if invocation.output.save_vae_frames and default_vae_frame_directory(path).exists():
                collision += 1
                continue
            if invocation.request.hdr is not None and default_hdr_exr_directory(path).exists():
                collision += 1
                continue
            if invocation.output.save_hdr_heic_frames and default_hdr_heic_directory(path).exists():
                collision += 1
                continue
            try:
                with path.open("xb"):
                    pass
            except FileExistsError:
                collision += 1
                continue
            break
    except OSError as exc:
        raise ConfigError(f"[output]: cannot reserve generated path {base_path}: {exc}") from exc
    output = replace(
        invocation.output,
        path=path,
    )
    resolved_config = dict(invocation.resolved_config)
    resolved_config["output"] = dataclass_to_mapping(output)
    return replace(
        invocation,
        output=output,
        resolved_config=resolved_config,
        generated_output=True,
    )


__all__ = [
    "Invocation",
    "OutputConfig",
    "assemble",
    "build_timestamped_output_path",
    "resolve_for_execution",
    "sanitize_output_prefix",
    "validate_for_execution",
]
