"""Resolve TOML, environment, and CLI input into typed LTX-2 config."""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from kinomlx.cli.seed_config import seed_config
from kinomlx.config import (
    ConfigError,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    validate_top_level,
)
from kinomlx.config.assembly import (
    collect_invocation_sources,
    refuse_restart_changes,
    restart_source,
)
from kinomlx.config.paths import TargetPath
from kinomlx.config.persistence import rebase_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_records import derive_config
from kinomlx.config_rules import validate_capabilities, validate_record
from kinomlx.debug.metadata import SidecarPaths
from kinomlx.io.output_paths import (
    OutputCollisionError,
    reserve_output_base,
    timestamped_output_base,
)
from kinomlx.io.reservation import reservation_path
from kinomlx.model_registry import config_registry
from kinomlx.output import (
    default_hdr_exr_directory,
    default_hdr_heic_directory,
    default_vae_frame_directory,
)
from kinomlx.settings import Settings

from .artifacts import LTX2ArtifactConfig, requested_artifacts, sidecar_paths
from .output_config import OutputConfig, sidecar_selection
from .restart import RestartConfig, RestartManifest, load_restart_manifest
from .settings import LTX2Settings
from .types import DistilledRequest


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
    config_intent: PreparedConfig
    options: argparse.Namespace
    restart: RestartManifest | None = None
    generated_output: bool = False

    @property
    def explicit_config_paths(self) -> frozenset[TargetPath]:
        """Supplied presence comes from retained intent, never resolved-value equality."""
        return self.config_intent.explicit


@dataclass(frozen=True)
class _StationRecords:
    """One side of a restart comparison, built by the single shared record path."""

    settings: Settings
    model_settings: LTX2Settings
    model_artifacts: LTX2ArtifactConfig
    request: DistilledRequest
    output: OutputConfig

    def station_mapping(self, restart: RestartConfig) -> dict[str, object]:
        """Return the six root records one station admission compares."""
        return {
            "settings": self.settings,
            "model_settings": self.model_settings,
            "generate": self.request,
            "output": self.output,
            "model_artifacts": self.model_artifacts,
            "restart": restart,
        }


def _station_records(
    values: Mapping[str, object],
    *,
    base_settings: Settings,
    base_model_settings: LTX2Settings,
    table_prefix: str = "",
    validate_invocation: bool,
    model_settings_overrides: Mapping[str, object] | None = None,
) -> _StationRecords:
    """Build every root record from one normalized mapping, for either restart side.

    Both sides of a restart comparison run through this one path so a field
    policy or a settings preset can never apply to the current record alone.
    Only the current invocation is validated; the parent describes work that
    already ran.

    ``model_settings_overrides`` accepts the already coerced ``[model_settings]``
    table, so an invocation that needed it for source selection coerces it once.
    """
    settings = base_settings.with_overrides(
        **coerce_dataclass_fields(
            Settings,
            values.get("settings", {}),
            table=f"[{table_prefix}settings]",
        )
    )
    if validate_invocation:
        settings.validate()
    if model_settings_overrides is None:
        model_settings_overrides = coerce_dataclass_fields(
            LTX2Settings,
            values.get("model_settings", {}),
            table=f"[{table_prefix}model_settings]",
        )
    model_settings = base_model_settings.with_overrides(
        **model_settings_overrides
    ).resolve_presets()
    if validate_invocation:
        model_settings.validate()
    model_artifacts = dataclass_from_mapping(
        LTX2ArtifactConfig,
        values.get("model_artifacts", {}),
        table=f"[{table_prefix}model_artifacts]",
    )
    request = dataclass_from_mapping(
        DistilledRequest,
        values.get("generate", {}),
        table=f"[{table_prefix}generate]",
    )
    if validate_invocation:
        validate_record(request, location=f"[{table_prefix}generate]")
    output = dataclass_from_mapping(
        OutputConfig,
        values.get("output", {}),
        table=f"[{table_prefix}output]",
    )
    return _StationRecords(settings, model_settings, model_artifacts, request, output)


def _validate_restart_changes(
    restart: RestartManifest,
    current: dict[str, object],
    parent: dict[str, object],
    *,
    active_explicit: frozenset[tuple[str | int, ...]],
) -> None:
    schema = config_registry().model("ltx2")
    refuse_restart_changes(
        schema,
        schema.station_changes(
            current,
            parent,
            station=restart.config.phase,
            active_explicit=active_explicit,
        ),
        station=restart.config.phase,
    )


def assemble(
    options: argparse.Namespace,
    *,
    base_settings: Settings | None = None,
    base_model_settings: LTX2Settings | None = None,
) -> Invocation:
    """Resolve defaults < env < TOML < CLI < ``--set`` exactly once."""
    source = str(options.config) if options.config is not None else "invocation"
    schema = config_registry().model("ltx2")
    sources = collect_invocation_sources(
        schema, options, base_settings=base_settings, base_model_settings=base_model_settings
    )
    restart_config, sources = sources.restart_selection(RestartConfig, source=source)
    restart: RestartManifest | None = None
    restart_base = PreparedConfig({}, {})
    if restart_config is not None:
        with restart_source(restart_config.run):
            restart = load_restart_manifest(restart_config)
            restart_base = schema.normalize_prepared_layer(restart.base_config())
    inputs = sources.prepare(
        rebase_configuration(restart_base, restart_base.values),
        station="generate" if restart is None else restart.config.phase,
    )
    intent = inputs.state()
    merged = schema.normalize_config(intent.values)
    validate_top_level(
        merged,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
        source=source,
    )

    model = merged.get("model", "ltx2")
    if not isinstance(model, str):
        raise ConfigError(f"{source}: model must be a string")
    if model != "ltx2":
        raise ConfigError(f"{source}: model {model!r} does not select LTX-2")
    if restart is not None and model != restart.source_model:
        raise ConfigError(
            f"[restart]: source model {restart.source_model!r} cannot restart as {model!r}"
        )

    if base_settings is None:
        base_settings = Settings()

    model_settings_table = merged.get("model_settings", {})
    model_settings_overrides = coerce_dataclass_fields(
        LTX2Settings,
        model_settings_table,
        table="[model_settings]",
    )
    if base_model_settings is None:
        base_model_settings = LTX2Settings()
    # The declared source transform owns generation, packaging and text-source
    # invalidation. Apply its surviving higher-precedence choices to an API
    # baseline as well; null selections disappear from the merged sparse table
    # before with_overrides can clear that baseline on its own.
    origin_state = inputs.state()
    selected: dict[tuple[str, ...], object] = {}
    for path in inputs.active_explicit_paths():
        if len(path) != 2 or path[0] != "model_settings" or not isinstance(path[1], str):
            continue
        origin = origin_state.origins.get(path)
        if (
            origin is not None
            and origin.location is not None
            and origin.location.kind in {"file", "cli", "override"}
        ):
            selected[("model_settings", path[1])] = model_settings_overrides.get(path[1])
    source_transform = next(
        rule for rule in schema.layer_transforms if rule.name == "ltx_source_selection"
    )
    source_resets = dict(source_transform.run(selected))
    weights_key = ("model_settings", "weights_path")
    if (
        weights_key in source_resets
        and selected.get(("model_settings", "transformer_path")) is not None
        and selected.get(("model_settings", "model_generation")) is None
        and base_model_settings.weights_path is not None
        and base_model_settings.weights_path.expanduser().is_dir()
    ):
        # A pack directory contributes the other component paths after the
        # explicit transformer replaces its member. This one filesystem check
        # is source selection, not a pure configuration transform.
        source_resets.pop(weights_key)
    if source_resets:
        base_model_settings = derive_config(
            base_model_settings,
            **{path[1]: value for path, value in source_resets.items()},
        )
    current = _station_records(
        merged,
        base_settings=base_settings,
        base_model_settings=base_model_settings,
        validate_invocation=True,
        model_settings_overrides=model_settings_overrides,
    )
    settings = current.settings
    model_settings = current.model_settings
    model_artifacts = current.model_artifacts
    request = current.request
    output = current.output
    if restart is not None:
        # The parent run did not itself have to be a restart. Its four restart
        # selectors describe this invocation, not completed parent work, so both
        # sides carry the same selection record.
        parent = _station_records(
            schema.normalize_config(restart_base.values),
            base_settings=base_settings,
            base_model_settings=base_model_settings,
            table_prefix="restart source ",
            validate_invocation=False,
        )
        _validate_restart_changes(
            restart,
            current.station_mapping(restart.config),
            parent.station_mapping(restart.config),
            active_explicit=inputs.active_explicit_paths(),
        )

    resolved: dict[str, object] = {
        "model": model,
        "generate": effective_generate_config(request),
        "output": dataclass_to_mapping(output),
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "model_artifacts": dataclass_to_mapping(model_artifacts),
    }
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
        config_intent=intent,
        options=options,
        restart=restart,
    )


def validate_for_execution(invocation: Invocation) -> None:
    """Validate fields allowed to stay absent for ``--print-config``."""
    output = invocation.output
    validate_capabilities(
        output,
        {"hdr": invocation.request.hdr is not None, "fps": invocation.request.fps},
        location="[output]",
    )
    restart = invocation.restart
    try:
        requested_artifacts(
            invocation.model_artifacts,
            save_all=output.save_all_sidecars,
            has_media_conditioning=invocation.request.image is not None
            or invocation.request.hdr_reference is not None,
            station="generate" if restart is None else restart.config.phase,
            source_latent_stage="final" if restart is None else restart.selected_latent_stage,
        )
    except ValueError as exc:
        raise ConfigError(f"[model_artifacts].{exc}") from exc
    if restart is not None and restart.config.phase == "decode":
        return
    if (
        restart is not None
        and restart.config.phase == "stage-2"
        and invocation.request.text_conditioning is None
    ):
        raise ConfigError(
            "[restart]: stage-2 requires the parent text_conditioning output or --text-conditioning"
        )
    try:
        invocation.request.validate_for_generation()
    except ValueError as exc:
        raise ConfigError(f"[generate]: {exc}") from exc


def effective_generate_config(request: DistilledRequest) -> dict[str, object]:
    """Project one request into the resolved ``generate`` table the run sidecars record.

    Nested records keep their input presence, so an adapter control the user
    never supplied stays absent instead of surfacing as an explicit default.
    A duration request persists its resolved frame count alone, and a request
    that left both unset says so, so the machine-written layer cannot conflict
    with itself when it is reloaded. Every per-seed request is projected here
    too: a second projection with its own rules is how the sidecar once
    described defaults the invocation never held.
    """
    schema = config_registry().model("ltx2")
    generate = cast(
        dict[str, object],
        schema.normalize_effective(
            {"generate": seed_config(dataclass_to_mapping(request, preserve_nested_intent=True))}
        )["generate"],
    )
    if request.duration is not None:
        # DistilledRequest resolves duration to the exact valid frame count.
        generate.pop("duration", None)
    if request.frames is None and request.duration is None:
        generate["auto_duration"] = True
    return generate


def resolve_for_execution(
    invocation: Invocation,
    *,
    now: datetime | None = None,
) -> Invocation:
    """Validate an invocation and reserve its one authoritative output path."""
    validate_for_execution(invocation)
    sidecars = sidecar_selection(invocation.output)
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
        if sidecars.vae_frames:
            frame_directory = default_vae_frame_directory(invocation.output.path)
            if frame_directory.exists():
                raise ConfigError(
                    f"[output].save_vae_frames would overwrite existing path {frame_directory}"
                )
        if invocation.request.hdr is not None:
            exr_directory = default_hdr_exr_directory(invocation.output.path)
            if exr_directory.exists():
                raise ConfigError(f"[generate].hdr would overwrite existing path {exr_directory}")
            if sidecars.hdr_heic_frames:
                heic_directory = default_hdr_heic_directory(invocation.output.path)
                if heic_directory.exists():
                    raise ConfigError(
                        "[output].save_hdr_heic_frames would overwrite existing path "
                        f"{heic_directory}"
                    )
        return invocation

    base = timestamped_output_base(
        invocation.output.directory,
        invocation.output.prefix or f"ltx2-distilled-seed{invocation.request.seed}",
        now=now,
    )

    def reserve(candidate: Path) -> Path:
        path = candidate.with_name(f"{candidate.name}.mp4")
        paths = SidecarPaths.for_output(path).with_model_artifacts(sidecar_paths(path))
        # An earlier failed run may leave sidecars after removing its video.
        for target in paths.auxiliary_paths().values():
            if target.exists() or target.is_symlink() or reservation_path(target).exists():
                raise OutputCollisionError(f"output bundle already contains {target}")
        if sidecars.vae_frames and default_vae_frame_directory(path).exists():
            raise OutputCollisionError(f"VAE frame directory already exists for {path}")
        if invocation.request.hdr is not None and default_hdr_exr_directory(path).exists():
            raise OutputCollisionError(f"HDR EXR directory already exists for {path}")
        if sidecars.hdr_heic_frames and default_hdr_heic_directory(path).exists():
            raise OutputCollisionError(f"HDR HEIC directory already exists for {path}")
        try:
            with path.open("xb"):
                pass
        except FileExistsError as exc:
            raise OutputCollisionError(str(exc)) from exc
        return path

    try:
        base.parent.mkdir(parents=True, exist_ok=True)
        path = reserve_output_base(base, reserve, generated=True)
    except OSError as exc:
        raise ConfigError(f"[output]: cannot reserve generated path {base}: {exc}") from exc
    output = derive_config(
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
    "assemble",
    "effective_generate_config",
    "resolve_for_execution",
    "validate_for_execution",
]
