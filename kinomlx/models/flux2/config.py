"""Layered Klein invocation assembly before model construction."""

import argparse
from dataclasses import dataclass
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
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_records import derive_config
from kinomlx.config_rules import validate_record
from kinomlx.io.image import image_size
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.settings import Settings

from .config_schema import CONFIG_SCHEMA, Flux2Operation
from .image_processing import output_size, reference_geometry
from .restart import RestartManifest
from .settings import Flux2Settings
from .stations import Flux2Conditioning, load_conditioning
from .types import Flux2Request, Flux2RestartConfig


@dataclass(frozen=True)
class Flux2Invocation:
    settings: Settings
    model_settings: Flux2Settings
    operation: Flux2Operation
    generate: Flux2Request
    output: ImageOutputConfig
    resolved_config: dict[str, object]
    config_intent: PreparedConfig
    restart: RestartManifest | None = None
    conditioning: Flux2Conditioning | None = None
    input_paths: tuple[Path, ...] = ()

    @property
    def explicit_config_paths(self) -> frozenset[TargetPath]:
        """Supplied presence comes from retained intent, never resolved-value equality."""
        return self.config_intent.explicit


def assemble(options: argparse.Namespace) -> Flux2Invocation:
    schema = CONFIG_SCHEMA
    source = str(options.config) if options.config is not None else "invocation"
    sources = collect_invocation_sources(schema, options)
    selection, sources = sources.restart_selection(Flux2RestartConfig, source=source)
    restart: RestartManifest | None = None
    base = PreparedConfig({}, {})
    if selection is not None:
        with restart_source(selection.run):
            restart = RestartManifest.load(selection)
            base = restart.base_config()
    inputs = sources.prepare(
        base, station="generate" if restart is None else restart.selection.phase
    )
    input_paths = []
    for layer in (inputs.file, inputs.cli, inputs.override):
        values = layer.raw.get("generate")
        if isinstance(values, dict) and isinstance(values.get("prompt_file"), (str, Path)):
            input_paths.append(Path(values["prompt_file"]))
    intent = inputs.state()
    merged = schema.normalize_config(intent.values)
    validate_top_level(
        merged,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
    )
    if merged.get("model", "flux2") != "flux2":
        raise ConfigError("Klein configuration requires model = flux2")
    settings = Settings().with_overrides(
        **coerce_dataclass_fields(Settings, merged.get("settings", {}), table="[settings]")
    )
    model_settings = Flux2Settings().with_overrides(
        **coerce_dataclass_fields(
            Flux2Settings, merged.get("model_settings", {}), table="[model_settings]"
        )
    )
    generate = dataclass_from_mapping(Flux2Request, merged.get("generate", {}), table="[generate]")
    output = dataclass_from_mapping(ImageOutputConfig, merged.get("output", {}), table="[output]")
    operation = dataclass_from_mapping(
        Flux2Operation, merged.get("operation", {}), table="[operation]"
    )
    settings.validate()
    model_settings.validate()
    validate_record(output, location="[output]")
    if operation.dry_run and (
        generate.prompt is not None
        or generate.prompt_unconditional
        or restart is not None
        or generate.text_conditioning is not None
    ):
        raise ConfigError("Resource-only operations cannot be combined with generation or restart")
    if restart:
        current_roots = {
            "settings": settings,
            "model_settings": model_settings,
            "operation": operation,
            "generate": generate,
            "output": output,
            "restart": restart.selection,
        }
        parent_roots = {
            "settings": dataclass_from_mapping(
                Settings, base.values.get("settings", {}), table="[settings]"
            ),
            "model_settings": dataclass_from_mapping(
                Flux2Settings, base.values.get("model_settings", {}), table="[model_settings]"
            ),
            "operation": dataclass_from_mapping(
                Flux2Operation, base.values.get("operation", {}), table="[operation]"
            ),
            "generate": dataclass_from_mapping(
                Flux2Request, base.values.get("generate", {}), table="[generate]"
            ),
            "output": dataclass_from_mapping(
                ImageOutputConfig, base.values.get("output", {}), table="[output]"
            ),
            # The source run did not necessarily have a restart selector. These
            # selectors choose this invocation and are all explicitly mutable.
            "restart": restart.selection,
        }
        refuse_restart_changes(
            schema,
            schema.station_changes(
                current_roots,
                parent_roots,
                station=restart.selection.phase,
                active_explicit=inputs.active_explicit_paths(),
            ),
            station=restart.selection.phase,
        )
        # A multi-image source run is narrowed to its selected image by
        # RestartManifest.base_config, so anything still plural here is a
        # request the restart cannot satisfy.
        if len(generate.prompts()) != 1 or len(generate.selected_seeds()) != 1:
            raise ConfigError(
                "Saved-station restart resumes one image; select it with [restart].index "
                "and do not add prompts or seeds"
            )
    if restart is None:
        validate_record(generate, location="[generate]")
    conditioning = None
    if (restart is None and generate.text_conditioning is not None) or (
        restart is not None and restart.selection.phase == "denoise"
    ):
        artifact = generate.text_conditioning
        if restart:
            # Replacements are validated against saved prompt context without reopening the original.
            original = cast(dict[str, object], base.values.get("generate", {})).get(
                "text_conditioning"
            )
            replacement = (
                artifact if artifact is not None and str(artifact) != str(original) else None
            )
            artifact = restart.artifact("text_conditioning", replacement)
        assert artifact is not None
        conditioning = load_conditioning(artifact, generate.recipe(model_settings.variant))
        prompts = generate.prompts()
        if prompts and prompts != (conditioning.prompt,):
            raise ConfigError("Saved conditioning prompt does not match the generation request")
        if generate.negative_prompt is not None and derive_config(
            generate,
            prompt=conditioning.prompt,
            prompt_unconditional=generate.prompt_unconditional or conditioning.prompt == "",
        ).negatives() != (conditioning.negative_prompt,):
            raise ConfigError("Saved negative conditioning does not match the generation request")
        generate = derive_config(
            generate,
            prompt=conditioning.prompt,
            prompt_unconditional=generate.prompt_unconditional or conditioning.prompt == "",
            negative_prompt=conditioning.negative_prompt,
            text_conditioning=artifact,
            source_images=(),
            reference_fit=None,
            match_source_aspect=None,
            width=conditioning.width if generate.width is None else generate.width,
            height=conditioning.height if generate.height is None else generate.height,
        )
        if conditioning.geometries and (generate.width, generate.height) != (
            conditioning.width,
            conditioning.height,
        ):
            raise ConfigError("Edit replay must preserve the saved output grid")
    generate.validate(model_settings.variant)
    if not operation.dry_run and not (restart and restart.selection.phase == "decode"):
        if not generate.prompts() and not (options.print_config or options.save_config):
            raise ConfigError(
                "[generate].prompt, [generate].prompt_file, [generate].text_conditioning, "
                "or [restart].run is required"
            )
        if conditioning is None:
            sizes = tuple(image_size(path) for path in generate.source_images)
            width, height = output_size(
                sizes[0] if sizes else None,
                reference_count=len(sizes),
                width=generate.width,
                height=generate.height,
                match_source_aspect=generate.match_source_aspect,
            )
            for size in sizes:
                reference_geometry(
                    size,
                    (width, height),
                    reference_count=len(sizes),
                    reference_fit=generate.reference_fit,
                )
    resolved: dict[str, object] = {
        "model": "flux2",
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "operation": dataclass_to_mapping(operation),
        "generate": seed_config(dataclass_to_mapping(generate, preserve_nested_intent=True)),
        "output": dataclass_to_mapping(output),
    }
    if restart:
        resolved["restart"] = dataclass_to_mapping(restart.selection)
    return Flux2Invocation(
        settings,
        model_settings,
        operation,
        generate,
        output,
        resolved,
        intent,
        restart,
        conditioning,
        tuple(input_paths),
    )
