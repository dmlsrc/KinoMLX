"""Resolve TOML, environment, and CLI input into Krea 2 settings."""

import argparse
from dataclasses import dataclass

from kinomlx.cli.resource_operation import Operation, refuse_resource_only_work
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
    config_table,
    refuse_restart_changes,
    restart_source,
)
from kinomlx.config.paths import TargetPath
from kinomlx.config.persistence import rebase_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_records import construct_record, derive_config
from kinomlx.config_rules import validate_record
from kinomlx.image_restart_config import ImageStage
from kinomlx.io.output_bundle import BundleOutputError
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.model_registry import config_registry
from kinomlx.prompts import is_empty_prompt_batch
from kinomlx.settings import Settings

from .adapters.profiles import resolve_profile_adapters
from .adapters.turbo_profile import TURBO_4STEP_RECIPE
from .catalog import DEFAULT_VARIANT, Krea2Variant, variant_from_distilled, variant_spec
from .edit_validation import validate_edit_request
from .metadata import parse_model_index
from .output import plan_krea2_output
from .request_sources import resolve_request_sources
from .restart import RestartConfig, RestartManifest, load_restart_manifest
from .settings import Krea2Settings
from .sizing import SourceSizing, resolve_source_sizing
from .types import Krea2GenerateConfig, Krea2Request


@dataclass(frozen=True)
class Krea2Invocation:
    """One fully resolved Krea invocation before output reservation."""

    settings: Settings
    model_settings: Krea2Settings
    operation: Operation
    variant: Krea2Variant
    generate: Krea2GenerateConfig
    output: ImageOutputConfig
    request: Krea2Request | None
    resolved_config: dict[str, object]
    config_intent: PreparedConfig
    restart: RestartManifest | None = None
    source_sizing: SourceSizing | None = None

    @property
    def explicit_config_paths(self) -> frozenset[TargetPath]:
        """Supplied presence comes from retained intent, never resolved-value equality."""
        return self.config_intent.explicit


def _configured_variant(settings: Krea2Settings) -> Krea2Variant:
    if settings.variant is not None:
        return Krea2Variant(settings.variant)
    if settings.source_path is not None:
        index = settings.source_path.expanduser() / "model_index.json"
        if index.is_file():
            try:
                return variant_from_distilled(parse_model_index(index).is_distilled)
            except (OSError, TypeError, ValueError) as exc:
                raise ConfigError(f"[model_settings].source_path: {exc}") from exc
    return DEFAULT_VARIANT


def _effective_generate(
    generate: Krea2GenerateConfig,
    variant: Krea2Variant,
) -> Krea2GenerateConfig:
    spec = variant_spec(variant)
    return derive_config(
        generate,
        steps=(int(TURBO_4STEP_RECIPE["steps"]) if generate.turbo_4step else spec.default_steps)
        if generate.steps is None
        else generate.steps,
        guidance=(
            (TURBO_4STEP_RECIPE["guidance"] if generate.turbo_4step else spec.default_guidance)
            if generate.guidance is None
            else generate.guidance
        ),
        mu=(TURBO_4STEP_RECIPE["mu"] if generate.turbo_4step else spec.default_mu)
        if generate.mu is None
        else generate.mu,
    )


def _rebase_inherited_defaults(
    intent: PreparedConfig,
    parent: PreparedConfig,
    active_explicit: frozenset[TargetPath],
) -> PreparedConfig:
    """Forget only defaults whose source changed in this surviving invocation."""
    values = dict(intent.values)
    model = dict(config_table(values, "model_settings"))
    parent_model = config_table(parent.values, "model_settings")
    if (
        ("model_settings", "vae_path") in active_explicit
        and model.get("vae_path") != parent_model.get("vae_path")
        and ("model_settings", "vae_decoder") not in active_explicit
    ):
        model.pop("vae_decoder", None)
        values["model_settings"] = model

    generate = dict(config_table(values, "generate"))
    parent_generate = config_table(parent.values, "generate")
    if ("generate", "turbo_4step") in active_explicit and generate.get(
        "turbo_4step", False
    ) != parent_generate.get("turbo_4step", False):
        for name in TURBO_4STEP_RECIPE:
            path = ("generate", name)
            if path not in active_explicit and path not in parent.explicit:
                generate.pop(name, None)
        if generate.get("turbo_4step") is False:
            for name in (
                "turbo_4step_strength",
                "turbo_4step_exclusions",
                "turbo_4step_step_strengths",
                "turbo_4step_sigma_window",
            ):
                if ("generate", name) not in active_explicit:
                    generate.pop(name, None)
        values["generate"] = generate
    return rebase_configuration(intent, values)


def assemble(
    options: argparse.Namespace,
    *,
    base_settings: Settings | None = None,
    base_model_settings: Krea2Settings | None = None,
) -> Krea2Invocation:
    """Resolve defaults, environment, TOML, CLI, and typed set overrides."""
    source = str(options.config) if options.config is not None else "invocation"
    schema = config_registry().model("krea2")
    sources = collect_invocation_sources(
        schema, options, base_settings=base_settings, base_model_settings=base_model_settings
    )
    restart_config, sources = sources.restart_selection(RestartConfig, source=source)
    restart: RestartManifest | None = None
    restart_base = PreparedConfig({}, {})
    if restart_config is not None:
        with restart_source(restart_config.run):
            restart = load_restart_manifest(restart_config, schema=schema)
            restart_base = schema.normalize_prepared_layer(restart.base_config())
    inputs = sources.prepare(
        rebase_configuration(restart_base, restart_base.values),
        stage=ImageStage.TEXT if restart is None else restart.config.stage,
    )
    cli_layer, set_layer = inputs.cli.values, inputs.override.values
    explicit = inputs.supplied_values()
    intent = inputs.state()
    if restart is not None:
        intent = _rebase_inherited_defaults(intent, restart_base, inputs.active_explicit_paths())
    merged = schema.normalize_config(intent.values)
    validate_top_level(
        merged,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
        source=source,
    )
    model = merged.get("model", "krea2")
    if model != "krea2":
        raise ConfigError(f"{source}: model {model!r} does not select Krea 2")
    if restart is not None and restart.source_model != "krea2":
        raise ConfigError(
            f"[restart]: source model {restart.source_model!r} cannot restart as 'krea2'"
        )

    settings_overrides = coerce_dataclass_fields(
        Settings,
        merged.get("settings", {}),
        table="[settings]",
    )
    if base_settings is None:
        base_settings = Settings()
    settings = base_settings.with_overrides(**settings_overrides)
    settings.validate()

    model_overrides = coerce_dataclass_fields(
        Krea2Settings,
        merged.get("model_settings", {}),
        table="[model_settings]",
    )
    if base_model_settings is None:
        base_model_settings = Krea2Settings()
    model_settings = base_model_settings.with_overrides(**model_overrides)
    model_settings.validate()

    variant = _configured_variant(model_settings)
    if (
        restart is not None
        and restart.source_variant is not None
        and restart.source_variant != variant.value
    ):
        raise ConfigError(
            f"[restart]: source variant {restart.source_variant!r} cannot restart as "
            f"{variant.value!r}"
        )
    operation = dataclass_from_mapping(
        Operation,
        merged.get("operation", {}),
        table="[operation]",
    )
    generate = dataclass_from_mapping(
        Krea2GenerateConfig,
        merged.get("generate", {}),
        table="[generate]",
    )
    generate = _effective_generate(generate, variant)
    refuse_resource_only_work(
        dry_run=operation.dry_run,
        requested=(
            generate.prompt is not None
            or generate.prompt_unconditional
            or restart is not None
            or generate.text_conditioning is not None
        ),
        work="generation or restart",
    )
    output_values = coerce_dataclass_fields(
        ImageOutputConfig, merged.get("output", {}), table="[output]"
    )
    output = construct_record(
        ImageOutputConfig,
        output_values,
        frozenset(
            path[1]
            for path in intent.explicit
            if len(path) == 2
            and path[0] == "output"
            and isinstance(path[1], str)
            and path[1] in output_values
        ),
    )
    validate_record(output, location="[output]")
    try:
        validate_record(generate, location="[generate]")
        request = generate.request()
        if request is not None:
            request = resolve_request_sources(request)
    except OSError as exc:
        raise ConfigError(f"[generate].text_conditioning: {exc}") from exc
    except ConfigError:
        # A rule violation already names the table it was written at.
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[generate]: {exc}") from exc
    if (
        request is not None
        and (request.turbo_4step or request.turbo_sda)
        and (restart is None or restart.config.stage != "decode")
    ):
        try:
            _, selected_turbo, _ = resolve_profile_adapters(
                settings.hf_home,
                four_step=request.four_step_profile(),
                sda=request.sda_profile(),
                additional=request.resolved_adapters(),
            )
        except (OSError, ValueError) as exc:
            raise ConfigError(f"[generate].adapter profile: {exc}") from exc
        recorded_turbo = None if restart is None else restart.generation.get("turbo_4step")
        # Strength and exclusions are denoise inputs; only pin the release itself.
        if (
            isinstance(recorded_turbo, dict)
            and selected_turbo is not None
            and any(
                selected_turbo.get(key) != recorded_turbo.get(key)
                for key in ("repo_id", "revision", "filename")
            )
        ):
            raise ConfigError("Four-step Turbo adapter release differs from the saved run")
    source_sizing = None
    if request is not None and restart is None and request.match_source_aspect is not None:
        command_line = {*config_table(cli_layer, "generate"), *config_table(set_layer, "generate")}
        cli_sizes = frozenset(name for name in ("width", "height") if name in command_line)
        try:
            request, source_sizing = resolve_source_sizing(request, cli_sizes=cli_sizes)
        except FileNotFoundError as exc:
            raise ConfigError(f"[generate].source_images: {exc}") from exc
        except ValueError as exc:
            raise ConfigError(f"[generate].match_source_aspect: {exc}") from exc
        generate = derive_config(generate, width=request.width, height=request.height)
    if restart is not None:
        parent_values = dict(config_table(restart_base.values, "generate"))
        if (
            restart.config.stage == "denoise"
            and "text_conditioning" in config_table(explicit, "generate")
            and "prompts" in restart.generation
            and "negative_prompts" in restart.generation
        ):
            # Compare against recorded context without reopening an unused input.
            # Keep invocation batch expansion and explicit prompt provenance.
            parent_values["text_conditioning"] = None
            if parent_values.get("prompt") is None:
                parent_values["prompt"] = restart.generation["prompts"]
                if is_empty_prompt_batch(restart.generation["prompts"]):
                    parent_values["prompt_unconditional"] = True
            if parent_values.get("negative_prompt") is None:
                parent_values["negative_prompt"] = restart.generation["negative_prompts"]
                if (
                    parent_values["negative_prompt"] is None
                    and parent_values.get("guidance", variant_spec(variant).default_guidance) == 0
                    and request is not None
                ):
                    # A zero-guidance run does not record unused negative rows.
                    parent_values["negative_prompt"] = request.negative_prompts
        parent_generate = dataclass_from_mapping(
            Krea2GenerateConfig,
            parent_values,
            table="[restart source generate]",
        )
        parent_generate = _effective_generate(parent_generate, variant)
        try:
            validate_record(parent_generate, location="[restart source generate]")
            parent_request = parent_generate.request()
            if parent_request is not None:
                parent_request = resolve_request_sources(
                    parent_request, location="[restart source generate]"
                )
        except OSError as exc:
            raise ConfigError(f"[restart source generate].text_conditioning: {exc}") from exc
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[restart source generate]: {exc}") from exc
        if request is None or parent_request is None:
            raise ConfigError("[restart]: source invocation has no Krea generation request")
        parent_settings = dataclass_from_mapping(
            Settings,
            config_table(restart_base.values, "settings"),
            table="[restart source settings]",
        )
        parent_model_settings = dataclass_from_mapping(
            Krea2Settings,
            config_table(restart_base.values, "model_settings"),
            table="[restart source model_settings]",
        )
        parent_output = dataclass_from_mapping(
            ImageOutputConfig,
            config_table(restart_base.values, "output"),
            table="[restart source output]",
        )
        parent_operation = dataclass_from_mapping(
            Operation,
            config_table(restart_base.values, "operation"),
            table="[restart source operation]",
        )
        refuse_restart_changes(
            schema,
            schema.stage_changes(
                {
                    "settings": settings,
                    "model_settings": model_settings,
                    "operation": operation,
                    "generate": request,
                    "output": output,
                    "restart": restart.config,
                },
                {
                    "settings": parent_settings,
                    "model_settings": parent_model_settings,
                    "operation": parent_operation,
                    "generate": parent_request,
                    "output": parent_output,
                    "restart": restart.config,
                },
                stage=restart.config.stage,
                active_explicit=inputs.active_explicit_paths(),
            ),
            stage=restart.config.stage,
        )

        if (
            restart.config.stage == "denoise"
            and request.text_conditioning is not None
            and "negative_prompt" not in config_table(explicit, "generate")
        ):
            # The saved stage owns inherited negative rows. Preserve only a
            # negative supplied in this invocation as an explicit user request.
            generate = derive_config(generate, negative_prompt=None)
            request = resolve_request_sources(derive_config(request, negative_prompt=None))

    resolved: dict[str, object] = {
        "model": "krea2",
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "operation": dataclass_to_mapping(operation),
        "output": dataclass_to_mapping(output),
    }
    if request is not None or "generate" in explicit:
        resolved["generate"] = seed_config(
            dataclass_to_mapping(generate, preserve_nested_intent=True)
        )
    if restart is not None:
        resolved["restart"] = dataclass_to_mapping(restart.config)
    return Krea2Invocation(
        settings=settings,
        model_settings=model_settings,
        operation=operation,
        variant=variant,
        generate=generate,
        output=output,
        request=request,
        resolved_config=resolved,
        config_intent=intent,
        restart=restart,
        source_sizing=source_sizing,
    )


def validate_for_execution(invocation: Krea2Invocation) -> None:
    """Validate fields allowed to remain absent for a dry run or config output."""
    if invocation.request is None:
        raise ConfigError(
            "Krea 2 generation needs --prompt or --text-conditioning; "
            "use --dry-run to inspect resources without generating"
        )
    validate_record(invocation.request, location="[generate]")
    if invocation.request.source_images:
        validate_edit_request(invocation.request)
    if invocation.restart is not None:
        config_registry().model("krea2").validate_stage_outputs(
            {"output": dataclass_to_mapping(invocation.output)},
            stage=invocation.restart.config.stage,
        )
        if (
            invocation.restart.config.stage == "denoise"
            and invocation.request.text_conditioning is None
        ):
            raise ConfigError(
                "[restart]: denoise requires the parent text_conditioning output or "
                "--text-conditioning"
            )
    try:
        plan = plan_krea2_output(
            invocation.request,
            invocation.variant,
            invocation.output,
            stage=ImageStage.TEXT
            if invocation.restart is None
            else invocation.restart.config.stage,
        )
    except (ValueError, BundleOutputError) as exc:
        raise ConfigError(f"[output]: {exc}") from exc
    text_input = invocation.request.text_conditioning
    text_outputs = plan.artifacts.text_conditioning
    if text_input is not None and any(
        text_input.expanduser().absolute() == path.expanduser().absolute() for path in text_outputs
    ):
        raise ConfigError("[output] must not overwrite the consumed text-conditioning artifact")
    if invocation.restart is not None:
        source_outputs = {
            invocation.restart.source_run,
            *invocation.restart.source_outputs,
        }
        if invocation.restart.config.stage == "decode":
            source_outputs.add(invocation.restart.selected_latents)
        elif text_input is not None:
            source_outputs.add(text_input.expanduser().absolute())
        collisions = sorted(
            str(path) for path in plan.targets() if path.expanduser().absolute() in source_outputs
        )
        if collisions:
            raise ConfigError(
                "[output] must not overwrite restart source artifacts: " + ", ".join(collisions)
            )


__all__ = [
    "Krea2Invocation",
    "assemble",
    "validate_for_execution",
]
