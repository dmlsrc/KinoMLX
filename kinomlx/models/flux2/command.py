"""Human-facing Klein generation, preparation, and saved-station commands."""

import json
import logging
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from kinomlx._json import JsonObject
from kinomlx.cli.config_output import handle_config_output
from kinomlx.cli.invocation import bootstrap_json_from_arguments
from kinomlx.cli.presentation import (
    log_adapter_selections,
    log_output_paths,
    log_prompt_blocks,
    log_run_header,
    log_section,
    render_error,
    render_model_error,
    render_run_success,
)
from kinomlx.config import ConfigError
from kinomlx.config.persistence import read_configuration
from kinomlx.debug.metadata import (
    RunRecord,
    SidecarError,
    initialize_execution_log,
    sidecar_failure,
    write_effective_config,
)
from kinomlx.errors import RunCanceled
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.image import image_size
from kinomlx.io.output_bundle import BundleReservation, protect_sources
from kinomlx.io.output_paths import reserve_output_base
from kinomlx.io.safetensors import read_metadata
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.modeling.artifacts import WeightSource
from kinomlx.reporter_stack import command_reporter
from kinomlx.samplers.sigmas import render_sigmas
from kinomlx.settings import Settings
from kinomlx.ui import (
    configure_logging,
    configure_logging_from_settings,
    configure_machine_output,
)

from .adapters.loading import prepare_flux2_adapters
from .arguments import build_parser
from .config import Flux2Invocation, assemble
from .config_schema import CONFIG_SCHEMA
from .image_processing import output_size, reference_geometry
from .output import Flux2OutputPlan, Flux2OutputSink, plan_output
from .resources import Flux2Resources, inspect_resources
from .restart import Flux2Restart
from .runner import Flux2Runner
from .types import Flux2Request

_log = logging.getLogger(__name__)


_PHASE_LABELS = {
    "inspect_adapters": "Klein: validate adapters",
    "load_text_encoder": "Klein: bind Qwen text weights",
    "encode_text": "Klein: encode prompt with Qwen3",
    "encode_references": "Klein: encode reference images",
    "load_transformer": "Klein: load transformer",
    "fuse_adapters": "Klein: fuse adapters",
    "denoise": "Klein: denoise",
    "kv_extract": "Klein: extract reference KV",
    "kv_reuse": "Klein: reuse reference KV",
    "load_vae_encode": "Klein: load VAE encoder",
    "load_vae_decode": "Klein: load VAE decoder",
    "decode": "Klein: decode images",
    "write outputs": "Klein: write outputs",
}


def _log_run(invocation: Flux2Invocation, resolution: str) -> None:
    request = invocation.generate
    restart = invocation.restart.selection if invocation.restart else None
    editing = bool(
        request.source_images or (invocation.conditioning and invocation.conditioning.geometries)
    )
    log_run_header(
        _log,
        model=f"FLUX.2 Klein {invocation.model_settings.variant.title()}",
        action=f"restart {restart.phase}" if restart else "edit" if editing else "generate",
        resolution=resolution,
        prompts=len(request.prompts()),
        images=request.image_count,
        seed=request.selected_seeds()[0],
        extra=((("Source run", restart.run),) if restart else ())
        + (("Seed mode", request.seed_mode),)
        + (
            (("Seeds", ", ".join(map(str, request.selected_seeds()))),)
            if len(request.selected_seeds()) > 1
            else ()
        ),
    )
    compute: list[tuple[str, object]] = []
    if restart is None or restart.phase != "decode":
        compute.append(("Transformer dtype", invocation.model_settings.transformer_dtype))
    compute.extend((("VAE dtype", "bfloat16"), ("VAE tiling", "off")))
    log_section(_log, "Compute", compute)


def _log_resources(resources: Flux2Resources) -> None:
    log_section(
        _log,
        "Resources",
        (
            ("Variant", resources.profile.variant.value),
            ("Revision", resources.revision),
            ("Source", resources.source_root or "independent components"),
            ("Recipe", resources.recipe_identity),
            ("Reference KV", str(resources.profile.kv).lower()),
        ),
    )
    log_section(
        _log,
        "Components",
        (
            (
                component.kind,
                f"source={component.source.source_path}; "
                f"layout={component.source.layout.value if isinstance(component.source, WeightSource) else 'tokenizer-json'}; "
                f"source_tensors={len(component.bindings)}; "
                f"consumed_tensors={len(component.active_bindings)}",
            )
            for component in resources.components
        ),
    )


def _log_geometry(invocation: Flux2Invocation) -> tuple[int, int]:
    request = invocation.generate
    station = invocation.conditioning
    if station is not None:
        size = (request.width or station.width, request.height or station.height)
        geometries = station.geometries
    else:
        sizes = tuple(image_size(path) for path in request.source_images)
        size = output_size(
            sizes[0] if sizes else None,
            reference_count=len(sizes),
            width=request.width,
            height=request.height,
            match_source_aspect=request.match_source_aspect,
        )
        geometries = tuple(
            reference_geometry(
                source, size, reference_count=len(sizes), reference_fit=request.reference_fit
            )
            for source in sizes
        )
    _log_run(invocation, f"{size[0]}x{size[1]}")
    if not geometries:
        return size
    log_section(_log, "Editing", (("Target grid", f"{size[0] // 16}x{size[1] // 16}"),))
    _log.info(
        "  Source aspect: %s",
        request.match_source_aspect
        or (
            "implicit single-reference match"
            if len(geometries) == 1 and (request.width is None or request.height is None)
            else "explicit canvas/default"
        ),
    )
    for index, geometry in enumerate(geometries, 1):
        source = request.source_images[index - 1] if request.source_images else "saved conditioning"
        _log.info("  Reference %d: %s", index, source)
        _log.info(
            "    Source %s; %s -> grid %s (%d tokens); crop %s; H/W offset %s; resize %s",
            geometry.source_size,
            geometry.policy,
            geometry.reference_size,
            geometry.token_count,
            geometry.crop_box,
            geometry.offset_hw,
            geometry.resize_kernel,
        )
    tokens = sum(geometry.token_count for geometry in geometries)
    if tokens:
        _log.info("  Total reference tokens: %d", tokens)
        if invocation.model_settings.variant == "kv":
            # 32 blocks, K and V, 4096 hidden channels, two bytes per FP16/BF16 value.
            _log.info(
                "  Reference KV payload: %.1f MiB (payload only, not total peak memory)",
                tokens * 32 * 2 * 4096 * 2 / 1024**2,
            )
    return size


def _sidecar[T](
    errors: list[dict[str, str]], name: str, path: Path, operation: Callable[[], T]
) -> T | None:
    try:
        return operation()
    except (OSError, SidecarError) as exc:
        errors.append(sidecar_failure(name, path, exc))
        _log.warning("Could not write %s sidecar %s: %s", name, path, exc)
        return None


def _source_paths(invocation: Flux2Invocation, arguments: object) -> tuple[Path, ...]:
    generate = invocation.generate
    paths = (
        list(generate.source_images)
        + [adapter.path for adapter in generate.adapters]
        + list(invocation.input_paths)
    )
    for value in (
        generate.text_conditioning,
        getattr(arguments, "config", None),
        getattr(arguments, "prompt_file", None),
    ):
        if value is not None:
            paths.append(Path(value))
    for value in invocation.model_settings.component_overrides.values():
        paths.append(value)
    if invocation.restart:
        paths.append(invocation.restart.selection.run)
        previous = read_configuration(
            invocation.restart.payload, model="flux2", source=str(invocation.restart.selection.run)
        )
        previous_generate = cast(dict[str, object], previous.values.get("generate", {}))
        paths.extend(
            Path(value) for value in cast(list[str], previous_generate.get("source_images", []))
        )

        outputs = invocation.restart.payload.get("outputs")
        if isinstance(outputs, dict):
            for value in outputs.values():
                if isinstance(value, str):
                    paths.append(Path(value))
                elif isinstance(value, list):
                    paths.extend(Path(item) for item in value if isinstance(item, str))
        if invocation.restart.selection.latents:
            paths.append(invocation.restart.selection.latents)
    return tuple(paths)


def _run_images(
    invocation: Flux2Invocation, arguments: list[str], sources: tuple[Path, ...]
) -> int:
    generate, settings = invocation.generate, invocation.settings
    variant = invocation.model_settings.variant
    recipe = generate.recipe(variant)
    restart = invocation.restart
    decode = restart is not None and restart.selection.phase == "decode"
    count = 1 if decode else generate.image_count
    plan = plan_output(
        invocation.output,
        variant=variant,
        seed=generate.selected_seeds()[0],
        image_count=count,
        seeds=None if decode else generate.selected_seeds() * len(generate.prompts()),
        sources=sources,
        decode=decode,
    )
    # Admit all selected stations and adapters before any model preparation or loading.
    prepared = (
        restart.resources(
            variant,
            phase=restart.selection.phase,
            check_adapter_weights=settings.check_adapter_weights,
        )
        if restart
        else None
    )
    if not decode:
        prepare_flux2_adapters(generate.resolved_adapters())
    resources = (
        None if restart else inspect_resources(invocation.model_settings, infrastructure=settings)
    )
    resource_description = (
        resources.describe()
        if resources
        else cast(JsonObject, cast(JsonObject, restart.payload["generation"])["resources"])
        if restart
        else {}
    )
    planned_generation: dict[str, object] = {
        "schema": "kinomlx.flux2.generation.v1",
        "seeds": list(generate.selected_seeds()),
        "seed_mode": generate.seed_mode,
        "resources": resource_description,
        "images": _planned_images(generate, count),
    }
    started = time.perf_counter()
    record = None
    sidecar_errors: list[dict[str, str]] = []
    with ExitStack() as stack:

        def reserve(base: Path) -> tuple[Flux2OutputPlan, BundleReservation[Flux2OutputPlan]]:
            selected = plan_output(
                invocation.output,
                variant=variant,
                seed=generate.selected_seeds()[0],
                image_count=count,
                seeds=None if decode else generate.selected_seeds() * len(generate.prompts()),
                sources=sources,
                base=base,
                decode=decode,
            )
            return selected, stack.enter_context(selected.reserve())

        plan, reservation = reserve_output_base(
            plan.base, reserve, generated=invocation.output.path is None
        )
        sidecars = dict(plan.execution_sidecars)
        console_log = sidecars.get("execution_log")
        if console_log:
            _sidecar(
                sidecar_errors,
                "execution_log",
                console_log,
                lambda: initialize_execution_log(console_log, ["kinomlx", *arguments]),
            )
        try:
            configure_logging_from_settings(
                settings, log_file=console_log if not sidecar_errors else None
            )
        except OSError as exc:
            assert console_log is not None
            sidecar_errors.append(sidecar_failure("execution_log", console_log, exc))
            configure_logging_from_settings(settings)
        stack.callback(configure_logging_from_settings, settings)
        if config_path := sidecars.get("effective_config"):
            _sidecar(
                sidecar_errors,
                "effective_config",
                config_path,
                lambda: write_effective_config(
                    config_path,
                    CONFIG_SCHEMA.dump_config(invocation.resolved_config),
                    replace_existing=plan.force,
                ),
            )

        reporter = command_reporter(
            stack,
            replace(settings, quiet=settings.quiet or settings.json_output),
            phase_labels=_PHASE_LABELS,
        )
        reporter.memory_checkpoint("run_start")
        sink = Flux2OutputSink(plan, reporter=reporter)
        try:
            if record_path := sidecars.get("run_log"):
                record = _sidecar(
                    sidecar_errors,
                    "run_log",
                    record_path,
                    lambda: RunRecord(
                        record_path,
                        model="flux2",
                        effective_config=invocation.resolved_config,
                        config_intent=invocation.config_intent,
                        argv=["kinomlx", *arguments],
                        timings=reporter,
                        planned_outputs=plan.outputs(),
                        restart=asdict(restart.selection) if restart else None,
                        sidecar_errors=sidecar_errors,
                        planned_image_count=plan.image_count,
                        planned_generation=planned_generation,
                    ),
                )
            if decode:
                assert restart is not None
                latent_metadata = read_metadata(
                    restart.artifact("final_latents", restart.selection.latents)
                )
                _log_run(
                    invocation,
                    f"{latent_metadata.get('width', 'unknown')}x{latent_metadata.get('height', 'unknown')}",
                )
                _log.info(
                    "  Final latents: %s",
                    restart.artifact("final_latents", restart.selection.latents),
                )
                if invocation.output.save_all_sidecars:
                    _log.info(
                        "  Save-all includes decode outputs only; initial noise and text conditioning are unavailable"
                    )
                target_size = None
            else:
                target_size = _log_geometry(invocation)
            log_prompt_blocks(_log, generate.prompts())
            if any(generate.negatives()):
                log_prompt_blocks(_log, generate.negatives(), label="Negative prompt")
            if not decode:
                log_adapter_selections(_log, generate.resolved_adapters())
            log_output_paths(_log, plan.outputs(), heading="Planned outputs")
            if resources is not None:
                _log_resources(resources)
            if not decode:
                sampling: list[tuple[str, object]] = [
                    ("Steps", recipe.resolved()[0]),
                    ("Guidance", recipe.resolved()[1]),
                    ("Noise backend", generate.noise_backend),
                ]
                if target_size is not None:
                    tokens = target_size[0] // 16 * (target_size[1] // 16)
                    sampling.append(("Sigmas", render_sigmas(recipe.sigma_array(tokens))))
                log_section(_log, "Sampling", sampling)
            runtime = resources if resources is not None else prepared
            assert runtime is not None
            runtime = replace(runtime, memory_policy=MLXMemoryPolicy.from_settings(settings))
            runner = Flux2Runner(resources=runtime, reporter=reporter)
            if restart:
                selection = (
                    Flux2Restart(
                        latents=restart.artifact("final_latents", restart.selection.latents)
                    )
                    if decode
                    else Flux2Restart(phase="denoise", text_conditioning=generate.text_conditioning)
                )
                output = runner.restart(generate, selection, conditioning=invocation.conditioning)
            else:
                output = runner.generate(
                    generate,
                    conditioning=invocation.conditioning,
                    on_image=lambda index, image: sink.write_image(
                        index, image, recipe, reservation=reservation
                    ),
                )
            results = list(output.images)
            outputs = sink.write(results, recipe, reservation=reservation, publish_each=True)
            identity = _planned_images(generate, len(results))
            generation: dict[str, object] = {
                **planned_generation,
                "images": [
                    {
                        **identity[index],
                        "width": result.width,
                        "height": result.height,
                        "references": [geometry.to_metadata() for geometry in result.references],
                        "adapters": [receipt.to_metadata() for receipt in result.adapter_receipts],
                        "schedule": result.sigmas.tolist() if result.sigmas is not None else None,
                    }
                    for index, result in enumerate(results)
                ],
            }
            reporter.memory_checkpoint("output_complete")
            diagnostics = {"memory": reporter.memory_to_dict()}
            if restart and restart.selection.latents:
                generation["replacement_latents"] = {
                    "path": str(restart.selection.latents),
                    "sha256": file_sha256(restart.selection.latents),
                }
            for failure in sidecar_errors:
                outputs.pop(failure["artifact"], None)
            if record:
                _sidecar(
                    sidecar_errors,
                    "run_log",
                    record.path,
                    lambda: record.write(
                        status="completed",
                        outputs=outputs,
                        output_fingerprints=sink.fingerprints,
                        generation=generation,
                        diagnostics=diagnostics,
                    ),
                )
                if any(error["artifact"] == "run_log" for error in sidecar_errors):
                    outputs.pop("run_log", None)
            render_run_success(
                _log,
                model="flux2",
                action="restart" if restart else "generation",
                outputs=outputs,
                elapsed_seconds=time.perf_counter() - started,
                timings=reporter.to_dict(),
                diagnostics=diagnostics,
                sidecar_errors=sidecar_errors,
                json_output=settings.json_output,
                generation=generation,
                planned_outputs=plan.outputs(),
            )
            return 0
        except BaseException as exc:
            failure_status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
            failure_message = str(exc)
            reporter.memory_checkpoint("failure")
            if record:
                _sidecar(
                    sidecar_errors,
                    "run_log",
                    record.path,
                    lambda: record.write(
                        status=failure_status,
                        error=failure_message,
                        outputs=sink.completed_outputs(),
                        output_fingerprints=sink.fingerprints,
                        diagnostics={"memory": reporter.memory_to_dict()},
                    ),
                )
            if isinstance(exc, KeyboardInterrupt):
                raise RunCanceled(sink.completed_outputs()) from exc
            raise


def _planned_images(generate: Flux2Request, count: int) -> list[dict[str, object]]:
    """Name every planned image before sampling, so a canceled run keeps its identity."""
    seeds = generate.selected_seeds()
    prompts = generate.prompts()
    negatives = generate.negatives()
    return [
        {
            "prompt": prompts[index // len(seeds)],
            "negative_prompt": negatives[index // len(seeds)],
            "seed": seeds[index % len(seeds)],
        }
        for index in range(count)
    ]


def _failure_action(invocation: Flux2Invocation | None) -> str:
    """Name the phase a Klein failure happened in, the way Krea names its own."""
    if invocation is None or invocation.operation.dry_run:
        return "preparation"
    return "restart" if invocation.restart is not None else "generation"


def run_flux2_command(arguments: list[str]) -> int:
    options = build_parser().parse_args(arguments)
    # A configuration error raised by assemble() below is rendered through the
    # product console; without a handler it would reach logging's last resort.
    configure_logging()
    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    json_output = bootstrap_json_from_arguments(arguments, error_settings)
    invocation = None
    try:
        invocation = assemble(options)
        settings = invocation.settings
        json_output = settings.json_output
        configure_logging_from_settings(settings)
        sources = _source_paths(invocation, options)
        if options.save_config:
            protect_sources((options.save_config,), sources)
        config_result = handle_config_output(
            options,
            model="flux2",
            json_output=json_output,
            intent=invocation.config_intent,
        )
        if config_result is not None:
            return config_result
        if not invocation.operation.dry_run:
            return _run_images(invocation, arguments, sources)
        resources = inspect_resources(invocation.model_settings, infrastructure=settings)
        result = resources.describe()
        configure_machine_output("kinomlx.models.flux2.stdout").info(
            "%s", json.dumps(result, indent=2)
        )
        return 0
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    except (ArithmeticError, OSError, ValueError, TypeError, RuntimeError) as exc:
        # Anything that is not a configuration error is a Klein failure, and
        # is named as one instead of reading like a mistake the user made.
        return render_model_error(
            "flux2", _failure_action(invocation), exc, json_output=json_output
        )
