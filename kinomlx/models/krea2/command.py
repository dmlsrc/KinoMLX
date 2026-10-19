"""Installed ``--model krea2`` preparation and text-to-image command."""

import functools
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from kinomlx.cli.config_output import handle_config_output
from kinomlx.cli.invocation import bootstrap_json_mode
from kinomlx.cli.presentation import (
    log_adapter_selections,
    log_output_paths,
    log_prompt_blocks,
    log_run_header,
    log_section,
    model_display_name,
    render_error,
    render_model_error,
    render_run_success,
)
from kinomlx.cli.resource_operation import report_resources
from kinomlx.config import ConfigError
from kinomlx.debug import (
    RunRecord,
    SidecarError,
    initialize_execution_log,
    sidecar_failure,
    write_effective_config,
)
from kinomlx.errors import KinoMLXError, RunCanceled
from kinomlx.image_restart_config import ImageStage
from kinomlx.io.output_bundle import BundleReservation
from kinomlx.io.output_paths import reserve_output_base
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.output import Krea2OutputSink, plan_krea2_output
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.runner import Krea2Runner
from kinomlx.reporter_stack import command_reporter
from kinomlx.reporting import ScopedReporter, TimingReport, TimingReporter
from kinomlx.samplers.sigmas import render_sigmas
from kinomlx.settings import Settings
from kinomlx.ui import (
    configure_logging,
    configure_logging_from_settings,
)

from .adapters.edit_profile import resolve_edit_adapters
from .adapters.turbo_profile import resolve_turbo_adapters
from .arguments import build_parser
from .catalog import Krea2Variant
from .config import Krea2Invocation, assemble, validate_for_execution
from .output import Krea2OutputPlan
from .pipelines.text_to_image import resolve_request_recipe
from .resource_types import Krea2ComponentResource, Krea2Resources
from .restart import Krea2Restart
from .sampling import Krea2Recipe, timestep_schedule
from .types import Krea2Request

_log = logging.getLogger(__name__)


def _model_name(invocation: Krea2Invocation) -> str:
    return f"{model_display_name('krea2')} {invocation.variant.value.title()}"


def _log_request(invocation: Krea2Invocation) -> None:
    request = invocation.request
    assert request is not None
    tiling: str = request.vae_tiling
    if tiling == "custom":
        tiling += f"; tile={request.vae_tile_size}px; stride={request.vae_tile_stride}px"
    restart_stage = None if invocation.restart is None else invocation.restart.config.stage
    if restart_stage != "decode" and request.turbo_4step and invocation.variant.value == "raw":
        _log.warning(
            "The four-step adapter was trained for Krea Turbo. With Raw, also supply "
            "a standard 8-step Turbo LoRA through --adapter."
        )
    run_facts: list[tuple[str, object]] = [("Seed mode", request.seed_mode)]
    if len(request.selected_seeds()) > 1:
        run_facts.append(("Seeds", ", ".join(map(str, request.selected_seeds()))))
    if invocation.restart is not None:
        run_facts.append(("Source run", invocation.restart.source_run))
        if restart_stage == "decode":
            run_facts.append(("Final latents", invocation.restart.selected_latents))
        else:
            assert request.text_conditioning is not None
            run_facts.append(("Text conditioning", request.text_conditioning))
    log_run_header(
        _log,
        model=_model_name(invocation),
        action=("edit" if request.source_images else "generate")
        if restart_stage is None
        else f"restart {restart_stage}",
        resolution=f"{request.width}x{request.height}",
        prompts=len(request.prompts),
        images=request.image_count,
        seed=request.selected_seeds()[0],
        extra=run_facts,
    )
    if restart_stage is not None and invocation.output.save_all_sidecars:
        _log.info("Save-all sidecars limited to outputs produced by the %s restart", restart_stage)
    compute_facts: list[tuple[str, object]] = []
    if restart_stage != "decode":
        compute_facts.extend(
            (
                ("Denoiser", invocation.model_settings.denoise_mode),
                ("Transformer dtype", invocation.model_settings.transformer_dtype),
            )
        )
    compute_facts.extend(
        (
            ("VAE dtype", invocation.model_settings.vae_dtype),
            ("VAE tiling", tiling),
        )
    )
    log_section(
        _log,
        "Compute",
        compute_facts,
    )
    log_prompt_blocks(_log, request.prompts)
    if request.negative_prompts is not None:
        log_prompt_blocks(_log, request.negative_prompts, label="Negative prompt")
    if request.system_prompt is not None:
        _log.info(
            "  System prompt: %s",
            request.system_prompt or "(empty system message)",
        )
    if request.source_images and restart_stage != "decode":
        log_section(
            _log,
            "Editing",
            (
                ("Fit", request.reference_fit),
                ("Reference strength", request.reference_strength),
                ("Scene strength", request.scene_reference_strength),
                ("Grounding px", request.grounding_px),
                ("Regions", " | ".join(request.reference_boost_regions) or "none"),
            ),
        )
        for index, source in enumerate(request.source_images, 1):
            _log.info("  Source %d: %s", index, source)
        _log.info("  Boost mask: %s", request.reference_boost_mask or "none")
        _log.info(
            "  Grounding system prompt: %s", request.grounding_system_prompt or "training default"
        )
        if invocation.source_sizing is not None:
            _log.info("  Output size: %s", invocation.source_sizing.describe())
        selections, resolution = resolve_edit_adapters(
            invocation.settings.hf_home,
            edit_path=request.edit_adapter_path,
            edit_strength=request.edit_adapter_strength,
            edit_exclusions=request.edit_adapter_exclusions,
            additional=request.resolved_adapters(),
        )
        _log.info(
            "  Edit adapter: %s | Revision: %s",
            resolution["selection"],
            resolution.get("revision", "local file"),
        )
        selections, turbo = resolve_turbo_adapters(
            invocation.settings.hf_home,
            enabled=request.turbo_4step,
            strength=request.turbo_4step_strength,
            exclusions=request.turbo_4step_exclusions,
            additional=selections,
        )
        if turbo is not None:
            _log.info(
                "Four-step Turbo: strength %g | Revision: %s",
                request.turbo_4step_strength,
                turbo["revision"],
            )
        log_adapter_selections(_log, selections)
    elif restart_stage != "decode":
        selections, turbo = resolve_turbo_adapters(
            invocation.settings.hf_home,
            enabled=request.turbo_4step,
            strength=request.turbo_4step_strength,
            exclusions=request.turbo_4step_exclusions,
            additional=request.resolved_adapters(),
        )
        if turbo is not None:
            _log.info(
                "Four-step Turbo: strength %g | Revision: %s",
                request.turbo_4step_strength,
                turbo["revision"],
            )
        log_adapter_selections(_log, selections)


def _mu_label(recipe: Krea2Recipe) -> str:
    """Render the effective shift, marking dynamic and inert cases."""
    if recipe.explicit_sigmas:
        return "unused (explicit sigmas)"
    if recipe.mu is None:
        return f"{recipe.effective_mu:g} (dynamic)"
    return f"{recipe.effective_mu:g}"


def _warn_on_recipe_change(planned: Krea2Recipe, recorded: dict[str, object]) -> None:
    """Warn when the recipe a run recorded differs from the one announced before it."""
    announced = planned.to_metadata()
    changed = [
        f"{key}: announced {announced[key]!r}, recorded {recorded.get(key)!r}"
        for key in ("steps", "guidance", "effective_mu")
        if recorded.get(key) != announced[key]
    ]
    if changed:
        _log.warning("Sampling changed at run time: %s", "; ".join(changed))


def _log_resources(
    resources: Krea2Resources,
    *,
    decoder_only: bool = False,
    text_replay: bool = False,
) -> None:
    source = "independent components" if resources.source_root is None else resources.source_root
    facts: list[tuple[str, object]] = [
        ("Variant", resources.variant.value),
        ("Revision", resources.revision),
        ("Source", source),
    ]
    if not decoder_only:
        facts.extend(
            (
                ("Recipe", resources.recipe_identity),
                ("Denoise mode", resources.denoise_mode),
                ("Distilled", str(resources.capabilities.is_distilled).lower()),
            )
        )
    facts.extend(
        (
            ("VAE decoder", resources.vae_decoder.value),
            ("VAE output scale", f"{resources.vae_decoder.output_scale}x"),
        )
    )
    log_section(
        _log,
        "Resources",
        facts,
    )
    components: tuple[Krea2ComponentResource, ...]
    if decoder_only:
        components = (resources.vae,)
    elif text_replay:
        components = (resources.transformer, resources.vae)
    else:
        components = resources.components
    log_section(
        _log,
        "Components",
        (
            (
                component.kind.value,
                "; ".join(
                    (
                        f"source={component.source_path}",
                        f"layout={component.layout}",
                        f"source_tensors={component.tensor_count}",
                        f"consumed_tensors={component.consumed_tensor_count}",
                    )
                ),
            )
            for component in components
        ),
    )


def _run_generate(
    invocation: Krea2Invocation,
    reporter: TimingReporter,
    *,
    plan: Krea2OutputPlan,  # noqa: ARG001
    reservation: BundleReservation[Krea2OutputPlan],
    output_sink: Krea2OutputSink,
) -> dict[str, object]:
    request = invocation.request
    assert request is not None
    model_reporter = ScopedReporter(reporter, model_display_name("krea2"))
    runner = Krea2Runner(
        invocation.model_settings,
        infrastructure=invocation.settings,
        reporter=model_reporter,
    )
    if runner.resources.variant is not invocation.variant:
        raise RuntimeError(
            "Krea source variant changed after configuration resolution: "
            f"{invocation.variant.value} -> {runner.resources.variant.value}"
        )
    restart_stage = None if invocation.restart is None else invocation.restart.config.stage
    _log_resources(
        runner.resources,
        decoder_only=restart_stage == "decode",
        text_replay=request.text_conditioning is not None,
    )
    planned_recipe = None
    if restart_stage != "decode":
        _, planned_recipe = resolve_request_recipe(runner.resources, request)
        log_section(
            _log,
            "Sampling",
            (
                ("Steps", planned_recipe.steps),
                ("Guidance", f"{planned_recipe.guidance:g}"),
                ("Mu", _mu_label(planned_recipe)),
                ("Noise backend", request.noise_backend),
                ("Sigmas", render_sigmas(timestep_schedule(planned_recipe))),
            ),
        )
    if restart_stage is None:
        generation = runner.generate(
            request,
            on_image=lambda index, image: output_sink.write_image(
                index, image, reservation=reservation
            ),
        )
    elif restart_stage == "decode":
        assert invocation.restart is not None
        generation = runner.restart(
            request,
            Krea2Restart.decode(invocation.restart.selected_latents),
            on_image=lambda index, image: output_sink.write_image(
                index, image, reservation=reservation
            ),
        )
    else:
        assert invocation.restart is not None
        assert request.text_conditioning is not None
        generation = runner.restart(
            request,
            Krea2Restart.denoise(request.text_conditioning),
            on_image=lambda index, image: output_sink.write_image(
                index, image, reservation=reservation
            ),
        )
    with generation as result:
        reporter.memory_checkpoint("generation_ready")
        if planned_recipe is not None:
            recipe = result.metadata["recipe"]
            assert isinstance(recipe, dict)
            _warn_on_recipe_change(planned_recipe, recipe)
        artifacts = output_sink.write(
            result,
            reservation=reservation,
            publish_each=True,
        )
        reporter.memory_checkpoint("output_complete")
        metadata = dict(result.metadata)
    warnings = metadata.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise RuntimeError("Krea generation returned malformed warnings metadata")
    return {
        "action": "restart" if invocation.restart is not None else "generate",
        "outputs": {**artifacts.to_dict(), **output_sink.completed_outputs()}
        if output_sink.completed
        else artifacts.to_dict(),
        "generation": metadata,
        "output_fingerprints": output_sink.fingerprints,
        "output_fingerprint_errors": output_sink.fingerprint_errors,
        "warnings": warnings,
    }


def _planned_generation(request: Krea2Request, variant: Krea2Variant) -> dict[str, object]:
    """Name every planned image before sampling, so a canceled run keeps its identity.

    Prompts, negatives and per-image seeds are all decided by the request, and a
    restart of a canceled run narrows to one image through exactly these rows.
    A completed run replaces this block with its full generation receipt.
    """
    seeds = request.selected_seeds()
    try:
        negatives = request.expanded_negative_prompts()
    except ValueError:
        # The pipeline validates and reports a mismatched negative batch; the
        # record states only what it can state truthfully.
        negatives = None
    return {
        "variant": variant.value,
        "prompts": list(request.expanded_prompts()),
        "negative_prompts": None if negatives is None else list(negatives),
        "noise": {
            "per_image_seeds": [seeds[index % len(seeds)] for index in range(request.image_count)]
        },
    }


def _planned_outputs(plan: Krea2OutputPlan) -> dict[str, object]:
    paths = plan.artifacts.to_dict()
    if len(plan.artifacts.images) > 1:
        paths["seed_outputs"] = [
            plan.seed_artifacts(index).to_dict() for index in range(len(plan.artifacts.images))
        ]
    paths.update({name: str(path) for name, path in plan.sidecar_paths().items()})
    return paths


def _run_generation_command(
    invocation: Krea2Invocation,
    argv: list[str],
) -> tuple[
    dict[str, object],
    TimingReport,
    dict[str, object],
    list[dict[str, str]],
    dict[str, object],
]:
    request = invocation.request
    assert request is not None
    stage = ImageStage.TEXT if invocation.restart is None else invocation.restart.config.stage
    plan = plan_krea2_output(request, invocation.variant, invocation.output, stage=stage)
    command_argv = ["kinomlx", *argv]
    sidecar_errors: list[dict[str, str]] = []
    run_record = None
    payload: dict[str, object]
    timings: TimingReport
    diagnostics: dict[str, object]
    restart_record = (
        None
        if invocation.restart is None
        else invocation.restart.to_record(text_conditioning=request.text_conditioning)
    )

    with ExitStack() as output_stack:

        def reserve(base: Path) -> BundleReservation[Krea2OutputPlan]:
            selected = plan_krea2_output(
                request, invocation.variant, invocation.output, base=base, stage=stage
            )
            return output_stack.enter_context(selected.reserve())

        reservation = reserve_output_base(
            plan.base,
            reserve,
            generated=invocation.output.path is None,
        )
        plan = reservation.plan
        sidecar_paths = plan.sidecar_paths()
        planned_outputs = _planned_outputs(plan)
        effective_config = sidecar_paths.get("effective_config")
        if effective_config is not None:
            try:
                text = config_registry().model("krea2").dump_config(invocation.resolved_config)
                write_effective_config(
                    effective_config,
                    text,
                    replace_existing=plan.force,
                )
            except SidecarError as exc:
                sidecar_errors.append(sidecar_failure("effective_config", effective_config, exc))
                _log.warning("Could not write effective config %s: %s", effective_config, exc)

        execution_log = sidecar_paths.get("execution_log")
        execution_log_ready = False
        if execution_log is not None:
            try:
                initialize_execution_log(execution_log, command_argv)
                execution_log_ready = True
            except SidecarError as exc:
                sidecar_errors.append(sidecar_failure("execution_log", execution_log, exc))
                _log.warning("Could not initialize execution log %s: %s", execution_log, exc)
        try:
            configure_logging_from_settings(
                invocation.settings,
                log_file=(execution_log if execution_log_ready else None),
            )
        except OSError as exc:
            if execution_log is not None:
                sidecar_errors.append(sidecar_failure("execution_log", execution_log, exc))
            configure_logging_from_settings(invocation.settings)
            _log.warning("Could not attach execution log %s: %s", execution_log, exc)

        _log_request(invocation)
        log_output_paths(_log, planned_outputs, heading="Planned outputs")
        with ExitStack() as stack:
            reporter = command_reporter(stack, invocation.settings)
            reporter.memory_checkpoint("run_start")
            run_log = sidecar_paths.get("run_log")
            if run_log is not None:
                try:
                    run_record = RunRecord(
                        run_log,
                        schema=config_registry().model("krea2"),
                        effective_config=invocation.resolved_config,
                        config_intent=invocation.config_intent,
                        argv=command_argv,
                        timings=reporter,
                        planned_outputs=planned_outputs,
                        sidecar_errors=sidecar_errors,
                        restart=restart_record,
                        planned_image_count=plan.image_count,
                        planned_generation=_planned_generation(request, invocation.variant),
                    )
                except SidecarError as exc:
                    sidecar_errors.append(sidecar_failure("run_log", run_log, exc))
                    _log.warning("Could not initialize run log %s: %s", run_log, exc)
            output_sink = Krea2OutputSink(plan, reporter=reporter)
            try:
                payload = _run_generate(
                    invocation,
                    reporter,
                    plan=plan,
                    reservation=reservation,
                    output_sink=output_sink,
                )
                payload_warnings = payload.get("warnings", [])
                if not isinstance(payload_warnings, list) or any(
                    not isinstance(item, str) for item in payload_warnings
                ):
                    raise RuntimeError("Krea command returned malformed warnings metadata")
                outputs = payload["outputs"]
                assert isinstance(outputs, dict)
                failed_sidecars = {error["artifact"] for error in sidecar_errors}
                completed_outputs = dict(outputs)
                completed_outputs.update(
                    {
                        name: str(path)
                        for name, path in sidecar_paths.items()
                        if name not in failed_sidecars and path.is_file()
                    }
                )
                memory = reporter.memory_to_dict()
                diagnostics = {"memory": memory} if memory else {}
                if run_record is not None:
                    try:
                        generation = payload["generation"]
                        assert isinstance(generation, dict)
                        run_record.write(
                            status="completed",
                            outputs=completed_outputs,
                            output_fingerprints=cast(
                                dict[str, str], payload.get("output_fingerprints", {})
                            ),
                            output_fingerprint_errors=cast(
                                dict[str, str], payload.get("output_fingerprint_errors", {})
                            ),
                            generation=generation,
                            diagnostics=diagnostics,
                            warnings=payload_warnings,
                        )
                    except SidecarError as exc:
                        sidecar_errors.append(sidecar_failure("run_log", run_record.path, exc))
                        completed_outputs.pop("run_log", None)
                        _log.warning("Could not finalize run log %s: %s", run_record.path, exc)
                payload["outputs"] = completed_outputs
                timings = reporter.to_dict()
            except BaseException as exc:
                reporter.memory_checkpoint("failure")
                if run_record is not None:
                    failure_memory = reporter.memory_to_dict()
                    failed_outputs: dict[str, object] = {
                        name: str(path) for name, path in sidecar_paths.items() if path.is_file()
                    }
                    failed_outputs.update(output_sink.completed_outputs())
                    try:
                        run_record.write(
                            status=("failed" if isinstance(exc, Exception) else "aborted"),
                            outputs=failed_outputs,
                            output_fingerprints=output_sink.fingerprints,
                            output_fingerprint_errors=output_sink.fingerprint_errors,
                            diagnostics=({"memory": failure_memory} if failure_memory else {}),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    except SidecarError as run_log_exc:
                        sidecar_errors.append(
                            sidecar_failure("run_log", run_record.path, run_log_exc)
                        )
                        _log.warning(
                            "Could not finalize run log %s: %s",
                            run_record.path,
                            run_log_exc,
                        )
                if isinstance(exc, KeyboardInterrupt):
                    raise RunCanceled(output_sink.completed_outputs()) from exc
                raise
    return payload, timings, diagnostics, sidecar_errors, planned_outputs


def run_krea2_command(argv: list[str]) -> int:
    """Resolve, prepare, generate, or restart Krea 2, then return an exit code."""

    configure_logging()
    options = build_parser().parse_args(argv)
    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    json_output = bootstrap_json_mode(options, error_settings)
    try:
        invocation = assemble(options)
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    configure_logging_from_settings(invocation.settings)

    config_exit = handle_config_output(
        options,
        model="krea2",
        intent=invocation.config_intent,
        json_output=invocation.settings.json_output,
    )
    if config_exit is not None:
        return config_exit

    if invocation.operation.dry_run:
        return report_resources(
            "krea2",
            functools.partial(
                prepare_resources, invocation.model_settings, infrastructure=invocation.settings
            ),
            log=_log_resources,
            json_output=invocation.settings.json_output,
        )

    try:
        validate_for_execution(invocation)
    except ConfigError as exc:
        return render_error(
            f"config error: {exc}",
            json_output=invocation.settings.json_output,
        )

    started = time.perf_counter()
    try:
        payload, timings, diagnostics, sidecar_errors, planned_outputs = _run_generation_command(
            invocation, argv
        )
    except (
        KinoMLXError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        return render_model_error(
            "krea2",
            "restart" if invocation.restart is not None else "generation",
            exc,
            json_output=invocation.settings.json_output,
        )

    elapsed = time.perf_counter() - started
    outputs = payload["outputs"]
    warnings = payload["warnings"]
    assert isinstance(outputs, dict)
    assert isinstance(warnings, list)
    generation = payload.get("generation")
    assert generation is None or isinstance(generation, dict)
    extra = {
        name: value
        for name, value in payload.items()
        if name not in {"action", "outputs", "generation", "warnings"}
    }
    render_run_success(
        _log,
        model="krea2",
        action=str(payload["action"]),
        outputs=outputs,
        elapsed_seconds=elapsed,
        timings=timings,
        diagnostics=diagnostics,
        json_output=invocation.settings.json_output,
        generation=generation,
        warnings=warnings,
        sidecar_errors=sidecar_errors,
        extra=extra,
        planned_outputs=planned_outputs,
    )
    return 0


__all__ = ["run_krea2_command"]
