"""Installed LTX-2 generation command and operational error boundary."""

import functools
import logging
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path

from kinomlx.cli.config_output import handle_config_output
from kinomlx.cli.invocation import bootstrap_json_mode
from kinomlx.cli.machine_output import emit_json
from kinomlx.cli.presentation import (
    log_output_paths,
    model_display_name,
    model_failure_message,
    render_error,
    render_model_error,
    render_run_success,
)
from kinomlx.cli.resource_operation import report_resources
from kinomlx.config import ConfigError
from kinomlx.config.persistence import rebase_configuration
from kinomlx.debug import (
    RunRecord,
    SidecarArtifactSink,
    SidecarError,
    SidecarPaths,
    initialize_execution_log,
    sidecar_failure,
    write_effective_config,
)
from kinomlx.errors import KinoMLXError, RunCanceled
from kinomlx.model_registry import config_registry
from kinomlx.output import (
    default_hdr_exr_directory,
    default_hdr_heic_directory,
    default_vae_frame_directory,
)
from kinomlx.reporter_stack import command_reporter
from kinomlx.reporting import Reporter, ScopedReporter, TimingReporter
from kinomlx.settings import Settings
from kinomlx.ui import (
    configure_logging,
    configure_logging_from_settings,
)

from .arguments import build_parser
from .artifacts import (
    requested_artifacts,
    sidecar_paths,
)
from .config import Invocation, assemble, effective_generate_config, resolve_for_execution
from .generation import LTX2GenerationOutput
from .output import write_generation
from .output_config import sidecar_selection
from .pipelines.distilled import DistilledConditioning, generate_ltx2, prepare_distilled_recipe
from .resources import prepare_resources
from .restart import LTX2Restart
from .restart_rules import LTX2Stage
from .run_logging import (
    log_effective_sampling,
    log_request,
    log_resources,
    log_runtime_resources,
    published_stage_samplers,
)
from .runner import LTX2Runner
from .settings import LTX2Settings

_log = logging.getLogger(__name__)


def create_runner(
    *,
    model_settings: LTX2Settings,
    infrastructure: Settings,
    reporter: Reporter,
    artifact_sink: SidecarArtifactSink,
) -> LTX2Runner:
    """Construct the LTX-2 runtime from the resolved command dependencies."""
    return LTX2Runner(
        model_settings=model_settings,
        infrastructure=infrastructure,
        reporter=reporter,
        artifact_sink=artifact_sink,
    )


def _planned_outputs(
    invocation: Invocation,
    paths: SidecarPaths,
    requested_artifacts: frozenset[str],
) -> dict[str, str]:
    planned = {"video": str(paths.video)}
    output = invocation.output
    sidecars = sidecar_selection(output)
    artifact_paths = paths.artifact_paths()
    planned.update({name: str(artifact_paths[name]) for name in sorted(requested_artifacts)})
    planned.update(
        {name: str(path) for name, path in paths.selected_execution_paths(output).items()}
    )
    if sidecars.audio_sidecar and invocation.request.generate_audio:
        planned["audio_waveform"] = str(paths.audio_waveform)
    if sidecars.vae_frames:
        planned["vae_frames"] = str(default_vae_frame_directory(paths.video))
    if invocation.request.hdr is not None:
        planned["exr_frames"] = str(default_hdr_exr_directory(paths.video))
        if sidecars.hdr_heic_frames:
            planned["heic_frames"] = str(default_hdr_heic_directory(paths.video))
    postprocessed = output.vsr_spatial_mode != "off" or (
        output.target_fps is not None and abs(output.target_fps - invocation.request.fps) > 1e-6
    )
    if output.vsr_save_original and postprocessed:
        planned["original_video"] = str(paths.original_video)
    return planned


def _completed_outputs(
    output_path: Path,
    *,
    paths: SidecarPaths,
    artifact_manifest: dict[str, str],
    planned_outputs: dict[str, str],
    sidecar_errors: list[dict[str, str]],
) -> dict[str, str]:
    outputs = {"video": str(output_path), **artifact_manifest}
    failed = {error["artifact"] for error in sidecar_errors}
    for name, path in paths.execution_paths().items():
        if name in planned_outputs and name not in failed and path.is_file():
            outputs[name] = str(path)
    if "audio_waveform" in planned_outputs and paths.audio_waveform.is_file():
        outputs["audio_waveform"] = str(paths.audio_waveform)
    if "original_video" in planned_outputs and paths.original_video.is_file():
        outputs["original_video"] = str(paths.original_video)
    if "vae_frames" in planned_outputs:
        frame_directory = Path(planned_outputs["vae_frames"])
        if frame_directory.is_dir() and (frame_directory / "manifest.json").is_file():
            outputs["vae_frames"] = str(frame_directory)
    if "exr_frames" in planned_outputs:
        exr_directory = Path(planned_outputs["exr_frames"])
        if exr_directory.is_dir() and (exr_directory / "manifest.json").is_file():
            outputs["exr_frames"] = str(exr_directory)
    if "heic_frames" in planned_outputs:
        heic_directory = Path(planned_outputs["heic_frames"])
        if heic_directory.is_dir() and (heic_directory / "manifest.json").is_file():
            outputs["heic_frames"] = str(heic_directory)
    return outputs


def _runtime_diagnostics(
    *,
    reporter: TimingReporter,
    generation: LTX2GenerationOutput | None,
) -> dict[str, object]:
    """Collect allocator and lazy-output diagnostics."""
    diagnostics: dict[str, object] = {}
    memory = reporter.memory_to_dict()
    if memory:
        diagnostics["memory"] = memory
    if generation is not None:
        diagnostics.update(generation.runtime_diagnostics())
    return diagnostics


def _stale_sidecars(
    preexisting: dict[str, str],
    current_outputs: Mapping[str, object],
) -> dict[str, str]:
    """Return previous-run artifacts not replaced by the current manifest."""
    return {name: path for name, path in preexisting.items() if name not in current_outputs}


def _release_unused_generated_output(invocation: Invocation) -> None:
    """Remove the empty exclusive-create reservation after a failed run."""
    if not invocation.generated_output:
        return
    path = invocation.output.path
    if path is None:
        return
    try:
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
    except OSError:
        _log.warning("Could not release unused generated output reservation %s", path)


def run_ltx2_command(arguments: list[str]) -> int:
    """Parse, resolve, and run one LTX-2 argument vector."""

    configure_logging()
    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    options = build_parser().parse_args(arguments)
    bootstrap_json = bootstrap_json_mode(options, error_settings)
    try:
        invocation = assemble(options)
    except ConfigError as exc:
        return render_error(
            f"config error: {exc}",
            json_output=bootstrap_json,
        )

    configure_logging_from_settings(invocation.settings)

    config_exit = handle_config_output(
        options,
        model=invocation.model,
        intent=invocation.config_intent,
        json_output=invocation.settings.json_output,
    )
    if config_exit is not None:
        return config_exit

    if invocation.operation.dry_run:
        return report_resources(
            invocation.model,
            functools.partial(
                prepare_resources, invocation.model_settings, infrastructure=invocation.settings
            ),
            log=functools.partial(log_resources, _log, model_settings=invocation.model_settings),
            json_output=invocation.settings.json_output,
        )

    seeds = invocation.request.selected_seeds()
    if invocation.restart is not None and len(seeds) != 1:
        return render_error(
            "config error: saved-stage restart requires one seed",
            json_output=invocation.settings.json_output,
        )
    planned: list[Invocation] = []
    shared = DistilledConditioning()
    try:
        for index, seed in enumerate(seeds):
            request = replace(invocation.request, seed=seed, seeds=None)
            output = invocation.output
            if len(seeds) > 1 and output.path is not None:
                path = output.path
                output = replace(
                    output,
                    path=path.with_name(f"{path.stem}-{index + 1:03d}-seed{seed}{path.suffix}"),
                )
            resolved = dict(invocation.resolved_config)
            resolved["generate"] = effective_generate_config(request)
            seed_intent = dict(invocation.config_intent.values)
            generate_values = seed_intent.get("generate", {})
            assert isinstance(generate_values, dict)
            generate_intent = dict(generate_values)
            generate_intent.update(seed=seed, seeds=None)
            seed_intent["generate"] = generate_intent
            single = replace(
                invocation,
                request=request,
                output=output,
                resolved_config=resolved,
                config_intent=rebase_configuration(invocation.config_intent, seed_intent),
            )
            planned.append(resolve_for_execution(single))
        # One command emits one JSON record. Multi-seed runs are separate video
        # bundles on disk, but their payloads are collected into one envelope so
        # a caller can still parse the whole stdout stream as a single document.
        collected: list[dict[str, object]] | None = (
            [] if len(planned) > 1 and invocation.settings.json_output else None
        )
        for index, single in enumerate(planned):
            status = _execute_invocation(single, arguments, shared, seeds, index, collected)
            if status:
                _emit_collected(collected, seeds, status=status)
                return status
        _emit_collected(collected, seeds, status=0)
        return 0
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=invocation.settings.json_output)
    finally:
        shared.close()
        for single in planned:
            _release_unused_generated_output(single)


def _failed(
    collected: list[dict[str, object]] | None,
    invocation: Invocation,
    seed_index: int,
    status: int,
    message: str,
) -> int:
    """Record one seed's failure in the collected envelope and keep its status."""
    if collected is not None:
        collected.append(
            {
                "status": "error",
                "error": message,
                "seed": invocation.request.seed,
                "seed_index": seed_index,
            }
        )
    return status


def _emit_collected(
    collected: list[dict[str, object]] | None,
    seeds: tuple[int, ...],
    *,
    status: int,
) -> None:
    """Emit the one aggregate record a multi-seed JSON invocation produces."""
    if collected is None:
        return
    emit_json(
        {
            "status": "ok" if status == 0 else "error",
            "model": "ltx2",
            "seeds": list(seeds),
            "runs": list(collected),
        }
    )


def _execute_invocation(
    invocation: Invocation,
    arguments: list[str],
    shared: DistilledConditioning,
    seeds: tuple[int, ...],
    seed_index: int,
    collected: list[dict[str, object]] | None = None,
) -> int:
    run_started = time.perf_counter()
    # A collecting caller owns the single JSON record for the whole command.
    json_output = invocation.settings.json_output and collected is None
    resolved_output_path = invocation.output.path
    if resolved_output_path is None:
        return _failed(
            collected,
            invocation,
            seed_index,
            render_error("config error: output path was not resolved", json_output=json_output),
            "config error: output path was not resolved",
        )

    restart_record = None
    if invocation.restart is not None:
        try:
            restart_record = invocation.restart.to_record(
                text_conditioning=invocation.request.text_conditioning,
                decoder_seed=invocation.request.seed,
            )
        except ConfigError as exc:
            message = f"config error: {exc}"
            return _failed(
                collected,
                invocation,
                seed_index,
                render_error(message, json_output=json_output),
                message,
            )

    paths = SidecarPaths.for_output(resolved_output_path)
    paths = paths.with_model_artifacts(sidecar_paths(paths.video))
    enabled_artifacts = requested_artifacts(
        invocation.model_artifacts,
        save_all=invocation.output.save_all_sidecars,
        has_media_conditioning=(
            bool(invocation.request.images) or invocation.request.hdr_reference is not None
        ),
        stage=LTX2Stage.TEXT if invocation.restart is None else invocation.restart.config.stage,
        source_latent_stage=(
            None if invocation.restart is None else invocation.restart.selected_latent_stage
        ),
    )
    preexisting_sidecars = {
        name: str(path) for name, path in paths.auxiliary_paths().items() if path.is_file()
    }
    command_argv = ["kinomlx", *arguments]
    planned_outputs = _planned_outputs(invocation, paths, enabled_artifacts)
    sidecars = sidecar_selection(invocation.output)
    sidecar_errors: list[dict[str, str]] = []
    run_record = None
    try:
        effective_config_ready = False
        if sidecars.effective_config:
            try:
                effective_config_text = (
                    config_registry()
                    .model(invocation.model)
                    .dump_config(invocation.resolved_config)
                )
                write_effective_config(paths.effective_config, effective_config_text)
                effective_config_ready = True
            except SidecarError as exc:
                sidecar_errors.append(
                    sidecar_failure("effective_config", paths.effective_config, exc)
                )
                _log.warning(
                    "Could not write effective config %s: %s",
                    paths.effective_config,
                    exc,
                )
        execution_log_ready = False
        if sidecars.console_log:
            try:
                initialize_execution_log(paths.execution_log, command_argv)
                execution_log_ready = True
            except SidecarError as exc:
                sidecar_errors.append(sidecar_failure("execution_log", paths.execution_log, exc))
                _log.warning("Could not initialize execution log %s: %s", paths.execution_log, exc)
        try:
            configure_logging_from_settings(
                invocation.settings,
                log_file=(paths.execution_log if execution_log_ready else None),
            )
        except OSError as exc:
            sidecar_errors.append(sidecar_failure("execution_log", paths.execution_log, exc))
            configure_logging_from_settings(invocation.settings)
            _log.warning("Could not attach execution log %s: %s", paths.execution_log, exc)
        log_request(_log, invocation)
        log_output_paths(_log, planned_outputs, heading="Planned outputs")
        if preexisting_sidecars:
            _log.warning(
                "Existing sidecars for this output stem are previous-run artifacts until replaced: %s",
                ", ".join(sorted(preexisting_sidecars)),
            )
        # Keep phase accounting active in every CLI mode. Quiet disables live
        # rows and its logging level filters summaries; JSON retains a pure
        # stdout protocol while human progress remains on stderr.
        with ExitStack() as reporter_stack:
            reporter = command_reporter(reporter_stack, invocation.settings)
            model_reporter = ScopedReporter(
                reporter,
                model_display_name(invocation.model),
            )
            artifact_sink = SidecarArtifactSink(
                paths.artifact_paths(),
                enabled=enabled_artifacts,
                reporter=model_reporter,
                errors=sidecar_errors,
            )
            if sidecars.run_log:
                try:
                    run_record = RunRecord(
                        paths.run_log,
                        schema=config_registry().model(invocation.model),
                        effective_config=invocation.resolved_config,
                        config_intent=invocation.config_intent,
                        argv=command_argv,
                        timings=reporter,
                        planned_outputs=planned_outputs,
                        sidecar_errors=sidecar_errors,
                        preexisting_sidecars=preexisting_sidecars,
                        restart=restart_record,
                    )
                except SidecarError as exc:
                    sidecar_errors.append(sidecar_failure("run_log", paths.run_log, exc))
                    _log.warning("Could not initialize run log %s: %s", paths.run_log, exc)
            generation: LTX2GenerationOutput | None = None
            generation_warnings: tuple[str, ...] = ()
            reporter.memory_checkpoint("runner_start")
            try:
                runner = create_runner(
                    model_settings=invocation.model_settings,
                    infrastructure=invocation.settings,
                    reporter=model_reporter,
                    artifact_sink=artifact_sink,
                )
                if len(seeds) > 1 and seed_index == 0 and invocation.restart is None:
                    for seed in seeds:
                        prepare_distilled_recipe(
                            replace(invocation.request, seed=seed, seeds=None), runner.resources
                        )
                log_runtime_resources(
                    _log,
                    runner,
                    model_settings=invocation.model_settings,
                    request=invocation.request,
                )
                if invocation.restart is not None:
                    restart = LTX2Restart(
                        stage=invocation.restart.config.stage,
                        latent_stage=invocation.restart.selected_latent_stage,
                        latents=invocation.restart.selected_latents,
                        text_conditioning=(
                            invocation.request.text_conditioning
                            if invocation.restart.config.stage == LTX2Stage.STAGE_2
                            else None
                        ),
                        source_model_generation=invocation.restart.source_model_generation,
                    )
                    generation = runner.restart(invocation.request, restart)
                else:
                    generation = runner.run(
                        partial(generate_ltx2, conditioning=shared), invocation.request
                    )
                reporter.memory_checkpoint("generation_ready")
                generation_metadata = dict(generation.metadata)
                # Same selection receipt LTX2Runner.generate records, so a run
                # log reads identically whichever surface drove the seed.
                generation_metadata.update(
                    seed=invocation.request.seed, seeds=list(seeds), seed_index=seed_index
                )
                raw_generation_warnings = generation_metadata.get("warnings", ())
                if not isinstance(raw_generation_warnings, (list, tuple)) or any(
                    not isinstance(item, str) for item in raw_generation_warnings
                ):
                    raise RuntimeError("LTX-2 generation returned malformed warnings metadata")
                generation_warnings = tuple(raw_generation_warnings)
                log_effective_sampling(
                    _log,
                    generation_metadata,
                    request=invocation.request,
                    published=published_stage_samplers(runner),
                )
                output_path = write_generation(
                    generation,
                    invocation.output,
                    fps=invocation.request.fps,
                    hdr_authoring=invocation.request.hdr,
                    reporter=model_reporter,
                    native_verbose=invocation.settings.verbose,
                )
                reporter.memory_checkpoint("output_complete")
                # The run reached its end, so every artifact it planned should
                # have been handed to the sink by now. One that never arrived
                # was promised in "Planned outputs" and silently dropped, which
                # a later restart would only discover as a missing file. This
                # runs on normal completion alone: a cancellation or an
                # exception leaves by the handler below, where an unwritten
                # artifact is expected rather than a fault.
                unoffered = artifact_sink.unoffered
                if unoffered:
                    plural = len(unoffered) > 1
                    raise KinoMLXError(
                        f"planned {'artifacts' if plural else 'artifact'} "
                        f"{', '.join(unoffered)} "
                        f"{'were' if plural else 'was'} never produced by this run"
                    )
                completed_outputs = _completed_outputs(
                    output_path,
                    paths=paths,
                    artifact_manifest=artifact_sink.manifest,
                    planned_outputs=planned_outputs,
                    sidecar_errors=sidecar_errors,
                )
                diagnostics = _runtime_diagnostics(
                    reporter=reporter,
                    generation=generation,
                )
                timings = reporter.to_dict()
                if run_record is not None:
                    try:
                        run_record.write(
                            status="completed",
                            outputs=completed_outputs,
                            output_fingerprints=artifact_sink.fingerprints,
                            output_fingerprint_errors=artifact_sink.fingerprint_errors,
                            generation=generation_metadata,
                            diagnostics=diagnostics,
                            warnings=generation_warnings,
                            stale_sidecars=_stale_sidecars(
                                preexisting_sidecars,
                                completed_outputs,
                            ),
                        )
                    except SidecarError as exc:
                        sidecar_errors.append(sidecar_failure("run_log", paths.run_log, exc))
                        completed_outputs.pop("run_log", None)
                        _log.warning("Could not finalize run log %s: %s", paths.run_log, exc)
            except BaseException as exc:
                reporter.memory_checkpoint("failure")
                if run_record is not None:
                    partial_outputs = dict(artifact_sink.manifest)
                    partial_outputs["run_log"] = str(paths.run_log)
                    failed_sidecars = {error["artifact"] for error in sidecar_errors}
                    if (
                        execution_log_ready
                        and "execution_log" not in failed_sidecars
                        and paths.execution_log.is_file()
                    ):
                        partial_outputs["execution_log"] = str(paths.execution_log)
                    if effective_config_ready and paths.effective_config.is_file():
                        partial_outputs["effective_config"] = str(paths.effective_config)
                    try:
                        run_record.write(
                            status=("failed" if isinstance(exc, Exception) else "aborted"),
                            outputs=partial_outputs,
                            output_fingerprints=artifact_sink.fingerprints,
                            output_fingerprint_errors=artifact_sink.fingerprint_errors,
                            diagnostics=_runtime_diagnostics(
                                reporter=reporter,
                                generation=generation,
                            ),
                            warnings=generation_warnings,
                            stale_sidecars=_stale_sidecars(
                                preexisting_sidecars,
                                partial_outputs,
                            ),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    except SidecarError as run_log_exc:
                        sidecar_errors.append(
                            sidecar_failure("run_log", paths.run_log, run_log_exc)
                        )
                        _log.warning(
                            "Could not finalize run log %s: %s", paths.run_log, run_log_exc
                        )
                if isinstance(exc, KeyboardInterrupt):
                    raise RunCanceled(partial_outputs) from exc
                raise
    except (
        KinoMLXError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        action = "restart" if invocation.restart is not None else "generation"
        return _failed(
            collected,
            invocation,
            seed_index,
            render_model_error(invocation.model, action, exc, json_output=json_output),
            model_failure_message(invocation.model, action, exc),
        )
    finally:
        _release_unused_generated_output(invocation)

    payload = render_run_success(
        _log,
        model=invocation.model,
        action="restart" if invocation.restart is not None else "generate",
        outputs=completed_outputs,
        elapsed_seconds=time.perf_counter() - run_started,
        timings=timings,
        diagnostics=diagnostics,
        json_output=json_output,
        generation=generation_metadata,
        warnings=generation_warnings,
        sidecar_errors=sidecar_errors,
        planned_outputs=planned_outputs,
    )
    if collected is not None:
        collected.append({**payload, "seed": invocation.request.seed, "seed_index": seed_index})
    return 0


__all__ = ["run_ltx2_command"]
