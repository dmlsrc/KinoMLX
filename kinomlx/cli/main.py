"""Installed CLI entry point and typed operational error boundary."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from kinomlx.config import ConfigError
from kinomlx.errors import KinoMLXError
from kinomlx.output import (
    default_hdr_exr_directory,
    default_hdr_heic_directory,
    default_vae_frame_directory,
)
from kinomlx.settings import Settings

from ._registry import (
    config_registry,
    create_restart_request,
    create_runner,
    resolve_artifact_contribution,
    resolve_modal_model,
    resolve_model_selection,
    resolve_recipe,
)
from .args import build_parser, build_root_parser
from .common import (
    bootstrap_flag_present,
    bootstrap_json_from_arguments,
)
from .common import (
    bootstrap_json_mode as _bootstrap_json_mode,
)
from .common import (
    format_elapsed as _format_elapsed,
)
from .common import (
    render_error as _render_error,
)
from .output import emit_json, write_generation

if TYPE_CHECKING:
    from kinomlx.debug import SidecarPaths
    from kinomlx.reporting import Reporter, TimingReporter

    from ._registry import _RuntimeGeneration
    from .config import Invocation

_log = logging.getLogger(__name__)


class _RuntimeModelSettings(Protocol):
    @property
    def steel_attention_probe(self) -> bool: ...


def _planned_outputs(
    invocation: Invocation,
    paths: SidecarPaths,
    requested_artifacts: frozenset[str],
) -> dict[str, str]:
    planned = {"video": str(paths.video)}
    output = invocation.output
    artifact_paths = paths.artifact_paths()
    planned.update({name: str(artifact_paths[name]) for name in sorted(requested_artifacts)})
    planned.update(
        {name: str(path) for name, path in paths.selected_execution_paths(output).items()}
    )
    if output.save_audio_sidecar and invocation.request.generate_audio:
        planned["audio_waveform"] = str(paths.audio_waveform)
    if output.save_vae_frames:
        planned["vae_frames"] = str(default_vae_frame_directory(paths.video))
    if invocation.request.hdr is not None:
        planned["exr_frames"] = str(default_hdr_exr_directory(paths.video))
        if output.save_hdr_heic_frames:
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
    model_settings: _RuntimeModelSettings,
    *,
    reporter: TimingReporter,
    generation: _RuntimeGeneration | None,
) -> dict[str, object]:
    """Collect allocator, lazy-output, and opt-in kernel diagnostics."""
    diagnostics: dict[str, object] = {}
    memory = reporter.memory_to_dict()
    if memory:
        diagnostics["memory"] = memory
    if generation is not None:
        diagnostics.update(generation.runtime_diagnostics())
    if model_settings.steel_attention_probe:
        from kinomlx.kernels.steel_attention import steel_attention_summary

        diagnostics["steel_attention"] = steel_attention_summary()
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


def main(argv: list[str] | None = None) -> int:
    """Parse, resolve, run, and return a process exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    from kinomlx.ui import configure_logging

    configure_logging()
    if arguments and arguments[0] == "config":
        from .config_init import run_config_command

        return run_config_command(arguments[1:])
    if arguments and arguments[0] == "weights":
        from kinomlx.weights.cli import run_weights_command

        return run_weights_command(arguments[1:])
    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    bootstrap_json = bootstrap_json_from_arguments(arguments, error_settings)
    help_requested = bootstrap_flag_present(arguments, {"-h", "--help"})
    try:
        selection = resolve_model_selection(
            arguments,
            tolerate_errors=help_requested,
        )
    except ConfigError as exc:
        return _render_error(f"config error: {exc}", json_output=bootstrap_json)
    if help_requested and not selection.explicit:
        build_root_parser().print_help()
        return 0
    modal = resolve_modal_model(selection)
    if modal is not None:
        return modal(arguments)

    run_started = time.perf_counter()
    options = build_parser().parse_args(arguments)
    from .config import assemble, resolve_for_execution

    bootstrap_json = _bootstrap_json_mode(options, error_settings)
    try:
        base_settings = Settings.from_env()
    except (TypeError, ValueError) as exc:
        return _render_error(
            f"config error: environment settings: {exc}",
            json_output=bootstrap_json,
        )
    bootstrap_json = _bootstrap_json_mode(options, base_settings)
    try:
        invocation = assemble(options, base_settings=base_settings)
    except ConfigError as exc:
        return _render_error(
            f"config error: {exc}",
            json_output=bootstrap_json,
        )

    from kinomlx.ui import (
        RichReporter,
        configure_logging_from_settings,
    )

    configure_logging_from_settings(invocation.settings)
    from .config_output import handle_config_output

    config_exit = handle_config_output(
        options,
        model=invocation.model,
        resolved=invocation.resolved_config,
        json_output=invocation.settings.json_output,
    )
    if config_exit is not None:
        return config_exit

    try:
        invocation = resolve_for_execution(invocation)
    except ConfigError as exc:
        return _render_error(
            f"config error: {exc}",
            json_output=invocation.settings.json_output,
        )
    resolved_output_path = invocation.output.path
    if resolved_output_path is None:
        return _render_error(
            "config error: output path was not resolved",
            json_output=invocation.settings.json_output,
        )

    restart_record = None
    if invocation.restart is not None:
        try:
            restart_record = invocation.restart.to_record(
                text_conditioning=invocation.request.text_conditioning,
                decoder_seed=invocation.request.seed,
            )
        except ConfigError as exc:
            return _render_error(
                f"config error: {exc}",
                json_output=invocation.settings.json_output,
            )

    from kinomlx.debug import (
        RunRecord,
        SidecarArtifactSink,
        SidecarError,
        SidecarPaths,
        create_mlx_memory_sampler,
        initialize_execution_log,
        sidecar_failure,
        write_effective_config,
    )
    from kinomlx.reporting import TimingReporter

    artifact_contribution = resolve_artifact_contribution(invocation.model)
    paths = SidecarPaths.for_output(resolved_output_path)
    paths = paths.with_model_artifacts(artifact_contribution.sidecar_paths(paths.video))
    requested_artifacts = artifact_contribution.requested_artifacts(
        invocation.model_artifacts,
        save_all=invocation.output.save_all_sidecars,
        has_media_conditioning=(
            invocation.request.image is not None or invocation.request.hdr_reference is not None
        ),
    )
    if invocation.restart is not None:
        requested_artifacts = artifact_contribution.restart_artifacts(
            requested_artifacts,
            phase=invocation.restart.config.phase,
        )
    preexisting_sidecars = {
        name: str(path) for name, path in paths.auxiliary_paths().items() if path.is_file()
    }
    command_argv = list(sys.argv if argv is None else ["kinomlx", *argv])
    planned_outputs = _planned_outputs(invocation, paths, requested_artifacts)
    sidecar_errors: list[dict[str, str]] = []
    run_record = None
    try:
        effective_config_ready = False
        if invocation.output.save_effective_config:
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
        if invocation.output.save_console_log:
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
        if preexisting_sidecars:
            _log.warning(
                "Existing sidecars for this output stem are previous-run artifacts until replaced: %s",
                ", ".join(sorted(preexisting_sidecars)),
            )
        # Keep phase accounting active in every CLI mode. Quiet disables live
        # rows and its logging level filters summaries; JSON retains a pure
        # stdout protocol while human progress remains on stderr.
        with ExitStack() as reporter_stack:
            presentation = reporter_stack.enter_context(
                RichReporter(disable=invocation.settings.quiet)
            )
            presentation_reporter: Reporter = presentation
            if invocation.settings.profile_signposts:
                # Profiling is imported only at the host wiring point. Runtime
                # and model modules remain unaware of Instruments.
                from kinomlx.profiling import SignpostReporter

                presentation_reporter = reporter_stack.enter_context(
                    SignpostReporter(
                        presentation,
                        log_path=invocation.settings.profile_signpost_log,
                        build_dir=invocation.settings.cache_dir / "_native" / "signpost",
                    )
                )
            reporter = TimingReporter(
                presentation_reporter,
                memory_sampler=create_mlx_memory_sampler(),
            )
            artifact_sink = SidecarArtifactSink(
                paths.artifact_paths(),
                enabled=requested_artifacts,
                reporter=reporter,
                errors=sidecar_errors,
            )
            if invocation.output.save_run_log:
                try:
                    run_record = RunRecord(
                        paths.run_log,
                        model=invocation.model,
                        invocation=invocation.resolved_config,
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
            generation: _RuntimeGeneration | None = None
            reporter.memory_checkpoint("runner_start")
            try:
                runner = create_runner(
                    invocation.model,
                    model_settings=invocation.model_settings,
                    infrastructure=invocation.settings,
                    reporter=reporter,
                    artifact_sink=artifact_sink,
                )
                if invocation.restart is not None:
                    restart = create_restart_request(
                        invocation.model,
                        phase=invocation.restart.config.phase,
                        latent_stage=invocation.restart.selected_latent_stage,
                        latents=invocation.restart.selected_latents,
                        text_conditioning=(
                            invocation.request.text_conditioning
                            if invocation.restart.config.phase == "stage-2"
                            else None
                        ),
                        source_model_generation=invocation.restart.source_model_generation,
                    )
                    generation = runner.restart(invocation.request, restart)
                else:
                    generation = runner.run(
                        resolve_recipe(invocation.model),
                        invocation.request,
                    )
                reporter.memory_checkpoint("generation_ready")
                output_path = write_generation(
                    generation,
                    invocation.output,
                    fps=invocation.request.fps,
                    hdr_authoring=invocation.request.hdr,
                    reporter=reporter,
                    native_verbose=invocation.settings.verbose,
                )
                reporter.memory_checkpoint("output_complete")
                if run_record is not None:
                    try:
                        completed_outputs = _completed_outputs(
                            output_path,
                            paths=paths,
                            artifact_manifest=artifact_sink.manifest,
                            planned_outputs=planned_outputs,
                            sidecar_errors=sidecar_errors,
                        )
                        run_record.write(
                            status="completed",
                            outputs=completed_outputs,
                            output_fingerprints=artifact_sink.fingerprints,
                            output_fingerprint_errors=artifact_sink.fingerprint_errors,
                            generation=generation.metadata,
                            diagnostics=_runtime_diagnostics(
                                invocation.model_settings,
                                reporter=reporter,
                                generation=generation,
                            ),
                            stale_sidecars=_stale_sidecars(
                                preexisting_sidecars,
                                completed_outputs,
                            ),
                        )
                    except SidecarError as exc:
                        sidecar_errors.append(sidecar_failure("run_log", paths.run_log, exc))
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
                                invocation.model_settings,
                                reporter=reporter,
                                generation=generation,
                            ),
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
                raise
    except KinoMLXError as exc:
        return _render_error(
            f"generation failed: {exc}",
            json_output=invocation.settings.json_output,
        )
    finally:
        _release_unused_generated_output(invocation)

    if invocation.settings.json_output:
        emit_json(
            {
                "status": "ok",
                "model": invocation.model,
                "output": str(output_path),
            }
        )
    else:
        _log.info("Output: %s", output_path)
        _log.info("Total runtime: %s", _format_elapsed(time.perf_counter() - run_started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
