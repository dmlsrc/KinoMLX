"""Installed ``--model gmnet`` inference command over the public API."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from kinomlx.cli._registry import config_registry, model_choices, validate_model_parser
from kinomlx.cli.common import (
    add_invocation_arguments,
    bootstrap_json_mode,
    format_elapsed,
    render_error,
)
from kinomlx.config import ConfigError
from kinomlx.errors import KinoMLXError
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.settings import Settings, add_argparse_args, add_settings_argparse_args

if TYPE_CHECKING:
    from kinomlx.reporting import Reporter, TimingReporter

    from .config import GMNetInvocation
    from .output import GMNetOutputPlan, GMNetOutputReservation

_log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete GMNet parser without importing MLX."""
    config_schema = config_registry().model("gmnet")
    help_for = config_schema.cli_help
    parser = argparse.ArgumentParser(
        prog="kinomlx",
        description="GMNet SDR-to-HDR still expansion (--model gmnet).",
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=help_for("model"),
    )

    actions = parser.add_argument_group("GMNet input")
    actions.add_argument(
        "--image",
        type=Path,
        default=None,
        help=help_for("image"),
    )
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help=help_for("output_path"),
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=help_for("output_dir"),
    )
    output.add_argument(
        "--output-prefix",
        default=None,
        help=help_for("output_prefix"),
    )
    output.add_argument(
        "--exr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("exr"),
    )
    output.add_argument(
        "--heic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("heic"),
    )
    output.add_argument(
        "--save-gain-map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_gain_map"),
    )
    output.add_argument(
        "--save-run-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_run_log"),
    )
    output.add_argument(
        "--save-console-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_console_log"),
    )
    output.add_argument(
        "--save-effective-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_effective_config"),
    )
    output.add_argument(
        "--save-all-sidecars",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=config_schema.implication_help(("output", "save_all_sidecars")),
    )
    output.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("force"),
    )

    add_argparse_args(
        parser,
        choices_by_field=config_schema.table_choices(("settings",)),
        help_by_field=config_schema.table_help(("settings",)),
        negative_help_by_field=config_schema.table_help(
            ("settings",),
            negative=True,
        ),
    )
    add_settings_argparse_args(
        parser,
        GMNetSettings,
        title="GMNet settings (override env vars)",
        choices_by_field=config_schema.table_choices(("model_settings",)),
        help_by_field=config_schema.table_help(("model_settings",)),
        negative_help_by_field=config_schema.table_help(
            ("model_settings",),
            negative=True,
        ),
    )
    validate_model_parser(parser, "gmnet")
    return parser


def _reporter_stack(stack: ExitStack, settings: Settings) -> TimingReporter:
    from kinomlx.debug import create_mlx_memory_sampler
    from kinomlx.reporting import TimingReporter
    from kinomlx.ui import RichReporter

    presentation = stack.enter_context(RichReporter(disable=settings.quiet))
    presentation_reporter: Reporter = presentation
    if settings.profile_signposts:
        from kinomlx.profiling import SignpostReporter

        presentation_reporter = stack.enter_context(
            SignpostReporter(
                presentation,
                log_path=settings.profile_signpost_log,
                build_dir=settings.cache_dir / "_native" / "signpost",
            )
        )
    return TimingReporter(
        presentation_reporter,
        memory_sampler=create_mlx_memory_sampler(),
    )


def _run_expand(
    invocation: GMNetInvocation,
    reporter: TimingReporter,
    *,
    plan: GMNetOutputPlan | None = None,
    reservation: GMNetOutputReservation | None = None,
) -> dict[str, object]:
    from kinomlx.models.gmnet.output import GMNetOutputSink, plan_gmnet_output
    from kinomlx.models.gmnet.runner import GMNetRunner
    from kinomlx.videotoolbox.heic import PQ_REFERENCE_WHITE_NITS

    request = invocation.request
    assert request is not None
    selected_plan = plan_gmnet_output(request, invocation.output) if plan is None else plan

    def execute(selected_reservation: GMNetOutputReservation) -> dict[str, object]:
        runner = GMNetRunner(
            invocation.model_settings,
            infrastructure=invocation.settings,
            reporter=reporter,
        )
        _log.info(
            "GMNet variant %s; weights %s",
            runner.resources.spec.variant.value,
            runner.resources.weights_path,
        )
        result = runner.expand(request)
        reporter.memory_checkpoint("expansion_ready")
        artifacts = GMNetOutputSink(selected_plan, reporter=reporter).write(
            result,
            reservation=selected_reservation,
        )
        reporter.memory_checkpoint("output_complete")
        _log.info(
            "Predicted Qmax %.3f; peak %.2fx SDR white (model contract about %.0f nits)",
            result.qmax_normalized,
            result.peak_linear,
            result.peak_linear * result.spec.sdr_reference_white_nits,
        )
        payload: dict[str, object] = {
            "action": "expand",
            "outputs": artifacts.to_dict(),
            "variant": result.spec.variant.value,
            "weights_path": str(runner.resources.weights_path),
            "model_sdr_reference_white_nits": result.spec.sdr_reference_white_nits,
            "qmax_normalized": result.qmax_normalized,
            "achieved_peak_over_sdr_white": result.peak_linear,
            "achieved_model_contract_peak_nits": (
                result.peak_linear * result.spec.sdr_reference_white_nits
            ),
        }
        if artifacts.heic is not None:
            delivery_peak_nits = result.peak_linear * PQ_REFERENCE_WHITE_NITS
            _log.info(
                "PQ HEIC delivery uses %.0f-nit reference white; encoded peak about %.0f nits",
                PQ_REFERENCE_WHITE_NITS,
                delivery_peak_nits,
            )
            payload["pq_delivery_reference_white_nits"] = PQ_REFERENCE_WHITE_NITS
            payload["achieved_pq_delivery_peak_nits"] = delivery_peak_nits
        return payload

    if reservation is not None:
        return execute(reservation)
    with selected_plan.reserve() as owned_reservation:
        return execute(owned_reservation)


def run_gmnet_command(argv: list[str]) -> int:
    """Parse, resolve, run, and return a process exit code for GMNet."""
    from kinomlx.ui import configure_logging, configure_logging_from_settings

    configure_logging()
    options = build_parser().parse_args(argv)
    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    json_output = bootstrap_json_mode(options, error_settings)
    try:
        base_settings = Settings.from_env()
    except (TypeError, ValueError) as exc:
        return render_error(
            f"config error: environment settings: {exc}",
            json_output=json_output,
        )

    from .config import assemble, validate_for_execution

    try:
        invocation = assemble(options, base_settings=base_settings)
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    configure_logging_from_settings(invocation.settings)
    from kinomlx.cli.config_output import handle_config_output

    config_exit = handle_config_output(
        options,
        model="gmnet",
        resolved=invocation.resolved_config,
        json_output=invocation.settings.json_output,
    )
    if config_exit is not None:
        return config_exit
    try:
        validate_for_execution(invocation)
    except ConfigError as exc:
        return render_error(
            f"config error: {exc}",
            json_output=invocation.settings.json_output,
        )

    from kinomlx.debug import (
        RunRecord,
        SidecarError,
        initialize_execution_log,
        sidecar_failure,
        write_effective_config,
    )
    from kinomlx.models.gmnet.output import plan_gmnet_output

    assert invocation.request is not None
    plan = plan_gmnet_output(invocation.request, invocation.output)
    sidecar_paths = plan.sidecar_paths()
    planned_outputs = {
        **plan.artifacts.to_dict(),
        **{name: str(path) for name, path in sidecar_paths.items()},
    }
    command_argv = ["kinomlx", *argv]
    sidecar_errors: list[dict[str, str]] = []
    run_record = None
    started = time.perf_counter()
    try:
        with plan.reserve() as reservation:
            effective_config = sidecar_paths.get("effective_config")
            if effective_config is not None:
                try:
                    text = config_registry().model("gmnet").dump_config(invocation.resolved_config)
                    write_effective_config(
                        effective_config,
                        text,
                        replace_existing=plan.force,
                    )
                except SidecarError as exc:
                    sidecar_errors.append(
                        sidecar_failure("effective_config", effective_config, exc)
                    )
                    _log.warning(
                        "Could not write effective config %s: %s",
                        effective_config,
                        exc,
                    )

            execution_log = sidecar_paths.get("execution_log")
            execution_log_ready = False
            if execution_log is not None:
                try:
                    initialize_execution_log(execution_log, command_argv)
                    execution_log_ready = True
                except SidecarError as exc:
                    sidecar_errors.append(sidecar_failure("execution_log", execution_log, exc))
                    _log.warning(
                        "Could not initialize execution log %s: %s",
                        execution_log,
                        exc,
                    )
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

            with ExitStack() as stack:
                reporter = _reporter_stack(stack, invocation.settings)
                reporter.memory_checkpoint("run_start")
                run_log = sidecar_paths.get("run_log")
                if run_log is not None:
                    try:
                        run_record = RunRecord(
                            run_log,
                            model="gmnet",
                            invocation=invocation.resolved_config,
                            argv=command_argv,
                            timings=reporter,
                            planned_outputs=planned_outputs,
                            sidecar_errors=sidecar_errors,
                        )
                    except SidecarError as exc:
                        sidecar_errors.append(sidecar_failure("run_log", run_log, exc))
                        _log.warning("Could not initialize run log %s: %s", run_log, exc)
                try:
                    payload = _run_expand(
                        invocation,
                        reporter,
                        plan=plan,
                        reservation=reservation,
                    )
                    artifact_outputs = payload["outputs"]
                    assert isinstance(artifact_outputs, dict)
                    failed_sidecars = {error["artifact"] for error in sidecar_errors}
                    completed_outputs = dict(artifact_outputs)
                    completed_outputs.update(
                        {
                            name: str(path)
                            for name, path in sidecar_paths.items()
                            if name not in failed_sidecars and path.is_file()
                        }
                    )
                    diagnostics = reporter.memory_to_dict()
                    if run_record is not None:
                        try:
                            run_record.write(
                                status="completed",
                                outputs=completed_outputs,
                                generation={
                                    name: value
                                    for name, value in payload.items()
                                    if name not in {"action", "outputs"}
                                },
                                diagnostics=({"memory": diagnostics} if diagnostics else {}),
                            )
                        except SidecarError as exc:
                            sidecar_errors.append(sidecar_failure("run_log", run_record.path, exc))
                            completed_outputs.pop("run_log", None)
                            _log.warning(
                                "Could not finalize run log %s: %s",
                                run_record.path,
                                exc,
                            )
                    payload["outputs"] = completed_outputs
                    timings = reporter.to_dict()
                except BaseException as exc:
                    reporter.memory_checkpoint("failure")
                    if run_record is not None:
                        failure_memory = reporter.memory_to_dict()
                        failed_outputs = {
                            name: str(path)
                            for name, path in sidecar_paths.items()
                            if path.is_file()
                        }
                        try:
                            run_record.write(
                                status=("failed" if isinstance(exc, Exception) else "aborted"),
                                outputs=failed_outputs,
                                diagnostics=({"memory": failure_memory} if failure_memory else {}),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        except SidecarError as run_log_exc:
                            sidecar_errors.append(
                                sidecar_failure(
                                    "run_log",
                                    run_record.path,
                                    run_log_exc,
                                )
                            )
                            _log.warning(
                                "Could not finalize run log %s: %s",
                                run_record.path,
                                run_log_exc,
                            )
                    raise
    except (KinoMLXError, FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return render_error(
            f"GMNet failed: {exc}",
            json_output=invocation.settings.json_output,
        )

    elapsed = time.perf_counter() - started
    if invocation.settings.json_output:
        from kinomlx.cli.output import emit_json

        emit_json(
            {
                "status": "ok",
                "model": "gmnet",
                **payload,
                "elapsed_seconds": elapsed,
                "timings": timings,
                "diagnostics": {"memory": diagnostics} if diagnostics else {},
            }
        )
    else:
        outputs = payload["outputs"]
        assert isinstance(outputs, dict)
        for name, path in outputs.items():
            _log.info("%s: %s", str(name).upper(), path)
        _log.info("Total runtime: %s", format_elapsed(elapsed))
    return 0


__all__ = ["build_parser", "run_gmnet_command"]
