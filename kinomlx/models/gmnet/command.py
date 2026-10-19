"""Installed ``--model gmnet`` inference command over the public API."""

import functools
import logging
import time
from contextlib import ExitStack
from pathlib import Path

from kinomlx.cli.config_output import handle_config_output
from kinomlx.cli.invocation import bootstrap_json_mode
from kinomlx.cli.presentation import (
    log_output_paths,
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
from kinomlx.io.output_paths import reserve_output_base
from kinomlx.model_registry import config_registry
from kinomlx.models.gmnet.output import GMNetOutputSink, plan_gmnet_output
from kinomlx.models.gmnet.runner import GMNetRunner
from kinomlx.reporter_stack import command_reporter
from kinomlx.reporting import ScopedReporter, TimingReporter
from kinomlx.settings import Settings
from kinomlx.ui import (
    configure_logging,
    configure_logging_from_settings,
)
from kinomlx.videotoolbox.heic import PQ_REFERENCE_WHITE_NITS

from .arguments import build_parser
from .config import GMNetInvocation, assemble, validate_for_execution
from .output import GMNetOutputPlan, GMNetOutputReservation
from .resources import GMNetResources, prepare_resources

_log = logging.getLogger(__name__)


def _model_name(invocation: GMNetInvocation) -> str:
    return f"{model_display_name('gmnet')} {invocation.model_settings.variant.title()}"


def _log_request(invocation: GMNetInvocation, plan: GMNetOutputPlan) -> None:
    request = invocation.request
    assert request is not None
    log_section(
        _log,
        "Run",
        (
            ("Model", _model_name(invocation)),
            ("Action", "expand"),
            ("Input", request.image),
            ("Requested artifacts", ", ".join(plan.artifacts.to_dict())),
        ),
    )


def _log_resources(resources: GMNetResources) -> None:
    log_section(
        _log,
        "Resources",
        (
            ("Variant", resources.spec.variant.value),
            ("Weights", resources.weights_path),
            ("Network scale", resources.spec.network_scale),
            ("Model peak over SDR white", f"{resources.spec.peak_over_sdr_white:g}x"),
            ("Model SDR reference white", f"{resources.spec.sdr_reference_white_nits:g} nits"),
        ),
    )


def _run_expand(
    invocation: GMNetInvocation,
    reporter: TimingReporter,
    *,
    plan: GMNetOutputPlan | None = None,
    reservation: GMNetOutputReservation | None = None,
) -> dict[str, object]:
    request = invocation.request
    assert request is not None
    selected_plan = plan_gmnet_output(request, invocation.output) if plan is None else plan
    model_reporter = ScopedReporter(reporter, model_display_name("gmnet"))

    def execute(selected_reservation: GMNetOutputReservation) -> dict[str, object]:
        runner = GMNetRunner(
            invocation.model_settings,
            infrastructure=invocation.settings,
            reporter=model_reporter,
        )
        _log_resources(runner.resources)
        result = runner.expand(request)
        reporter.memory_checkpoint("expansion_ready")
        artifacts = GMNetOutputSink(selected_plan, reporter=model_reporter).write(
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
        generation: dict[str, object] = {
            "variant": result.spec.variant.value,
            "weights_path": str(runner.resources.weights_path),
            "model_sdr_reference_white_nits": result.spec.sdr_reference_white_nits,
            "qmax_normalized": result.qmax_normalized,
            "achieved_peak_over_sdr_white": result.peak_linear,
            "achieved_model_contract_peak_nits": (
                result.peak_linear * result.spec.sdr_reference_white_nits
            ),
        }
        payload: dict[str, object] = {
            "action": "expand",
            "outputs": artifacts.to_dict(),
            "generation": generation,
            "warnings": [],
        }
        if artifacts.heic is not None:
            delivery_peak_nits = result.peak_linear * PQ_REFERENCE_WHITE_NITS
            _log.info(
                "PQ HEIC delivery uses %.0f-nit reference white; encoded peak about %.0f nits",
                PQ_REFERENCE_WHITE_NITS,
                delivery_peak_nits,
            )
            generation["pq_delivery_reference_white_nits"] = PQ_REFERENCE_WHITE_NITS
            generation["achieved_pq_delivery_peak_nits"] = delivery_peak_nits
        return payload

    if reservation is not None:
        return execute(reservation)
    with selected_plan.reserve() as owned_reservation:
        return execute(owned_reservation)


def run_gmnet_command(argv: list[str]) -> int:
    """Parse, resolve, run, and return a process exit code for GMNet."""

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
        model="gmnet",
        intent=invocation.config_intent,
        json_output=invocation.settings.json_output,
    )
    if config_exit is not None:
        return config_exit
    if invocation.operation.dry_run:
        return report_resources(
            "gmnet",
            functools.partial(
                prepare_resources, invocation.model_settings, infrastructure=invocation.settings
            ),
            log=_log_resources,
            json_output=invocation.settings.json_output,
        )
    started = time.perf_counter()
    try:
        validate_for_execution(invocation)
    except ConfigError as exc:
        return render_error(
            f"config error: {exc}",
            json_output=invocation.settings.json_output,
        )

    request = invocation.request
    assert request is not None
    plan = plan_gmnet_output(request, invocation.output)
    command_argv = ["kinomlx", *argv]
    sidecar_errors: list[dict[str, str]] = []
    run_record = None
    try:
        with ExitStack() as output_stack:

            def reserve(base: Path) -> GMNetOutputReservation:
                selected = plan_gmnet_output(request, invocation.output, base=base)
                return output_stack.enter_context(selected.reserve())

            reservation = reserve_output_base(
                plan.base,
                reserve,
                generated=invocation.output.path is None,
            )
            plan = reservation.plan
            sidecar_paths = plan.sidecar_paths()
            planned_outputs = {
                **plan.artifacts.to_dict(),
                **{name: str(path) for name, path in sidecar_paths.items()},
            }
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

            _log_request(invocation, plan)
            log_output_paths(_log, planned_outputs, heading="Planned outputs")
            with ExitStack() as stack:
                reporter = command_reporter(stack, invocation.settings)
                reporter.memory_checkpoint("run_start")
                run_log = sidecar_paths.get("run_log")
                if run_log is not None:
                    try:
                        run_record = RunRecord(
                            run_log,
                            schema=config_registry().model("gmnet"),
                            effective_config=invocation.resolved_config,
                            config_intent=invocation.config_intent,
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
                    memory = reporter.memory_to_dict()
                    diagnostics = {"memory": memory} if memory else {}
                    if run_record is not None:
                        try:
                            generation = payload["generation"]
                            warnings = payload["warnings"]
                            assert isinstance(generation, dict)
                            assert isinstance(warnings, list)
                            run_record.write(
                                status="completed",
                                outputs=completed_outputs,
                                generation=generation,
                                diagnostics=diagnostics,
                                warnings=warnings,
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
                    if isinstance(exc, KeyboardInterrupt):
                        raise RunCanceled(failed_outputs) from exc
                    raise
    except (KinoMLXError, FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return render_model_error(
            "gmnet",
            "expansion",
            exc,
            json_output=invocation.settings.json_output,
        )

    elapsed = time.perf_counter() - started
    outputs = payload["outputs"]
    generation = payload["generation"]
    warnings = payload["warnings"]
    assert isinstance(outputs, dict)
    assert isinstance(generation, dict)
    assert isinstance(warnings, list)
    render_run_success(
        _log,
        model="gmnet",
        action="expand",
        outputs=outputs,
        elapsed_seconds=elapsed,
        timings=timings,
        diagnostics=diagnostics,
        json_output=invocation.settings.json_output,
        generation=generation,
        warnings=warnings,
        sidecar_errors=sidecar_errors,
        planned_outputs=planned_outputs,
    )
    return 0


__all__ = ["run_gmnet_command"]
