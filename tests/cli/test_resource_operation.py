"""The shared resource-only operation: its table, its refusal, its branch."""

import json
import logging

import pytest

from kinomlx._json import JsonObject
from kinomlx.cli.resource_operation import (
    Operation,
    operation_table,
    refuse_resource_only_work,
    report_resources,
)
from kinomlx.config import ConfigError
from kinomlx.errors import KinoMLXError


class _Resources:
    def __init__(self, report: JsonObject) -> None:
        self.report = report

    def describe(self) -> JsonObject:
        return self.report


def _prepare(report: JsonObject):
    return lambda: _Resources(report)


def _raising(error: BaseException):
    def prepare() -> _Resources:
        raise error

    return prepare


def test_the_table_locks_the_flag_at_every_stage_a_model_declares() -> None:
    table = operation_table(cli_group="Klein execution", stages=("decode", "denoise"))

    assert table.path == ("operation",)
    assert table.record is Operation
    assert table.stage_record is Operation
    (field,) = table.fields
    assert field.name == "dry_run"
    assert field.cli is not None
    assert field.cli.action == "store_true"
    assert {policy.stage: policy.mode for policy in field.stage_policies} == {
        "decode": "locked",
        "denoise": "locked",
    }
    assert all(policy.reason.strip() for policy in field.stage_policies)


def test_a_model_without_restart_declares_no_stage_record() -> None:
    table = operation_table(cli_group="GMNet execution", stages=())

    assert table.stage_record is None
    assert table.fields[0].stage_policies == ()


@pytest.mark.parametrize("requested", [True, False])
def test_the_refusal_fires_only_when_a_dry_run_also_asks_for_work(requested: bool) -> None:
    if requested:
        with pytest.raises(ConfigError, match="cannot be combined with an expansion"):
            refuse_resource_only_work(dry_run=True, requested=True, work="an expansion")
    else:
        refuse_resource_only_work(dry_run=True, requested=False, work="an expansion")
    # A run that is not resource-only is never refused for asking to work.
    refuse_resource_only_work(dry_run=False, requested=requested, work="an expansion")


def test_the_branch_logs_for_a_reader_and_emits_the_report_for_a_machine(capsys, caplog) -> None:
    report: JsonObject = {"model": "fixture", "components": []}
    logged: list[object] = []

    with caplog.at_level(logging.INFO, logger="kinomlx"):
        assert (
            report_resources("fixture", _prepare(report), log=logged.append, json_output=False) == 0
        )
    assert len(logged) == 1
    assert capsys.readouterr().out == ""

    assert report_resources("fixture", _prepare(report), log=logged.append, json_output=True) == 0
    assert json.loads(capsys.readouterr().out) == report
    # The machine surface reports instead of logging, not as well as.
    assert len(logged) == 1


def test_a_preparation_time_configuration_mistake_stays_the_user_s(caplog) -> None:
    """Preparation reaches the user's own configuration, so it can refuse it.

    Krea's decoder-selection rule is only checkable against the checkpoint, so
    a contradiction surfaces here rather than during assembly. It is still the
    user's mistake and must not read like a fault in the model.
    """
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = report_resources(
            "fixture",
            _raising(ConfigError("[model_settings]: contradicts the detected decoder")),
            log=lambda _resources: pytest.fail("a refused invocation reports nothing"),
            json_output=False,
        )

    assert exit_code == 2
    assert "config error: [model_settings]:" in caplog.text
    assert "preparation failed" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        KinoMLXError("checkpoint is incomplete"),
        FileNotFoundError("no such component"),
        ValueError("tensor contract"),
        ArithmeticError("impossible geometry"),
    ],
)
def test_any_other_preparation_failure_is_named_as_one(error: BaseException, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = report_resources(
            "fixture",
            _raising(error),
            log=lambda _resources: pytest.fail("a failed preparation reports nothing"),
            json_output=False,
        )

    assert exit_code == 2
    assert "preparation" in caplog.text
    assert "config error:" not in caplog.text
