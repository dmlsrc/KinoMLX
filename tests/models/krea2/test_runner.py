"""Krea runner error translation and heavyweight-object lifetime."""

import gc
import logging
import weakref
from types import SimpleNamespace

import pytest

from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.krea2.runner import Krea2Error, Krea2Runner
from kinomlx.models.krea2.types import Krea2Request


def test_translated_failure_does_not_retain_recipe_traceback_objects() -> None:
    reference = None

    class Heavyweight:
        pass

    def recipe(*_args, **_kwargs):
        nonlocal reference
        heavyweight = Heavyweight()
        reference = weakref.ref(heavyweight)
        raise RuntimeError("injected failure")

    runner = Krea2Runner(
        resources=SimpleNamespace(memory_policy=MLXMemoryPolicy()),
        components=SimpleNamespace(),
    )
    try:
        runner.run(recipe, Krea2Request("test"))
    except Krea2Error as exc:
        translated = exc
    else:
        raise AssertionError("the injected failure was not translated")

    gc.collect()
    assert reference is not None
    assert reference() is None
    assert translated.__cause__ is None
    assert translated.__context__ is None


def test_translated_failure_logs_original_traceback_at_debug(caplog) -> None:
    def recipe(*_args, **_kwargs):
        raise RuntimeError("debuggable failure")

    runner = Krea2Runner(
        resources=SimpleNamespace(memory_policy=MLXMemoryPolicy()),
        components=SimpleNamespace(),
    )
    with (
        caplog.at_level(logging.DEBUG, logger="kinomlx.models.krea2.runner"),
        pytest.raises(Krea2Error, match="debuggable failure"),
    ):
        runner.run(recipe, Krea2Request("test"))

    record = next(record for record in caplog.records if record.levelno == logging.DEBUG)
    assert record.exc_info is None
    assert "Traceback (most recent call last)" in record.getMessage()
    assert "in recipe" in record.getMessage()
    assert "RuntimeError: debuggable failure" in record.getMessage()
