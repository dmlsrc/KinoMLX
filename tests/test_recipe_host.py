"""The shared recipe host labels every failure of the operation it runs."""

import logging

import pytest

from kinomlx.config_values import ConfigError
from kinomlx.errors import KinoMLXError
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.recipe_host import execute_recipe


class _RecipeError(KinoMLXError):
    pass


def _host(operation):
    return execute_recipe(
        operation,
        result_type=dict,
        error_type=_RecipeError,
        label="Probe generation",
        logger=logging.getLogger("tests.recipe_host"),
        memory_policy=MLXMemoryPolicy(),
    )


def test_a_request_refused_inside_the_recipe_is_labeled_like_any_other_failure():
    def refuse():
        raise ConfigError("[generate]: steps must be greater than or equal to 1")

    with pytest.raises(_RecipeError) as error:
        _host(refuse)
    assert str(error.value) == (
        "Probe generation failed: [generate]: steps must be greater than or equal to 1"
    )
    assert error.value.__cause__ is None


def test_an_ordinary_value_error_keeps_the_same_label():
    def fail():
        raise ValueError("latents are not finite")

    with pytest.raises(_RecipeError, match="^Probe generation failed: latents are not finite$"):
        _host(fail)


def test_a_result_of_the_wrong_type_is_a_programming_error():
    with pytest.raises(TypeError, match="must return dict"):
        _host(lambda: ["not", "a", "dict"])
