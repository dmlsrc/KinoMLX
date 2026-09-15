"""Stateless recipe host over immutable GMNet resources and public ports."""

import logging
from functools import partial
from typing import Protocol

from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.errors import KinoMLXError
from kinomlx.recipe_host import execute_recipe
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .components import GMNetComponents, NativeGMNetComponents
from .expand import ExpansionResult
from .pipeline import expand_gmnet
from .resources import GMNetResources, prepare_resources
from .settings import GMNetSettings
from .types import GMNetExpandConfig, GMNetRequest

_log = logging.getLogger(__name__)


class GMNetError(KinoMLXError, RuntimeError):
    """Typed operational failure at the GMNet host boundary."""


class Recipe[RequestT](Protocol):
    """A public GMNet recipe callable hosted through standard ports."""

    def __call__(
        self,
        request: RequestT,
        resources: GMNetResources,
        *,
        components: GMNetComponents | None = None,
        reporter: Reporter | None = None,
    ) -> ExpansionResult: ...


class GMNetRunner:
    """Convenience host retaining immutable resources and stateless ports."""

    def __init__(
        self,
        model_settings: GMNetSettings | None = None,
        *,
        infrastructure: Settings | None = None,
        resources: GMNetResources | None = None,
        components: GMNetComponents | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.reporter = reporter if reporter is not None else NullReporter()
        if resources is None:
            selected = model_settings if model_settings is not None else GMNetSettings.from_env()
            try:
                resources = prepare_resources(
                    selected,
                    infrastructure=infrastructure,
                )
            except KinoMLXError:
                raise
            except FileNotFoundError:
                raise
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                raise GMNetError(f"cannot prepare GMNet resources: {exc}") from exc
        elif model_settings is not None or infrastructure is not None:
            raise ValueError("pass settings or prepared resources, not both")
        self.resources = resources
        self.components = (
            components if components is not None else NativeGMNetComponents(self.reporter)
        )

    def run[RequestT](self, recipe: Recipe[RequestT], request: RequestT) -> ExpansionResult:
        """Run any GMNet recipe that accepts the standard stateless host ports."""
        return execute_recipe(
            partial(
                recipe, request, self.resources, components=self.components, reporter=self.reporter
            ),
            memory_policy=self.resources.memory_policy,
            result_type=ExpansionResult,
            error_type=GMNetError,
            label="GMNet expansion",
            logger=_log,
        )

    def expand(self, request: GMNetRequest) -> ExpansionResult:
        """Run the standard SDR-to-HDR expansion recipe."""
        return self.run(expand_gmnet, request)


API_BOUNDARIES = (
    APIBoundary(
        "gmnet",
        GMNetRunner.__init__,
        (
            configuration("model_settings", GMNetSettings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
            runtime("resources", "prepared resource inventory"),
            runtime("components", "component provider"),
            runtime("reporter", "progress reporter"),
        ),
    ),
    APIBoundary(
        "gmnet",
        GMNetRunner.run,
        (
            runtime("recipe", "recipe callable"),
            runtime("request", "recipe-selected request port"),
        ),
    ),
    APIBoundary(
        "gmnet",
        GMNetRunner.expand,
        (configuration("request", GMNetExpandConfig, "expand", projection=True),),
    ),
)


__all__ = ["API_BOUNDARIES", "GMNetError", "GMNetRunner", "Recipe"]
