"""Stateless Klein recipe host over prepared resources and reporting interfaces."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Protocol

from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.errors import KinoMLXError
from kinomlx.recipe_host import execute_recipe
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .generation import Flux2ImageResult
from .pipeline import Flux2GenerationOutput, generate_flux2, restart_flux2
from .resources import Flux2Resources, inspect_resources
from .restart import Flux2Restart, PreparedResources
from .settings import Flux2Settings
from .stations import Flux2Conditioning
from .types import Flux2Request

_log = logging.getLogger(__name__)


class Flux2Error(KinoMLXError, RuntimeError):
    """Typed operational failure at the Klein host boundary."""


class Recipe[RequestT](Protocol):
    """A Klein recipe accepting the standard stateless host interfaces."""

    def __call__(
        self,
        request: RequestT,
        resources: Flux2Resources | PreparedResources,
        *,
        reporter: Reporter | None = None,
        conditioning: Flux2Conditioning | None = None,
    ) -> Flux2GenerationOutput: ...


class Flux2Runner:
    """Retain resource descriptions and reporting, never live models or reference KV."""

    def __init__(
        self,
        model_settings: Flux2Settings | None = None,
        *,
        infrastructure: Settings | None = None,
        resources: Flux2Resources | PreparedResources | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.reporter = reporter if reporter is not None else NullReporter()
        if resources is None:
            selected = model_settings if model_settings is not None else Flux2Settings.from_env()
            settings = infrastructure if infrastructure is not None else Settings.from_env()
            settings.validate()
            try:
                resources = inspect_resources(selected, infrastructure=settings)
            except KinoMLXError, FileNotFoundError:
                raise
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                raise Flux2Error(f"cannot prepare Klein resources: {exc}") from exc
        elif model_settings is not None or infrastructure is not None:
            raise ValueError("pass settings or prepared resources, not both")
        self.resources = resources

    def run[RequestT](
        self,
        recipe: Recipe[RequestT],
        request: RequestT,
        *,
        conditioning: Flux2Conditioning | None = None,
    ) -> Flux2GenerationOutput:
        """Run a typed recipe with consistent reporting and operational errors."""
        return execute_recipe(
            partial(
                recipe, request, self.resources, reporter=self.reporter, conditioning=conditioning
            ),
            memory_policy=self.resources.memory_policy,
            result_type=Flux2GenerationOutput,
            error_type=Flux2Error,
            label="FLUX.2 Klein generation",
            logger=_log,
        )

    def generate(
        self,
        request: Flux2Request,
        *,
        conditioning: Flux2Conditioning | None = None,
        on_image: Callable[[int, Flux2ImageResult], None] | None = None,
    ) -> Flux2GenerationOutput:
        """Run the selected base, distilled, or KV generation/edit recipe."""
        return self.run(
            partial(generate_flux2, on_image=on_image), request, conditioning=conditioning
        )

    def restart(
        self,
        request: Flux2Request,
        restart: Flux2Restart,
        *,
        conditioning: Flux2Conditioning | None = None,
    ) -> Flux2GenerationOutput:
        """Resume from complete conditioning or final latents."""
        return self.run(partial(restart_flux2, restart=restart), request, conditioning=conditioning)


API_BOUNDARIES = (
    APIBoundary(
        "flux2",
        Flux2Runner.__init__,
        (
            configuration("model_settings", Flux2Settings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
            runtime("resources", "prepared resource inventory"),
            runtime("reporter", "progress reporter"),
        ),
    ),
    APIBoundary(
        "flux2",
        Flux2Runner.run,
        (
            runtime("recipe", "recipe callable"),
            runtime("request", "recipe-selected request port"),
            runtime("conditioning", "retained prepared conditioning"),
        ),
    ),
    APIBoundary(
        "flux2",
        Flux2Runner.generate,
        (
            configuration("request", Flux2Request, "generate"),
            runtime("conditioning", "retained prepared conditioning"),
            runtime("on_image", "image callback"),
        ),
    ),
    APIBoundary(
        "flux2",
        Flux2Runner.restart,
        (
            configuration("request", Flux2Request, "generate"),
            runtime("restart", "prepared station handoff"),
            runtime("conditioning", "retained prepared conditioning"),
        ),
    ),
)
