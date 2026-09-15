"""Stateless Krea recipe host over immutable resources and structural ports."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Protocol

import kinomlx.models.krea2.pipelines.restart as _restart_pipeline
from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.errors import KinoMLXError
from kinomlx.recipe_host import execute_recipe
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .components import Krea2Components, NativeKrea2Components
from .pipelines.image_edit import generate_krea2_edit
from .pipelines.text_to_image import Krea2GenerationOutput, generate_krea2
from .resource_types import Krea2Resources
from .resources import prepare_resources
from .restart import Krea2Restart
from .sampling import Krea2Checkpoint
from .settings import Krea2Settings
from .text_conditioning import Krea2TextConditioner
from .types import Krea2GenerateConfig, Krea2Request

_log = logging.getLogger(__name__)


class Krea2Error(KinoMLXError, RuntimeError):
    """Typed operational failure at the Krea host boundary."""


class Recipe[RequestT](Protocol):
    """A public Krea recipe callable hosted through the standard ports."""

    def __call__(
        self,
        request: RequestT,
        resources: Krea2Resources,
        *,
        components: Krea2Components | None = None,
        text_conditioner: Krea2TextConditioner | None = None,
        reporter: Reporter | None = None,
        checkpoint: Krea2Checkpoint | None = None,
    ) -> Krea2GenerationOutput: ...


class Krea2Runner:
    """Convenience host retaining immutable resources and stateless ports only."""

    def __init__(
        self,
        model_settings: Krea2Settings | None = None,
        *,
        infrastructure: Settings | None = None,
        resources: Krea2Resources | None = None,
        components: Krea2Components | None = None,
        text_conditioner: Krea2TextConditioner | None = None,
        reporter: Reporter | None = None,
        checkpoint: Krea2Checkpoint | None = None,
    ) -> None:
        self.reporter = reporter if reporter is not None else NullReporter()
        if resources is None:
            selected = model_settings if model_settings is not None else Krea2Settings.from_env()
            selected_infrastructure = (
                infrastructure if infrastructure is not None else Settings.from_env()
            )
            try:
                resources = prepare_resources(
                    selected,
                    infrastructure=selected_infrastructure,
                )
            except KinoMLXError:
                raise
            except FileNotFoundError:
                raise
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                raise Krea2Error(f"cannot prepare Krea 2 resources: {exc}") from exc
        elif model_settings is not None or infrastructure is not None:
            raise ValueError("pass settings or prepared resources, not both")
        self.resources = resources
        self.components = (
            components if components is not None else NativeKrea2Components(reporter=self.reporter)
        )
        self.text_conditioner = text_conditioner
        self.checkpoint = checkpoint

    def run[RequestT](self, recipe: Recipe[RequestT], request: RequestT) -> Krea2GenerationOutput:
        """Run any Krea recipe accepting the standard stateless host ports."""
        return execute_recipe(
            partial(
                recipe,
                request,
                self.resources,
                components=self.components,
                text_conditioner=self.text_conditioner,
                reporter=self.reporter,
                checkpoint=self.checkpoint,
            ),
            memory_policy=self.resources.memory_policy,
            result_type=Krea2GenerationOutput,
            error_type=Krea2Error,
            label="Krea 2 generation",
            logger=_log,
        )

    def generate(
        self,
        request: Krea2Request,
        *,
        on_image: Callable[[int, Krea2GenerationOutput], None] | None = None,
    ) -> Krea2GenerationOutput:
        """Run the standard Raw/Turbo text-to-image recipe."""
        return self.run(
            partial(
                generate_krea2_edit if request.source_images else generate_krea2, on_image=on_image
            ),
            request,
        )

    def restart(
        self,
        request: Krea2Request,
        restart: Krea2Restart,
        *,
        on_image: Callable[[int, Krea2GenerationOutput], None] | None = None,
    ) -> Krea2GenerationOutput:
        """Resume from saved text conditioning or final latents."""
        recipe = partial(_restart_pipeline.restart_krea2, restart=restart, on_image=on_image)
        return self.run(recipe, request)


API_BOUNDARIES = (
    APIBoundary(
        "krea2",
        Krea2Runner.__init__,
        (
            configuration("model_settings", Krea2Settings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
            runtime("resources", "prepared resource inventory"),
            runtime("components", "component provider"),
            runtime("text_conditioner", "text-conditioning provider"),
            runtime("reporter", "progress reporter"),
            runtime("checkpoint", "cancellation checkpoint callback"),
        ),
    ),
    APIBoundary(
        "krea2",
        Krea2Runner.run,
        (
            runtime("recipe", "recipe callable"),
            runtime("request", "recipe-selected request port"),
        ),
    ),
    APIBoundary(
        "krea2",
        Krea2Runner.generate,
        (
            configuration("request", Krea2GenerateConfig, "generate", projection=True),
            runtime("on_image", "image callback"),
        ),
    ),
    APIBoundary(
        "krea2",
        Krea2Runner.restart,
        (
            configuration("request", Krea2GenerateConfig, "generate", projection=True),
            runtime("restart", "prepared station handoff"),
            runtime("on_image", "image callback"),
        ),
    ),
)


__all__ = ["API_BOUNDARIES", "Krea2Error", "Krea2Runner", "Recipe"]
