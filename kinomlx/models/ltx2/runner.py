"""Stateless recipe host over immutable LTX-2 resources and public ports."""

import logging
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from typing import Protocol

import kinomlx.models.ltx2.pipelines.restart as _restart_pipeline
from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.errors import KinoMLXError
from kinomlx.recipe_host import execute_recipe
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .components import LTX2Components, LTX2ComponentsProvider
from .generation import LTX2GenerationOutput
from .pipelines.distilled import DistilledConditioning, generate_ltx2, prepare_distilled_recipe
from .resources import LTX2Resources, prepare_resources
from .restart import LTX2Restart
from .settings import LTX2Settings
from .text_conditioning import LTX2TextConditionerPort
from .types import LTX2Request

_log = logging.getLogger(__name__)


class LTX2Error(KinoMLXError, RuntimeError):
    """Typed operational failure at the LTX-2 host boundary."""


class Recipe[RequestT](Protocol):
    """A public recipe callable hosted without recipe-specific branching."""

    def __call__(
        self,
        request: RequestT,
        resources: LTX2Resources,
        *,
        components: LTX2ComponentsProvider | None = None,
        text_conditioner: LTX2TextConditionerPort | None = None,
        reporter: Reporter | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> LTX2GenerationOutput: ...


class LTX2Runner:
    """Convenience host retaining only immutable resources and stateless ports."""

    def __init__(
        self,
        model_settings: LTX2Settings | None = None,
        *,
        infrastructure: Settings | None = None,
        resources: LTX2Resources | None = None,
        components: LTX2ComponentsProvider | None = None,
        text_conditioner: LTX2TextConditionerPort | None = None,
        reporter: Reporter | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self.reporter = reporter if reporter is not None else NullReporter()
        self.artifact_sink = artifact_sink if artifact_sink is not None else NullArtifactSink()
        if resources is None:
            selected_model_settings = (
                model_settings if model_settings is not None else LTX2Settings.from_env()
            )
            try:
                resources = prepare_resources(
                    selected_model_settings,
                    infrastructure=infrastructure,
                )
            except KinoMLXError:
                raise
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                raise LTX2Error(f"cannot prepare LTX-2 resources: {exc}") from exc
        elif model_settings is not None or infrastructure is not None:
            raise ValueError("pass settings or prepared resources, not both")
        self.resources = resources
        self.components = (
            components if components is not None else LTX2Components(reporter=self.reporter)
        )
        # Recipes own their default replay policy. Keeping an absent port as
        # None lets ordinary generation select strict saved-text replay while
        # stage restart selects observational identity handling.
        self.text_conditioner = text_conditioner

    def run[RequestT](self, recipe: Recipe[RequestT], request: RequestT) -> LTX2GenerationOutput:
        """Run any LTX-2 recipe that accepts the standard stateless host ports."""
        return execute_recipe(
            partial(
                recipe,
                request,
                self.resources,
                components=self.components,
                text_conditioner=self.text_conditioner,
                reporter=self.reporter,
                artifact_sink=self.artifact_sink,
            ),
            memory_policy=self.resources.memory_policy,
            result_type=LTX2GenerationOutput,
            error_type=LTX2Error,
            label="LTX-2 generation",
            logger=_log,
        )

    def generate(self, request: LTX2Request) -> Iterator[LTX2GenerationOutput]:
        """Yield complete videos sequentially, sharing conditioning across seeds.

        Consume and close each output before requesting the next. Close the
        iterator when stopping early to release retained conditioning. Each
        output records the same ``seed``/``seeds``/``seed_index`` selection
        receipt the CLI writes; the CLI drives its own loop because it owns one
        sidecar bundle and run record per seed.
        """
        shared = DistilledConditioning()
        seeds = request.selected_seeds()
        try:
            for seed in seeds:
                prepare_distilled_recipe(replace(request, seed=seed, seeds=None), self.resources)
            for index, seed in enumerate(seeds):
                output = self.run(
                    partial(generate_ltx2, conditioning=shared),
                    replace(request, seed=seed, seeds=None),
                )
                output.metadata.update(seed=seed, seeds=list(seeds), seed_index=index)
                yield output
        finally:
            shared.close()

    def restart(
        self,
        request: LTX2Request,
        restart: LTX2Restart,
    ) -> LTX2GenerationOutput:
        """Resume the distilled graph from one typed saved-stage product."""
        recipe = partial(_restart_pipeline.restart_ltx2, restart=restart)
        return self.run(recipe, request)


API_BOUNDARIES = (
    APIBoundary(
        "ltx2",
        LTX2Runner.__init__,
        (
            configuration("model_settings", LTX2Settings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
            runtime("resources", "prepared resource inventory"),
            runtime("components", "component provider"),
            runtime("text_conditioner", "text-conditioning provider"),
            runtime("reporter", "progress reporter"),
            runtime("artifact_sink", "artifact writer"),
        ),
    ),
    APIBoundary(
        "ltx2",
        LTX2Runner.run,
        (
            runtime("recipe", "recipe callable"),
            runtime("request", "recipe-selected request port"),
        ),
    ),
    APIBoundary(
        "ltx2",
        LTX2Runner.generate,
        (configuration("request", LTX2Request, "generate"),),
    ),
    APIBoundary(
        "ltx2",
        LTX2Runner.restart,
        (
            configuration("request", LTX2Request, "generate"),
            runtime("restart", "prepared stage handoff"),
        ),
    ),
)


__all__ = [
    "API_BOUNDARIES",
    "LTX2Error",
    "LTX2Runner",
    "Recipe",
]
