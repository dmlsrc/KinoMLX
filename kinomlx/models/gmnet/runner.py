"""Stateless recipe host over immutable GMNet resources and public ports."""

from __future__ import annotations

from typing import Protocol

from kinomlx.errors import KinoMLXError
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .components import GMNetComponents, NativeGMNetComponents
from .expand import ExpansionResult
from .resources import GMNetResources, prepare_resources
from .settings import GMNetSettings
from .types import GMNetRequest


class GMNetError(KinoMLXError, RuntimeError):
    """Typed operational failure at the GMNet host boundary."""


class Recipe(Protocol):
    """A public GMNet recipe callable hosted through standard ports."""

    def __call__(
        self,
        request: GMNetRequest,
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
                    reporter=self.reporter,
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

    def run(self, recipe: Recipe, request: GMNetRequest) -> ExpansionResult:
        """Run any GMNet recipe that accepts the standard stateless host ports."""
        try:
            output = recipe(
                request,
                self.resources,
                components=self.components,
                reporter=self.reporter,
            )
        except KinoMLXError:
            raise
        except FileNotFoundError:
            raise
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            raise GMNetError(f"GMNet expansion failed: {exc}") from exc
        if not isinstance(output, ExpansionResult):
            raise TypeError("a GMNet recipe must return ExpansionResult")
        return output

    def expand(self, request: GMNetRequest) -> ExpansionResult:
        """Run the standard SDR-to-HDR expansion recipe."""
        from .pipeline import expand_gmnet

        return self.run(expand_gmnet, request)


__all__ = ["GMNetError", "GMNetRunner", "Recipe"]
