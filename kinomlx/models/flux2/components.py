"""Public Klein component leases and the default native provider."""

import gc
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.components import ComponentLease
from kinomlx.modeling.dtypes import resolve_bound_dtype
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.seed_execution import SeedModels

from .adapters.loading import PreparedAdapter, apply_flux2_adapters
from .resources import Flux2Resources
from .restart import PreparedResources
from .source_loading import component_weight_files, load_transformer_component, load_vae
from .transformer import Flux2Transformer
from .vae import Flux2VAE
from .velocity import PreparedVelocity, VelocityModel

type Resources = Flux2Resources | PreparedResources


class Flux2TransformerPort(VelocityModel, Protocol):
    """Velocity surface consumed by the Klein recipes, plus its fused adapters."""

    @property
    def adapter_receipts(self) -> tuple[AdapterReceipt, ...]: ...

    def prepare_velocities(
        self, text: mx.array, image_ids: tuple[mx.array, ...], text_ids: mx.array
    ) -> tuple[PreparedVelocity, ...]: ...


class Flux2TransformerProvider(Protocol):
    """Structural provider for bounded transformer ownership."""

    def transformer(
        self,
        resources: Resources,
        *,
        adapter_plan: tuple[PreparedAdapter, ...] = (),
    ) -> ComponentLease[Flux2TransformerPort]: ...


class Flux2ImageDecoderPort(Protocol):
    """Typed normalized-patch decode surface consumed by the Klein recipes."""

    def decode_normalized(self, patches: mx.array) -> mx.array: ...


class Flux2ImageDecoderProvider(Protocol):
    """Structural provider for bounded image-decoder ownership."""

    def image_decoder(self, resources: Resources) -> ComponentLease[Flux2ImageDecoderPort]: ...


class Flux2ComponentsProvider(Flux2TransformerProvider, Flux2ImageDecoderProvider, Protocol):
    """Component surface required by the Klein recipes."""


def _cleanup_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def load_transformer(
    resources: Resources,
    *,
    reporter: Reporter | None = None,
    adapter_plan: tuple[PreparedAdapter, ...] = (),
) -> ComponentLease[Flux2TransformerPort]:
    """Bind and lease one transformer, fusing any adapters onto that fresh model."""
    model: Flux2Transformer | None = None
    try:
        sink = reporter if reporter is not None else NullReporter()
        component = resources.component("transformer")
        model = load_transformer_component(
            component, dtype=resolve_bound_dtype(component.dtype_policy), reporter=sink
        )
        if adapter_plan:
            with reported_phase(sink, "fuse_adapters"):
                model.adapter_receipts = apply_flux2_adapters(
                    model,
                    adapter_plan,
                    variant=resources.profile.variant.value,
                    check_weights=resources.check_adapter_weights,
                )
        port: Flux2TransformerPort = model
        return ComponentLease(port, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_image_decoder(
    resources: Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[Flux2ImageDecoderPort]:
    """Load and lease one decode-only Klein VAE."""
    model: Flux2VAE | None = None
    try:
        sink = reporter if reporter is not None else NullReporter()
        component = resources.component("vae")
        model = load_vae(
            component_weight_files(component),
            mode="decode",
            dtype=resolve_bound_dtype(component.dtype_policy, owner="Klein VAE"),
            reporter=sink,
        )
        port: Flux2ImageDecoderPort = model
        return ComponentLease(port, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


@dataclass(frozen=True)
class Flux2Components:
    """Default provider implementing the public Klein component surface."""

    reporter: Reporter | None = None

    def transformer(
        self,
        resources: Resources,
        *,
        adapter_plan: tuple[PreparedAdapter, ...] = (),
    ) -> ComponentLease[Flux2TransformerPort]:
        return load_transformer(resources, reporter=self.reporter, adapter_plan=adapter_plan)

    def image_decoder(self, resources: Resources) -> ComponentLease[Flux2ImageDecoderPort]:
        return load_image_decoder(resources, reporter=self.reporter)


@dataclass(frozen=True)
class Flux2SeedComponents:
    """Borrow request-owned components while preserving the normal stage ports."""

    base: Flux2ComponentsProvider
    models: SeedModels

    def transformer(
        self,
        resources: Resources,
        *,
        adapter_plan: tuple[PreparedAdapter, ...] = (),
    ) -> ComponentLease[Flux2TransformerPort]:
        return self.models.borrow(
            "transformer", lambda: self.base.transformer(resources, adapter_plan=adapter_plan)
        )

    def image_decoder(self, resources: Resources) -> ComponentLease[Flux2ImageDecoderPort]:
        return self.models.borrow("vae", lambda: self.base.image_decoder(resources))


__all__ = [
    "Flux2Components",
    "Flux2ComponentsProvider",
    "Flux2ImageDecoderPort",
    "Flux2ImageDecoderProvider",
    "Flux2SeedComponents",
    "Flux2TransformerPort",
    "Flux2TransformerProvider",
    "load_image_decoder",
    "load_transformer",
]
