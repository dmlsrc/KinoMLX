"""Structural Krea 2 ports and default native component provider."""

import gc
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.components import ComponentLease
from kinomlx.modeling.dtypes import resolve_bound_dtype
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.seed_execution import SeedModels

from .adapters.loading import Krea2AdapterPlan, apply_krea2_adapters
from .resource_types import Krea2Resources
from .transformer import (
    Krea2BoundaryObserver,
    Krea2Transformer,
    Krea2TransformerOutput,
    load_krea2_transformer_weights,
)
from .transformer.reference import ReferenceAttention
from .vae import (
    Krea2DecodedRGB,
    Krea2ImageDecoder,
    Krea2VAEBoundaryObserver,
    Krea2VAETilingConfig,
    load_krea2_vae_weights,
)


class Krea2TransformerPort(Protocol):
    """Callable MMDiT surface consumed by later Krea recipes."""

    @property
    def adapter_receipts(self) -> tuple[AdapterReceipt, ...]: ...

    def __call__(
        self,
        image_tokens: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None,
        observer: Krea2BoundaryObserver | None = None,
        reference_attention: ReferenceAttention | None = None,
    ) -> Krea2TransformerOutput: ...


class Krea2TransformerProvider(Protocol):
    """Structural provider for bounded transformer ownership."""

    def transformer(
        self,
        resources: Krea2Resources,
        *,
        adapter_plan: Krea2AdapterPlan | None = None,
    ) -> ComponentLease[Krea2TransformerPort]: ...


class Krea2ImageDecoderPort(Protocol):
    """Typed floating-RGB surface consumed by later Krea recipes."""

    def __call__(
        self,
        latent: mx.array,
        *,
        reporter: Reporter | None = None,
        observer: Krea2VAEBoundaryObserver | None = None,
        tiling: Krea2VAETilingConfig | None = None,
    ) -> Krea2DecodedRGB: ...


class Krea2ImageDecoderProvider(Protocol):
    """Structural provider for bounded image-decoder ownership."""

    def image_decoder(
        self,
        resources: Krea2Resources,
    ) -> ComponentLease[Krea2ImageDecoderPort]: ...


class Krea2ComponentsProvider(Krea2TransformerProvider, Krea2ImageDecoderProvider, Protocol):
    """Component surface required by the current Krea implementation phase."""


def _cleanup_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def load_transformer(
    resources: Krea2Resources,
    *,
    reporter: Reporter | None = None,
    adapter_plan: Krea2AdapterPlan | None = None,
) -> ComponentLease[Krea2TransformerPort]:
    """Bind and lease one transformer, leaving its weights lazy until first use."""
    model: Krea2Transformer | None = None
    try:
        sink = reporter if reporter is not None else NullReporter()
        with reported_phase(sink, "load transformer"):
            model = Krea2Transformer(
                resources.transformer_config,
                reference_denoise=resources.reference_denoise,
            )
            load_krea2_transformer_weights(
                model,
                source_bindings=resources.transformer.bindings,
                expected_dtype=resolve_bound_dtype(
                    resources.transformer.dtype_policy, owner="Krea 2 transformer"
                ),
            )
        if adapter_plan is not None and adapter_plan.adapters:
            with reported_phase(sink, "fuse adapters"):
                model.adapter_receipts = apply_krea2_adapters(
                    model,
                    adapter_plan,
                    variant=resources.variant.value,
                    check_weights=resources.check_adapter_weights,
                )
        port: Krea2TransformerPort = model
        return ComponentLease(port, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_image_decoder(
    resources: Krea2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[Krea2ImageDecoderPort]:
    """Load, evaluate, and lease one exact prepared Qwen Image decoder."""
    model: Krea2ImageDecoder | None = None
    try:
        sink = reporter if reporter is not None else NullReporter()
        with reported_phase(sink, "load VAE"):
            vae_dtype = resolve_bound_dtype(resources.vae.dtype_policy, owner="Krea 2 VAE")
            model = Krea2ImageDecoder(
                resources.vae_config,
                compute_dtype=vae_dtype,
                output_scale=resources.vae_decoder.output_scale,
            )
            load_krea2_vae_weights(
                model,
                source_bindings=resources.vae.bindings,
                expected_dtype=vae_dtype,
            )
            mx.eval(model.parameters())
        port: Krea2ImageDecoderPort = model
        return ComponentLease(port, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


@dataclass(frozen=True)
class Krea2Components:
    """Default native provider with no retained heavyweight component."""

    reporter: Reporter | None = None

    def transformer(
        self,
        resources: Krea2Resources,
        *,
        adapter_plan: Krea2AdapterPlan | None = None,
    ) -> ComponentLease[Krea2TransformerPort]:
        return load_transformer(resources, reporter=self.reporter, adapter_plan=adapter_plan)

    def image_decoder(
        self,
        resources: Krea2Resources,
    ) -> ComponentLease[Krea2ImageDecoderPort]:
        return load_image_decoder(resources, reporter=self.reporter)


__all__ = [
    "Krea2Components",
    "Krea2ComponentsProvider",
    "Krea2ImageDecoderPort",
    "Krea2ImageDecoderProvider",
    "Krea2TransformerPort",
    "Krea2TransformerProvider",
    "load_image_decoder",
    "load_transformer",
]


@dataclass(frozen=True)
class Krea2SeedComponents:
    """Borrow request-owned components while preserving the normal stage ports."""

    base: Krea2ComponentsProvider
    models: SeedModels

    def transformer(
        self, resources: Krea2Resources, *, adapter_plan: Krea2AdapterPlan | None = None
    ) -> ComponentLease[Krea2TransformerPort]:
        return self.models.borrow(
            "transformer", lambda: self.base.transformer(resources, adapter_plan=adapter_plan)
        )

    def image_decoder(self, resources: Krea2Resources) -> ComponentLease[Krea2ImageDecoderPort]:
        return self.models.borrow("vae", lambda: self.base.image_decoder(resources))
