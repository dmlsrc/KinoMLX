"""Structural Klein component providers and lease-ownership contracts."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.components import ComponentLease
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.flux2 import components, generation
from kinomlx.models.flux2.catalog import profile
from kinomlx.models.flux2.components import Flux2Components
from kinomlx.models.flux2.pipelines import generate_flux2
from kinomlx.models.flux2.types import Flux2Request

from .test_text_conditioning import conditioning


def _resources(variant: str = "distilled"):
    return SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        check_adapter_weights=True,
        profile=profile(variant),
        component=lambda kind: SimpleNamespace(dtype_policy="bfloat16", bindings=()),
    )


def test_native_provider_leases_a_transformer_without_retaining_it(monkeypatch):
    events: list[str] = []

    class _Model:
        adapter_receipts = ()

    monkeypatch.setattr(
        components, "load_transformer_component", lambda *a, **k: events.append("load") or _Model()
    )
    lease = Flux2Components().transformer(_resources())
    assert events == ["load"]
    assert isinstance(lease.value, _Model)
    lease.close()


def test_native_provider_cleans_up_after_a_failed_transformer_load(monkeypatch):
    cleaned: list[str] = []

    def explode(*_args, **_kwargs):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(components, "load_transformer_component", explode)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: cleaned.append("cleanup"))
    with pytest.raises(RuntimeError, match="bind failed"):
        Flux2Components().transformer(_resources())
    assert cleaned == ["cleanup"]


def test_native_provider_cleans_up_after_a_failed_decoder_load(monkeypatch):
    cleaned: list[str] = []

    def explode(*_args, **_kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(components, "load_vae", explode)
    monkeypatch.setattr(components, "component_weight_files", lambda component: ())
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: cleaned.append("cleanup"))
    with pytest.raises(RuntimeError, match="decoder failed"):
        Flux2Components().image_decoder(_resources())
    assert cleaned == ["cleanup"]


def test_an_injected_provider_replaces_every_native_load(monkeypatch):
    events: list[str] = []
    value = conditioning()

    class _Transformer:
        adapter_receipts = ()

    class _Decoder:
        def decode_normalized(self, patches):
            events.append("decode")
            return mx.zeros((1, 3, 64, 64))

    class _Provider:
        def transformer(self, _resources, *, adapter_plan=()):
            events.append("transformer")
            return ComponentLease(_Transformer(), cleanup=lambda: events.append("close"))

        def image_decoder(self, _resources):
            events.append("decoder")
            return ComponentLease(_Decoder(), cleanup=lambda: events.append("close"))

    def forbidden(*_args, **_kwargs):
        pytest.fail("an injected provider must replace the native loaders")

    for name in ("load_transformer_component", "load_vae", "apply_flux2_adapters"):
        monkeypatch.setattr(components, name, forbidden)
    monkeypatch.setattr(generation, "denoise", lambda model, noise, *a, **k: noise)

    output = generate_flux2(
        Flux2Request(prompt=value.prompt, seeds=(3,)),
        _resources(),
        conditioning=value,
        components=_Provider(),
    )
    assert len(output.images) == 1
    assert events.count("transformer") == 1
    assert events.count("decoder") == 1
    assert "decode" in events
