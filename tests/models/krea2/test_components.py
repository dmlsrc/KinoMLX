"""Structural Krea transformer provider and lease-ownership contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import kinomlx.models.krea2.components as components
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.krea2.adapters.loading import (
    Krea2AdapterPlan,
    Krea2PreparedAdapter,
)
from kinomlx.models.krea2.catalog import Krea2VAEDecoder
from kinomlx.models.krea2.transformer.loading import Krea2TransformerLoadReceipt
from kinomlx.models.krea2.vae.loading import Krea2VAELoadReceipt
from kinomlx.reporting import TimingReporter
from tests._reporting import RecordingReporter

from ._transformer_fixtures import tiny_transformer_config
from ._vae_fixtures import tiny_vae_config


def test_adapter_provider_defers_base_and_replacement_evaluation(tmp_path, monkeypatch):
    from kinomlx.adapters.selection import AdapterSelection

    selected = (AdapterSelection(tmp_path / "adapter.safetensors", 1.0),)
    tmp_path / "transformer.safetensors"
    received = []
    model = SimpleNamespace(adapter_receipts=())
    monkeypatch.setattr(components, "Krea2Transformer", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        components, "load_krea2_transformer_weights", lambda *_args, **_kwargs: None
    )

    plan = Krea2AdapterPlan(
        tiny_transformer_config().config_digest,
        (Krea2PreparedAdapter(selected[0], "source:test"),),
    )

    def apply(current, adapter_plan, *, variant, check_weights):
        received.append((current, adapter_plan, variant, check_weights))
        return ()

    monkeypatch.setattr(components, "apply_krea2_adapters", apply)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        transformer_config=tiny_transformer_config(),
        reference_denoise=False,
        transformer=SimpleNamespace(bindings=(), dtype_policy="bfloat16"),
        variant=SimpleNamespace(value="turbo"),
        check_adapter_weights=False,
    )
    with components.load_transformer(resources, adapter_plan=plan):
        assert received == [(model, plan, "turbo", False)]


def test_native_components_binds_lazy_transformer_without_retaining_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []

    class _Model:
        def __init__(self, config, *, reference_denoise) -> None:
            events.append(("construct", config, reference_denoise))

        def parameters(self):
            events.append("parameters")
            return ()

        def __call__(self, *_args, **_kwargs):
            return "result"

    def fake_load(
        model, *, expected_dtype=None, reporter=None, source_bindings=()
    ) -> Krea2TransformerLoadReceipt:
        events.append(("load", model, source_bindings, expected_dtype, reporter))
        return Krea2TransformerLoadReceipt(
            target_count=68,
            target_digest="target-digest",
            source_files=("transformer.safetensors",),
        )

    monkeypatch.setattr(components, "Krea2Transformer", _Model)
    monkeypatch.setattr(components, "load_krea2_transformer_weights", fake_load)
    monkeypatch.setattr(components.mx, "eval", lambda *_values: events.append("eval"))
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))
    reporter = RecordingReporter()
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        transformer_config=tiny_transformer_config(),
        reference_denoise=False,
        transformer=SimpleNamespace(bindings=(), dtype_policy="bfloat16"),
    )
    provider = components.NativeKrea2Components(reporter=reporter)

    lease = provider.transformer(resources)

    assert provider.__dict__ == {"reporter": reporter}
    assert lease.value("input") == "result"
    assert events == [
        ("construct", resources.transformer_config, False),
        ("load", lease.value, (), "bfloat16", None),
    ]
    lease.close()
    lease.close()
    assert events[-1] == "cleanup"
    assert lease.closed
    assert reporter.events == [
        ("start", "load transformer", {"total": None, "unit": "it"}),
        ("end", "load transformer", {}),
    ]


def test_transformer_provider_cleans_up_after_partial_load_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []

    class _Model:
        def __init__(self, _config, *, reference_denoise) -> None:
            events.append("construct")

    monkeypatch.setattr(components, "Krea2Transformer", _Model)
    monkeypatch.setattr(
        components,
        "load_krea2_transformer_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("broken cache")),
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        transformer_config=tiny_transformer_config(),
        reference_denoise=False,
        transformer=SimpleNamespace(bindings=(), dtype_policy="bfloat16"),
    )

    with pytest.raises(ValueError, match="broken cache"):
        components.load_transformer(resources)

    assert events == ["construct", "cleanup"]


def test_native_components_loads_and_leases_image_decoder_without_retaining_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []

    class _Model:
        def __init__(self, config, *, compute_dtype, output_scale) -> None:
            events.append(("construct", config, compute_dtype, output_scale))

        def parameters(self):
            events.append("parameters")
            return ()

        def __call__(self, *_args, **_kwargs):
            return "rgb"

    def fake_load(
        model, *, expected_dtype=None, source_bindings=(), reporter=None
    ) -> Krea2VAELoadReceipt:
        events.append(("load", model, expected_dtype, source_bindings, reporter))
        return Krea2VAELoadReceipt(
            target_count=104,
            target_digest="target-digest",
            source_files=("vae.safetensors",),
        )

    monkeypatch.setattr(components, "Krea2ImageDecoder", _Model)
    monkeypatch.setattr(components, "load_krea2_vae_weights", fake_load)
    monkeypatch.setattr(components.mx, "eval", lambda *_values: events.append("eval"))
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))
    reporter = RecordingReporter()
    bindings = ("source-binding",)
    resources = SimpleNamespace(
        vae_config=tiny_vae_config(),
        vae_decoder=Krea2VAEDecoder.UPSCALE2X,
        vae=SimpleNamespace(bindings=bindings, dtype_policy="float16"),
    )
    provider = components.NativeKrea2Components(reporter=reporter)

    lease = provider.image_decoder(resources)

    assert provider.__dict__ == {"reporter": reporter}
    assert lease.value("latent") == "rgb"
    assert events[:4] == [
        ("construct", resources.vae_config, "float16", 2),
        ("load", lease.value, "float16", bindings, None),
        "parameters",
        "eval",
    ]
    lease.close()
    lease.close()
    assert events[-1] == "cleanup"
    assert lease.closed
    assert reporter.events == [
        ("start", "load VAE", {"total": None, "unit": "it"}),
        ("end", "load VAE", {}),
    ]


@pytest.mark.parametrize("component", ["transformer", "image_decoder"])
def test_native_load_phase_accounts_for_its_evaluation_boundary(
    tmp_path: Path, monkeypatch, component: str
) -> None:
    now = [0.0]
    resident_bytes = [0]

    class _Model:
        def __init__(self, *_args, **_kwargs):
            now[0] += 0.5

        def parameters(self):
            return ()

    def bind(*_args, **_kwargs):
        now[0] += 1.0

    def materialize(*_values):
        now[0] += 3.0
        resident_bytes[0] = 1024

    monkeypatch.setattr(components, "Krea2Transformer", _Model)
    monkeypatch.setattr(components, "Krea2ImageDecoder", _Model)
    monkeypatch.setattr(components, "load_krea2_transformer_weights", bind)
    monkeypatch.setattr(components, "load_krea2_vae_weights", bind)
    monkeypatch.setattr(components.mx, "eval", materialize)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)
    reporter = TimingReporter(
        clock=lambda: now[0],
        memory_sampler=lambda: {
            "active_bytes": resident_bytes[0],
            "cache_bytes": 0,
            "peak_bytes": resident_bytes[0],
        },
    )
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        transformer_config=tiny_transformer_config(),
        reference_denoise=False,
        transformer=SimpleNamespace(bindings=(), dtype_policy="bfloat16"),
        vae_config=tiny_vae_config(),
        vae_decoder=Krea2VAEDecoder.QWEN,
        vae=SimpleNamespace(
            bindings=(),
            source_path=tmp_path / "vae.safetensors",
            dtype_policy="bfloat16",
        ),
    )
    provider = components.NativeKrea2Components(reporter=reporter)
    lease = getattr(provider, component)(resources)
    lease.close()

    (phase,) = reporter.to_dict()["phases"]
    assert phase["phase"] == ("load transformer" if component == "transformer" else "load VAE")
    assert phase["duration_seconds"] == (1.5 if component == "transformer" else 4.5)
    assert phase["status"] == "completed"
    expected_peak = 0 if component == "transformer" else 1024
    assert reporter.memory_to_dict()["phase_peaks"][0]["peak_bytes"] == expected_peak


def test_image_decoder_provider_cleans_up_after_partial_load_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []

    class _Model:
        def __init__(self, _config, *, compute_dtype, output_scale) -> None:
            del compute_dtype, output_scale
            events.append("construct")

    monkeypatch.setattr(components, "Krea2ImageDecoder", _Model)
    monkeypatch.setattr(
        components,
        "load_krea2_vae_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("broken VAE cache")),
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))
    resources = SimpleNamespace(
        vae_config=tiny_vae_config(),
        vae_decoder=Krea2VAEDecoder.QWEN,
        vae=SimpleNamespace(
            bindings=(),
            source_path=tmp_path / "vae.safetensors",
            dtype_policy="bfloat16",
        ),
    )

    with pytest.raises(ValueError, match="broken VAE cache"):
        components.load_image_decoder(resources)

    assert events == ["construct", "cleanup"]
