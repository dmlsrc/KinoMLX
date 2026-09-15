"""Synthetic source-weight binding, adapter fusion, and block-streaming tests."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from kinomlx.io.safetensors import load_weights
from kinomlx.models.ltx2.weights.loading import bind_transformer_weights
from kinomlx.models.ltx2.weights.schema import (
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
)
from kinomlx.models.ltx2.weights.streaming import TransformerBlockStreamer
from tests.models.ltx2._source_weights import transformer_weights


def _transformer_options() -> dict[str, object]:
    return {
        "include_audio": False,
        "video_ff_layout_specs": (("project_out", "pretranspose"),),
        "video_ff_layout_layers": (0,),
        "video_attn_layout_specs": (),
        "video_attn_layout_layers": (),
    }


class _ProjectIn(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)


class _FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.project_in = _ProjectIn(width)
        self.project_out = nn.Linear(width, width)
        self._project_in_weight_t = None
        self._project_out_weight_t = None


class _Block(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.ff = _FeedForward(width)
        self.idx = -1


class _Transformer(nn.Module):
    def __init__(self, blocks: int = 1, width: int = 32, *, bias: bool = True) -> None:
        super().__init__()
        self.transformer_blocks = [_Block(width) for _ in range(blocks)]
        self.width = width
        self.has_bias = bias

    def expected_parameter_shapes(self, *, include_audio):
        result = {}
        for index in range(len(self.transformer_blocks)):
            base = f"transformer_blocks.{index}.ff.project_out"
            result[f"{base}.weight"] = (self.width, self.width)
            if self.has_bias:
                result[f"{base}.bias"] = (self.width,)
        return result


def test_weights_binders_require_graph_introspection_by_default(tmp_path: Path) -> None:
    cache_file = tmp_path / "partial-fixture.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {"transformer_blocks.0.ff.project_out.weight": mx.eye(32, dtype=mx.bfloat16)},
    )

    with pytest.raises(TypeError, match="expected_parameter_shapes"):
        bind_transformer_weights(object(), load_weights(cache_file))
    with pytest.raises(TypeError, match="expected_parameter_shapes"):
        TransformerBlockStreamer(
            weight_loader=lambda _index: load_weights(cache_file), expected_model=object()
        )


def test_full_loader_restores_base_linear_after_quantized_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "quant.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.eye(
                32, dtype=mx.bfloat16
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.zeros(
                (32,), dtype=mx.bfloat16
            ),
        },
    )
    quantized = transformer_weights(
        checkpoint,
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
        video_ff_quantize_layers=(0,),
        **_transformer_options(),
    )
    unquantized = transformer_weights(
        checkpoint,
        **_transformer_options(),
    )
    model = _Transformer()
    bind_transformer_weights(
        model,
        quantized,
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
    )
    assert isinstance(
        model.transformer_blocks[0].ff.project_out,
        nn.QuantizedLinear,
    )
    bind_transformer_weights(model, unquantized)
    assert isinstance(model.transformer_blocks[0].ff.project_out, nn.Linear)


def test_full_loader_clears_stale_layout_before_normal_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "layout.safetensors"
    source_weight = mx.arange(32 * 32).reshape(32, 32).astype(mx.bfloat16)
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": (source_weight),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.zeros(
                (32,), dtype=mx.bfloat16
            ),
        },
    )
    layout = transformer_weights(
        checkpoint,
        **_transformer_options(),
    )
    normal_options = {
        **_transformer_options(),
        "video_ff_layout_specs": (),
        "video_ff_layout_layers": (),
    }
    normal = transformer_weights(
        checkpoint,
        **normal_options,
    )
    model = _Transformer()

    bind_transformer_weights(model, layout)
    feed_forward = model.transformer_blocks[0].ff
    assert feed_forward._project_out_weight_t is not None
    assert "weight" not in feed_forward.project_out

    bind_transformer_weights(model, normal)
    assert feed_forward._project_out_weight_t is None
    assert mx.array_equal(feed_forward.project_out.weight, source_weight).item()


def test_quantized_weights_requires_weight_and_scales(tmp_path: Path) -> None:
    cache_file = tmp_path / "incomplete.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {
            f"{QUANT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight": (
                mx.zeros((32, 1), dtype=mx.uint32)
            )
        },
    )
    with pytest.raises(ValueError, match="missing scales"):
        bind_transformer_weights(
            _Transformer(),
            load_weights(cache_file),
            video_ff_quantize_specs=(("project_out", "mxfp8"),),
        )


def test_full_loader_rejects_weights_missing_a_model_block(tmp_path: Path) -> None:
    cache_file = tmp_path / "partial.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {
            "transformer_blocks.0.ff.project_out.weight": mx.eye(32, dtype=mx.bfloat16),
            "transformer_blocks.0.ff.project_out.bias": mx.zeros((32,)),
        },
    )

    with pytest.raises(
        ValueError,
        match=r"parameter graph mismatch: missing.*transformer_blocks\.1\.ff\.project_out",
    ):
        bind_transformer_weights(
            _Transformer(blocks=2),
            load_weights(cache_file),
        )


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([0.0])
        self.idx = -1


def test_block_streamer_reloads_evicted_weights_from_weights(tmp_path: Path) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {
            "transformer_blocks.0.weight": mx.array([1.0]),
            "transformer_blocks.1.weight": mx.array([2.0]),
        },
    )
    streamer = TransformerBlockStreamer(weight_loader=lambda _index: load_weights(cache_file))
    block_zero = streamer.bind(_TinyBlock(), 0, evict_block_idx=1)
    block_one = streamer.bind(_TinyBlock(), 1, evict_block_idx=0)
    assert float(block_zero.weight.item()) == 1.0
    assert float(block_one.weight.item()) == 2.0
    assert block_one.idx == 1
    streamer.close()


def test_block_streamer_clears_layout_before_normal_rebind(
    tmp_path: Path,
) -> None:
    source_weight = mx.arange(32 * 32).reshape(32, 32).astype(mx.bfloat16)
    layout_cache = tmp_path / "layout.safetensors"
    normal_cache = tmp_path / "normal.safetensors"
    mx.save_safetensors(
        str(layout_cache),
        {
            f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t": (
                mx.contiguous(source_weight.T)
            )
        },
    )
    mx.save_safetensors(
        str(normal_cache),
        {"transformer_blocks.0.ff.project_out.weight": source_weight},
    )
    block = _Block()
    layout_streamer = TransformerBlockStreamer(
        weight_loader=lambda _index: load_weights(layout_cache)
    )
    normal_streamer = TransformerBlockStreamer(
        weight_loader=lambda _index: load_weights(normal_cache)
    )

    layout_streamer.bind(block, 0)
    assert block.ff._project_out_weight_t is not None
    assert "weight" not in block.ff.project_out

    normal_streamer.bind(block, 0)
    assert block.ff._project_out_weight_t is None
    assert mx.array_equal(block.ff.project_out.weight, source_weight).item()

    layout_streamer.close()
    normal_streamer.close()


@pytest.mark.parametrize("packed_layers", [(1,), (0, 2)])
def test_mixed_block_binding_keeps_dense_and_packed_weights(packed_layers):
    model = _Transformer(blocks=3, bias=False)
    weight = mx.eye(32, dtype=mx.float16)
    packed, scales = mx.quantize(weight, 32, 8, mode="mxfp8")
    weights = {}
    for layer in range(3):
        base = f"transformer_blocks.{layer}.ff.project_out"
        if layer in packed_layers:
            weights[f"{QUANT_KEY_PREFIX}{base}.weight"] = packed
            weights[f"{QUANT_KEY_PREFIX}{base}.scales"] = scales
        else:
            weights[f"{base}.weight"] = weight
    bind_transformer_weights(model, weights, transformer_quantize="mxfp8-blocks")
    x = mx.ones((1, 32), dtype=mx.float16)
    for layer, block in enumerate(model.transformer_blocks):
        linear = block.ff.project_out
        assert isinstance(linear, nn.QuantizedLinear) == (layer in packed_layers)
        # Ignore constructor bias in the dense synthetic modules.
        if "bias" in linear:
            del linear.bias
        assert mx.array_equal(linear(x), x).item()


def test_streamed_adapter_checks_only_consumed_targets_once_per_profile():
    from kinomlx.adapters.validation import AdapterWeightError
    from kinomlx.models.ltx2.adapters.loading import streamed_adapter_check
    from kinomlx.reporting import NullReporter

    block = _Block()
    base = "transformer_blocks.7.ff.project_out"
    check = streamed_adapter_check(frozenset({base}), enabled=True, reporter=NullReporter())
    block.ff.project_out.weight = mx.full((32, 32), float("inf"), dtype=mx.float16)
    check(((7, block),), lambda _: False)
    with pytest.raises(AdapterWeightError, match="transformer_blocks.7"):
        check(((7, block),), lambda _: True)
    block.ff.project_out.weight = mx.eye(32, dtype=mx.float16)
    check(((7, block),), lambda _: True)
    block.ff.project_out.weight = mx.full((32, 32), float("inf"), dtype=mx.float16)
    check(((7, block),), lambda _: True)
    fresh = streamed_adapter_check(frozenset({base}), enabled=True, reporter=NullReporter())
    with pytest.raises(AdapterWeightError):
        fresh(((7, block),), lambda _: True)
    disabled = streamed_adapter_check(frozenset({base}), enabled=False, reporter=NullReporter())
    disabled(((7, block),), lambda _: True)


@pytest.mark.parametrize("strength", [0.25, 0.5])
def test_streamed_source_reapplies_adapter_without_accumulating(tmp_path, strength):
    from kinomlx.adapters.formats import AdapterOperation
    from kinomlx.adapters.fusion import WeightedAdapterOperation, fuse_weight
    from kinomlx.models.ltx2.preparation import prepare_transformer_source

    key = "transformer_blocks.0.ff.project_out.weight"
    weight = mx.eye(32, dtype=mx.bfloat16)
    path = tmp_path / "source.safetensors"
    mx.save_safetensors(str(path), {key: weight})
    operation = AdapterOperation(
        key.removesuffix(".weight"),
        "lora",
        "weight",
        {"down": mx.ones((1, 32)), "up": mx.ones((32, 1))},
        (),
        alpha=None,
    )
    ops = (WeightedAdapterOperation(operation, strength, "0"),)
    preparation = prepare_transformer_source(
        path,
        source_bindings={key: key},
        transformer_dtype=mx.float16,
        include_audio=False,
        video_ff_layout_specs=(("project_out", "pretranspose"),),
        video_ff_layout_layers=(0,),
        video_attn_layout_specs=(),
        video_attn_layout_layers=(),
        adjust_dense=lambda _, value: fuse_weight(value, ops),
    )
    item = next(item for item in preparation.sources if item.convert)
    first = dict(preparation.convert(item, weight).tensors)
    reloaded = dict(preparation.convert(item, weight).tensors)
    target = "__layout__.transformer_blocks.0.ff.project_out.weight_t"
    expected = fuse_weight(weight.astype(mx.float16), ops).T
    assert mx.array_equal(first[target], expected).item()
    assert mx.array_equal(reloaded[target], expected).item()


def test_streamed_adapter_check_reads_private_feed_forward_layout():
    from kinomlx.adapters.validation import AdapterWeightError
    from kinomlx.models.ltx2.adapters.loading import streamed_adapter_check
    from kinomlx.models.ltx2.weights.layout import install_block_layout_weight
    from kinomlx.reporting import NullReporter

    block = _Block()
    install_block_layout_weight(
        block, "ff.project_out.weight_t", mx.full((32, 32), float("inf"), dtype=mx.float16)
    )
    check = streamed_adapter_check(
        frozenset({"transformer_blocks.7.ff.project_out"}), enabled=True, reporter=NullReporter()
    )
    with pytest.raises(AdapterWeightError, match="ff.project_out"):
        check(((7, block),), lambda _: True)


def test_streamed_source_windows_reuse_open_and_preserve_block_eviction(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kinomlx.io.safetensors import SafetensorsBinding
    from kinomlx.models.ltx2 import source_loading
    from kinomlx.models.ltx2.resources import _weight_policy
    from kinomlx.models.ltx2.settings import LTX2Settings

    source = tmp_path / "blocks.safetensors"
    weights = {
        f"transformer_blocks.{index}.norm.weight": mx.full((2,), index + 1, dtype=mx.bfloat16)
        for index in range(4)
    }
    mx.save_safetensors(str(source), weights)
    bindings = tuple(SafetensorsBinding(source, key, key) for key in weights)
    monkeypatch.setattr(source_loading, "resolve_transformer_bindings", lambda *a, **k: bindings)
    calls = []

    def read(path):
        calls.append(path)
        return load_weights(path)

    monkeypatch.setattr(source_loading, "load_weights", read)
    resources = SimpleNamespace(
        require=lambda _kind: SimpleNamespace(source_path=source),
        transformer_config=object(),
        dtype_policy=SimpleNamespace(transformer=mx.float16),
        weight_policy=_weight_policy(LTX2Settings(transformer_resident_blocks=2).resolve_presets()),
    )
    provider = source_loading.transformer_source_loader(resources)
    streamer = TransformerBlockStreamer(weight_loader=provider.load)
    slot = nn.Module()
    slot.norm = nn.LayerNorm(2, bias=False)
    previous = None
    for _ in range(2):
        for index in range(4):
            streamer.bind(slot, index, evict_block_idx=previous)
            mx.eval(slot.parameters())
            assert mx.array_equal(slot.norm.weight, mx.full((2,), index + 1)).item()
            previous = index
    # Initial lazy inventory, then two window reloads for the second traversal.
    assert len(calls) == 3
    streamer.close()


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("quantize", ["off", "mxfp8-blocks"])
def test_source_half_admission_excludes_packed_targets_and_runs_once(
    tmp_path, monkeypatch, quantize, retained
):
    from types import SimpleNamespace

    from kinomlx.adapters.formats import parse_adapter
    from kinomlx.adapters.fusion import WeightedAdapterOperation
    from kinomlx.models.ltx2 import source_loading
    from kinomlx.models.ltx2.metadata import TransformerTensorBinding
    from kinomlx.models.ltx2.resources import _weight_policy
    from kinomlx.models.ltx2.settings import LTX2Settings

    key = "transformer_blocks.0.ff.project_out.weight"
    source = tmp_path / "source.safetensors"
    mx.save_safetensors(str(source), {key: mx.zeros((32, 32), dtype=mx.bfloat16)})
    binding = TransformerTensorBinding(key, key, (32, 32), "BF16")
    monkeypatch.setattr(source_loading, "resolve_transformer_bindings", lambda *a, **k: (binding,))
    monkeypatch.setattr(
        source_loading, "transformer_parameter_shapes", lambda *a, **k: {key: (32, 32)}
    )
    (op,) = parse_adapter(
        {
            "w.lora_A.weight": mx.full((1, 32), 1e-5),
            "w.lora_B.weight": mx.full((32, 1), 1e5),
        }
    ).operations
    operations = {key: (WeightedAdapterOperation(op, 1.0, "a"),)}
    monkeypatch.setattr(source_loading, "community_adapter_operations", lambda *a, **k: operations)
    calls = []
    admit = source_loading.half_accumulation_fallbacks

    def record(selected):
        calls.append(set(selected))
        return admit(selected)

    monkeypatch.setattr(source_loading, "half_accumulation_fallbacks", record)
    settings = LTX2Settings(
        transformer_dtype="float16",
        transformer_resident_blocks=1,
        transformer_quantize=quantize,
        transformer_retain=("ff.project_out.weight",) if retained else (),
    )
    resources = SimpleNamespace(
        require=lambda _kind: SimpleNamespace(source_path=source),
        transformer_config=object(),
        dtype_policy=SimpleNamespace(transformer=mx.float16),
        weight_policy=_weight_policy(settings),
    )
    provider = source_loading.transformer_source_loader(resources, adapters=(object(),))
    for _ in range(2):
        weights = provider.load(0)
        mx.eval(*weights.values())
        assert all(mx.all(mx.isfinite(value)).item() for value in weights.values())
        if quantize == "off":
            assert len(weights) == 1
            assert mx.all(next(iter(weights.values())) == 1).item()
    assert calls == [{key} if quantize == "off" else set()]


@pytest.mark.parametrize("retained", [(), ("small",)])
def test_retained_sources_reuse_arrays_and_skip_reload_conversion(tmp_path, monkeypatch, retained):
    from types import SimpleNamespace

    from kinomlx.models.ltx2 import source_loading
    from kinomlx.models.ltx2.metadata import TransformerTensorBinding
    from kinomlx.models.ltx2.resources import _weight_policy
    from kinomlx.models.ltx2.settings import LTX2Settings

    source = tmp_path / "retained.safetensors"
    values = {f"transformer_blocks.{i}.norm.weight": mx.ones((2,)) for i in range(2)}
    mx.save_safetensors(str(source), values)
    bindings = tuple(TransformerTensorBinding(k, k, (2,), "F32") for k in values)
    monkeypatch.setattr(source_loading, "resolve_transformer_bindings", lambda *a, **k: bindings)
    calls = []
    read = source_loading.load_weights

    def counted(path):
        calls.append(path)
        return read(path)

    monkeypatch.setattr(source_loading, "load_weights", counted)
    resources = SimpleNamespace(
        require=lambda _kind: SimpleNamespace(source_path=source),
        transformer_config=object(),
        dtype_policy=SimpleNamespace(transformer=mx.float32),
        weight_policy=_weight_policy(
            LTX2Settings(
                transformer_resident_blocks=1,
                transformer_retain=retained,
            )
        ),
    )
    provider = source_loading.transformer_source_loader(resources)
    initial = provider.load(None)
    mx.eval(initial)
    for _ in range(2):
        for i in range(2):
            weights = provider.load(i)
            key = f"transformer_blocks.{i}.norm.weight"
            assert set(weights) == {key}
            assert (weights[key] is initial[key]) == bool(retained)
    assert len(calls) == (1 if retained else 5)
