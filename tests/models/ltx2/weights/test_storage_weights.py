"""FP8 conversion, quant routing, and single-file storage tests."""

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import load_weights, read_dtypes, read_header
from kinomlx.models.ltx2.weights.dtypes import normalize_transformer_dtypes
from kinomlx.models.ltx2.weights.schema import (
    QUANT_KEY_PREFIX,
    TRANSFORMER_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
from kinomlx.models.ltx2.weights.tensors import (
    checkpoint_has_fp8_tensors,
    fp8_scale_companions,
    iter_fp8_checkpoint_weights,
    quant_defaults,
    quant_mode_for_key,
    quant_pretransposed,
)


def _write_raw_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, dtype, shape, data in tensors:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs))


def test_checkpoint_loader_handles_resolved_huggingface_blob_without_extension(
    tmp_path: Path,
) -> None:
    blob = tmp_path / "b33b7fe4bbfe084f"
    staged = blob.with_suffix(".safetensors")
    mx.save_safetensors(str(staged), {"tensor": mx.array([1.0, 2.0])})
    staged.rename(blob)

    loaded = load_weights(blob)

    assert mx.array_equal(loaded["tensor"], mx.array([1.0, 2.0])).item()


def test_header_reader_reports_raw_fp8_dtype_without_tensor_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fp8.safetensors"
    _write_raw_safetensors(path, [("weight", "F8_E4M3", [1], bytes([0x38]))])
    assert read_dtypes(path) == {"weight": "F8_E4M3"}
    assert read_header(path)["weight"]["shape"] == [1]
    assert checkpoint_has_fp8_tensors(path)


def test_header_reader_rejects_truncated_prefix_and_header(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix.safetensors"
    prefix.write_bytes(b"short")
    with pytest.raises(ValueError, match="length prefix"):
        read_header(prefix)
    header = tmp_path / "header.safetensors"
    header.write_bytes(struct.pack("<Q", 10) + b"{}")
    with pytest.raises(ValueError, match="truncated safetensors header"):
        read_header(header)


def test_fp8_companion_classification_is_scoped_to_real_fp8_weights() -> None:
    dtypes = {
        "layer.weight": "F8_E4M3",
        "layer.weight_scale": "F32",
        "layer.input_scale": "F32",
        "layer.comfy_quant": "U8",
        "unrelated.comfy_quant": "U8",
        "scale_shift_table": "F32",
    }
    fp8, weight_scales, input_scales, tags = fp8_scale_companions(dtypes)
    assert fp8 == {"layer.weight"}
    assert weight_scales == {"layer.weight_scale"}
    assert input_scales == {"layer.input_scale"}
    assert tags == {"layer.comfy_quant"}


def test_fp8_iterator_applies_scale_and_drops_activation_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fp8.safetensors"
    _write_raw_safetensors(
        path,
        [
            ("layer.weight", "F8_E4M3", [1], bytes([0x38])),
            ("layer.weight_scale", "F32", [], struct.pack("<f", 2.0)),
            ("layer.input_scale", "F32", [1], struct.pack("<f", 7.0)),
            (
                "layer.comfy_quant",
                "U8",
                [26],
                b'{"format":"float8_e4m3fn"}',
            ),
            ("ordinary", "F32", [1], struct.pack("<f", 3.0)),
        ],
    )
    converted = dict(iter_fp8_checkpoint_weights(path, mx.float16))
    assert set(converted) == {"layer.weight", "ordinary"}
    assert converted["layer.weight"].dtype == mx.float16
    assert float(converted["layer.weight"].item()) == 2.0
    assert float(converted["ordinary"].item()) == 3.0


def test_fp8_iterator_rejects_e5m2_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "e5m2.safetensors"
    _write_raw_safetensors(path, [("weight", "F8_E5M2", [1], bytes([0]))])
    with pytest.raises(ValueError, match="F8_E5M2"):
        dict(iter_fp8_checkpoint_weights(path, mx.bfloat16))


def test_fp8_iterator_rejects_non_scalar_weight_scale(tmp_path: Path) -> None:
    path = tmp_path / "vector-scale.safetensors"
    _write_raw_safetensors(
        path,
        [
            ("layer.weight", "F8_E4M3", [2, 2], bytes([0x38] * 4)),
            ("layer.weight_scale", "F32", [2], struct.pack("<ff", 2.0, 3.0)),
        ],
    )

    with pytest.raises(ValueError, match="layer.weight_scale.*scalar"):
        dict(iter_fp8_checkpoint_weights(path, mx.bfloat16))


def test_fp8_companion_classifier_rejects_scale_for_non_fp8_parent() -> None:
    with pytest.raises(ValueError, match="ordinary.weight_scale.*non-FP8"):
        fp8_scale_companions(
            {
                "quantized.weight": "F8_E4M3",
                "ordinary.weight": "BF16",
                "ordinary.weight_scale": "F32",
            }
        )


def test_transformer_weights_dtype_normalization_is_canonical() -> None:
    assert normalize_transformer_dtypes(
        "bfloat16",
        mx.bfloat16,
        None,
    ) == (None, None, None)
    assert normalize_transformer_dtypes(
        "float16",
        mx.float16,
        mx.float16,
    ) == (mx.float16, None, None)
    assert normalize_transformer_dtypes(
        "bfloat16",
        mx.float16,
        None,
    ) == (None, mx.float16, None)
    with pytest.raises(ValueError, match="conflicts with transformer dtype"):
        normalize_transformer_dtypes(
            "float16",
            mx.bfloat16,
            None,
        )


def test_quant_routing_covers_whole_block_and_targeted_ff_modes() -> None:
    attention = "transformer_blocks.0.attn1.to_q.weight"
    ff = "transformer_blocks.3.ff.project_out.weight"
    assert (
        quant_mode_for_key(
            attention,
            transformer_quantize=TRANSFORMER_QUANTIZE_MXFP8_BLOCKS,
            video_ff_quantize_specs=(),
            video_ff_quantize_layers=(),
        )
        == "mxfp8"
    )
    assert (
        quant_mode_for_key(
            ff,
            transformer_quantize="off",
            video_ff_quantize_specs=(("project_out", "nvfp4"),),
            video_ff_quantize_layers=(3,),
        )
        == "nvfp4"
    )
    assert (
        quant_mode_for_key(
            ff,
            transformer_quantize="off",
            video_ff_quantize_specs=(("project_out", "nvfp4"),),
            video_ff_quantize_layers=(2,),
        )
        is None
    )
    assert quant_pretransposed(TRANSFORMER_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE)
    assert quant_defaults("mxfp8", None, None) == (32, 8)
    assert quant_defaults("affine", 128, 6) == (128, 6)
    with pytest.raises(ValueError, match="must be positive"):
        quant_defaults("affine", 0, 4)
    with pytest.raises(ValueError, match="must be positive"):
        quant_defaults("affine", 64, 0)


def test_route_fuses_adapters_before_packing_quantized_targets(tmp_path: Path) -> None:
    from kinomlx.adapters.fusion import fuse_weight
    from kinomlx.adapters.lora import LoRAConfig
    from kinomlx.io.safetensors import load_weights
    from kinomlx.models.ltx2.adapters.loading import community_adapter_operations
    from kinomlx.models.ltx2.preparation import prepare_transformer_source

    path = tmp_path / "source.safetensors"
    key = "model.diffusion_model.transformer_blocks.0.ff.project_out.weight"
    weight = (mx.arange(32 * 32, dtype=mx.float32).reshape(32, 32) / 100).astype(mx.bfloat16)
    mx.save_safetensors(str(path), {key: weight})
    adapter = tmp_path / "adapter.safetensors"
    prefix = "diffusion_model.transformer_blocks.0.ff.net.2"
    mx.save_safetensors(
        str(adapter),
        {
            f"{prefix}.lora_A.weight": mx.full((2, 32), 0.25, dtype=mx.float32),
            f"{prefix}.lora_B.weight": mx.full((32, 2), 0.5, dtype=mx.float32),
        },
    )
    base = "transformer_blocks.0.ff.project_out"
    operations = community_adapter_operations(
        [LoRAConfig(adapter, strength=0.8)], target_shapes={base: (32, 32)}, include_audio=False
    )
    options = {
        "include_audio": False,
        "transformer_dtype": mx.bfloat16,
        "video_ff_layout_specs": (),
        "video_ff_layout_layers": (),
        "video_attn_layout_specs": (),
        "video_attn_layout_layers": (),
        "video_ff_quantize_specs": (("project_out", "mxfp8"),),
        "video_ff_quantize_layers": (0,),
    }

    def convert_all(preparation):
        raw = load_weights(path)
        packed = {}
        for item in preparation.sources:
            if item.convert:
                packed.update(preparation.convert(item, raw[item.source_key]).tensors)
        return packed

    fused_pack = convert_all(
        prepare_transformer_source(
            path,
            adjust_quantized=lambda name, value: fuse_weight(value, operations[name]),
            **options,
        )
    )
    plain_pack = convert_all(prepare_transformer_source(path, **options))

    quant_base = f"{QUANT_KEY_PREFIX}{base}"
    assert set(fused_pack) == set(plain_pack) == {f"{quant_base}.weight", f"{quant_base}.scales"}
    expected_fused = mx.quantize(
        fuse_weight(weight, operations[f"{base}.weight"]), 32, 8, mode="mxfp8"
    )
    expected_plain = mx.quantize(weight, 32, 8, mode="mxfp8")
    for index, part in enumerate(("weight", "scales")):
        assert mx.array_equal(fused_pack[f"{quant_base}.{part}"], expected_fused[index]).item()
        assert mx.array_equal(plain_pack[f"{quant_base}.{part}"], expected_plain[index]).item()
    assert not mx.array_equal(
        fused_pack[f"{quant_base}.weight"], plain_pack[f"{quant_base}.weight"]
    ).item()


@pytest.mark.parametrize("mode", ["mxfp8-blocks", "mxfp8-blocks-pretranspose"])
def test_excluded_blocks_keep_fp16_and_skip_prepack_adapter_fusion(tmp_path, mode):
    from kinomlx.models.ltx2.preparation import prepare_transformer_source

    weight = mx.ones((32, 64), dtype=mx.bfloat16)
    keys = [f"transformer_blocks.{layer}.ff.project_out.weight" for layer in (0, 4, 47)]
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(path), dict.fromkeys(keys, weight))
    adjusted = []

    def adjust(key, value):
        adjusted.append(key)
        return value + 1

    preparation = prepare_transformer_source(
        path,
        source_bindings={key: key for key in keys},
        transformer_dtype=mx.float16,
        include_audio=False,
        video_ff_layout_specs=(),
        video_ff_layout_layers=(),
        video_attn_layout_specs=(),
        video_attn_layout_layers=(),
        transformer_quantize=mode,
        transformer_quantize_exclude_layers=(0, 1, 2, 3, 47),
        adjust_quantized=adjust,
    )
    result = {}
    for item in preparation.sources:
        if item.convert:
            result.update(preparation.convert(item, weight).tensors)
    assert adjusted == [keys[1]]
    for key in (keys[0], keys[2]):
        assert result[key].dtype == mx.float16
        assert mx.array_equal(result[key], weight.astype(mx.float16)).item()
    packed = f"{QUANT_KEY_PREFIX}{keys[1]}"
    assert result[packed].dtype == mx.uint32
    assert keys[1] not in result
    assert result[packed].shape == ((64, 8) if mode.endswith("pretranspose") else (32, 16))
