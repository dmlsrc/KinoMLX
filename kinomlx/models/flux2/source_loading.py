"""Klein component source loading and lazy weight binding."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.modeling.artifacts import WeightSource
from kinomlx.modeling.layers import linear_shell
from kinomlx.modeling.loading import (
    WeightBinding,
    WeightTarget,
    bind_weight_targets,
)
from kinomlx.reporting import Reporter
from kinomlx.tokenizers.byte_bpe import QwenByteBPETokenizer

from .layout import pretransposed_weight
from .metadata import TEXT_TAPS, Flux2TextConfig
from .preparation import prepare_weight
from .resources import Flux2Component
from .shapes import text_shapes, transformer_shapes, vae_shapes
from .text_encoder import Flux2TextEncoder
from .transformer import Flux2Transformer
from .vae import Flux2VAE


def _target(model: nn.Module, key: str, shape: tuple[int, ...]) -> WeightTarget:
    parts = key.split(".")
    owner: object = model
    for part in parts[:-1]:
        owner = owner[int(part)] if isinstance(owner, list) else getattr(owner, part)
    return WeightTarget(owner, parts[-1], shape)


def load_tokenizer(component: Flux2Component) -> QwenByteBPETokenizer:
    """Construct a tokenizer from the resource's already validated tables."""
    return QwenByteBPETokenizer(component.tokenizer_tables)


def load_text_encoder(
    files: Sequence[Path], *, reporter: Reporter | None = None
) -> Flux2TextEncoder:
    model = Flux2TextEncoder(Flux2TextConfig())
    targets = {
        key: _target(model, key.removeprefix("model."), shape)
        for key, shape in text_shapes().items()
        if key == "model.embed_tokens.weight"
        or (key.startswith("model.layers.") and int(key.split(".")[2]) < TEXT_TAPS[-1])
    }
    bind_weight_targets(
        tuple(files),
        targets,
        lambda key: (key,),
        component="Klein text",
        phase="load_text_encoder",
        expected_dtype=mx.bfloat16,
        reporter=reporter,
    )
    return model


def transformer_targets(model: Flux2Transformer) -> dict[str, WeightTarget]:
    """Return source shapes and install the measured pretransposed target shells."""
    targets = {}
    for key, shape in transformer_shapes().items():
        if pretransposed_weight(key):
            module = _target(model, key.removesuffix(".weight"), ())
            setattr(module.owner, module.attribute, linear_shell(bias=False, pretransposed=True))
        targets[key] = _target(model, key, shape)
    return targets


def _check_transformer_dtype(dtype: mx.Dtype) -> None:
    if dtype not in {mx.float16, mx.bfloat16}:
        raise ValueError("Klein transformer loading supports only FP16/BF16")


def load_transformer(
    files: Sequence[Path],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> Flux2Transformer:
    """Bind lazily prepared source tensors; denoising evaluates the casts and layouts.

    Bundled Klein sources are BF16; header preflight validates source storage
    before the requested dtype and layout expressions are bound.
    """
    _check_transformer_dtype(dtype)
    model = Flux2Transformer()
    targets = transformer_targets(model)

    def prepare(binding: WeightBinding, value: mx.array) -> mx.array:
        return prepare_weight("transformer", binding.logical_key, value, dtype)

    bind_weight_targets(
        tuple(files),
        targets,
        lambda key: (key,),
        component="Klein transformer",
        phase="load_transformer",
        expected_dtype=mx.bfloat16,
        reporter=reporter,
        prepare=prepare,
    )
    return model


def component_weight_files(component: Flux2Component) -> tuple[Path, ...]:
    """Return the safetensors files a weight component binds from."""
    if not isinstance(component.source, WeightSource):
        raise ValueError(f"Klein {component.kind} has no weight source to load")
    return tuple(file.path for file in component.source.tensor_files)


def load_transformer_component(
    component: Flux2Component,
    *,
    dtype: mx.Dtype,
    reporter: Reporter | None = None,
) -> Flux2Transformer:
    """Bind the selected source transformer with lazy preparation."""
    return load_transformer(component_weight_files(component), dtype=dtype, reporter=reporter)


def load_vae(
    files: Sequence[Path],
    *,
    mode: Literal["encode", "decode", "both"] = "both",
    dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> Flux2VAE:
    if dtype not in {mx.bfloat16, mx.float32}:
        raise ValueError("Klein VAE supports BF16 execution and FP32 diagnostic execution")
    model = Flux2VAE(mode)
    targets = {}
    for key, shape in vae_shapes().items():
        if key == "bn.num_batches_tracked":
            continue
        if mode == "decode" and key.startswith(("encoder.", "quant_conv.")):
            continue
        if mode == "encode" and key.startswith(("decoder.", "post_quant_conv.")):
            continue
        targets[key] = _target(model, key, shape)
    bind_weight_targets(
        tuple(files),
        targets,
        lambda key: (key,),
        component="Klein VAE",
        phase=f"load_vae_{mode}",
        expected_dtype=mx.bfloat16,
        reporter=reporter,
    )
    for target in targets.values():
        value = getattr(target.owner, target.attribute).astype(dtype)
        if len(target.shape) == 4:
            value = mx.contiguous(value.transpose(0, 2, 3, 1))
        setattr(target.owner, target.attribute, value)
    return model
