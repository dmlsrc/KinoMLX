"""FourierFT coefficient expansion using MLX FFTs."""

import mlx.core as mx

from .formats import AdapterOperation
from .permutation import torch_cpu_permutation_prefix


def fourier_delta(operation: AdapterOperation, base: mx.array) -> mx.array:
    options = operation.fourier_options
    if options is None:
        raise ValueError("FourierFT requires adapter_config.json frequency locations and scale")
    if base.ndim != 2 or operation.part != "weight":
        raise ValueError("FourierFT requires a Linear weight target")
    spectrum = operation.factors["spectrum"]
    if spectrum.shape != (options.frequencies,) or options.frequencies > base.size:
        raise ValueError("FourierFT frequency count does not fit its Linear target")
    indices = mx.array(torch_cpu_permutation_prefix(base.size, options.frequencies, options.seed))
    dense = mx.zeros((base.size,), dtype=mx.float32).at[indices].add(spectrum.astype(mx.float32))
    return mx.fft.ifft2(dense.reshape(base.shape)).real * operation.scalar
