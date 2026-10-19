# Klein performance and memory

This page covers precision, memory, and the KV variant's reference cache. The memory controls
every model shares, and the High Power energy mode, are in [Performance and
memory](../PERFORMANCE.md).

- [Precision](#precision)
- [Model lifetimes](#model-lifetimes)
- [Reference KV](#reference-kv)
- [Decoder memory](#decoder-memory)

## Precision

The transformer runs in BF16 by default, the bundled weights' own precision.
`--transformer-dtype float16` is an optional speed setting: it narrows the numerical range and
changes rounding, and saves no memory, since both use two bytes per value.

The text encoder always runs in BF16. The decoder runs in BF16 by default, and
`--vae-dtype float32` decodes at FP32 for diagnostic comparisons, though such a run cannot be
restarted; FP16 is not a supported decoder precision. Choosing an FP16 transformer does not change
the text encoder's or the decoder's precision.

## Model lifetimes

The text encoder, reference encoder, transformer, and decoder load one after another, and each is
released before the next. Only the weights used for inference are loaded, and no converted weights
are written to disk.

The transformer is prepared on first use, so its first denoising step includes loading it and
merging any adapters. See [Model files and loading](../WEIGHTS.md#what-loading-costs) for first-use
costs and loading errors.

Adapters and the reference cache add to the denoising working set. Base guidance with
`--cfg-mode batched` runs both branches at once, using more activation memory than the default
`sequential`.

With several seeds, `--seed-mode` chooses whether models reload between seeds; see
[seed modes](../USAGE.md#how-long-models-stay-loaded-across-seeds).

## Reference KV

The `kv` variant is a separate checkpoint. It computes the references' keys and values on the first
step and reuses them on every later step, so editing with references is faster. The first step is
therefore slower than the rest. With no references the variant uses full attention, and a one-step
recipe has no later step to gain.

Reuse is exact for this checkpoint, which was built so that references do not change between
steps. The cache would not make the ordinary distilled model behave the same way.

The speed costs memory. For this 9B model, the cached keys and values come on top of the weights
and the working tensors:

| Reference tokens | KV cache |
| ---: | ---: |
| 1 | 512 KiB |
| 4,096 | 2 GiB |
| 8,192 | 4 GiB |

Reference size and count therefore matter even when the output size does not change. Klein never
shrinks references to fit memory on its own.

Before generating, the log reports the reference token count and the estimated cache size. That
estimate is not a promise that the whole run fits: weights, activations, attention workspace, and
other applications also use memory.

The cache lasts one generation. The next call, even through the same Python runner, extracts it
again, and it is cleared on success, failure, or cancellation. It is not a library of reference
images.

`--recompute-reference-kv` recomputes the cache every step. Adapters whose strength varies by step
or sigma window need it while editing, and it is useful for checking the reuse; it is never
faster.

## Decoder memory

Klein has no VAE tiling: the decoder always decodes the whole image at once. Its attention memory
is bounded, but its convolution memory grows with the image, so very large images can still exceed
memory.
