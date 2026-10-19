# Krea 2 performance and precision

This page covers the choices that trade speed, memory, and exactness. The memory controls every
model shares, and the High Power energy mode, are in [Performance and memory](../PERFORMANCE.md).

- [Denoising mode](#denoising-mode)
- [Precision](#precision)
- [Decoder memory](#decoder-memory)
- [Model lifetimes](#model-lifetimes)

## Denoising mode

By default the denoiser skips work on padding: it keeps only the real text tokens, drops the
padding and its masks, and stores some weights transposed for faster products. The run record
calls this mode `compact_unmasked_pretransposed`.

Removing padding changes matrix shapes and therefore floating-point rounding, so the image can
differ slightly from one made by the reference implementation with the same seed.
`--reference-denoise` restores the original padded, masked layout, for those comparisons:

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise, cinematic light" --reference-denoise
```

Use the default for normal generation. Both modes take adapters, and neither changes text encoding,
guidance, the scheduler, or decoding. `--no-reference-denoise` overrides a config or environment
that enables reference mode.

## Precision

The transformer runs in BF16 by default. `--transformer-dtype float16` is an optional speed
setting, not a memory saving: both use two bytes per value. FP16 has the narrower range, so it can
change the image and occasionally overflow. `float32` doubles the memory.

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise, cinematic light" \
  --transformer-dtype float16
```

Reference mode refuses an FP16 transformer before loading, because its padding rows can overflow
FP16 and spoil real outputs. Every dtype converts from the original weights, with no intermediate
rounding.

In either mode, a denoising step that produces nonfinite values stops the run with
`Krea2DenoiseError` instead of decoding a broken image. If FP16 overflows, use BF16 or FP32 and
check adapter strengths.

`--vae-dtype` sets the decoder's precision separately: `bfloat16` by default, or `float16` or
`float32`.

## Decoder memory

The decoder decodes the whole image when it fits and tiles it otherwise:

| `--vae-tiling` | Behavior |
| --- | --- |
| `auto` (default) | The whole image if its estimated working set fits, else the largest tiles that fit |
| `single` | The whole image |
| `custom` | `--vae-tile-size` (default 256) and `--vae-tile-stride` (default 192), in pixels of the generated image, multiples of 8 |

With the `upscale2x` decoder, tile sizes still describe the image before doubling. A size or
stride that is not a multiple of 8 is refused only when decoding starts, after denoising, so check
custom values on a small run first.

Tiles blend where they overlap. Tiling bounds memory, while a whole-image decode keeps global
context in the decoder's attention, so prefer it when it fits.

`auto` sizes its plan from the memory MLX can use: the GPU limit, including any
`iogpu.wired_limit_mb` setting, minus memory already allocated, and no more than macOS reports as
free or reclaimable. The file cache is not counted as used, and swap is not counted as available.
The log shows the ceiling, the allocations, the headroom, and the chosen plan.

A decode-only restart loads only the decoder half of the VAE, which lowers the memory needed to
retry decoding.

## Model lifetimes

The text encoder, transformer, and decoder load one after another. In the default seed mode each
is released before the next loads, so the transformer is gone before decoding, which lowers the
peak.

Weights are prepared when first used. The first denoising step therefore includes loading the
transformer and merging any adapters, and the progress bar's first step is slower than the rest.
The decoder's load time includes reading its weights.

With several seeds, `--seed-mode` chooses whether models reload between seeds; see
[seed modes](../USAGE.md#how-long-models-stay-loaded-across-seeds).
