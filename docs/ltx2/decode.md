# Video decoding

After denoising, a VAE decoder turns the final latent into frames. This page covers which decoder
to use, its precision, and how it fits in memory. Upscaling, frame rate, and encoding come after
decoding; see [the output page](output.md).

- [Choose a decoder](#choose-a-decoder)
- [Decode precision](#decode-precision)
- [Memory: tiling and cache](#memory-tiling-and-cache)
- [Sharpen the diffusion decoder](#sharpen-the-diffusion-decoder)
- [Decode again without regenerating](#decode-again-without-regenerating)

## Choose a decoder

| `--video-vae` | What it is | Choose it when |
| --- | --- | --- |
| `conv` (default) | The convolutional decoder of the selected checkpoint | You want speed and the reference look. |
| `diffusion` | LTX-2.5's neighborhood-attention diffusion decoder | You want its own fine texture and can spend the time. |

The diffusion decoder implements a different output prior, so its frames do not match the
convolutional decoder's, and its texture changes with the seed and noise backend. It also decodes
LTX-2.3 latents, because LTX-2.5 kept LTX-2.3's latent format; KinoMLX finds it in a cached
LTX-2.5 pack. `--video-vae-path` selects a decoder file directly and overrides `--video-vae`.

Decode times in FP16 with automatic tiling, on an idle M1 Max with 64 GB:

| Decoder | Video | Decode time | MLX peak |
| --- | --- | ---: | ---: |
| `conv` | 768x448, 121 frames | 3.8 s | 6.9 GiB |
| `diffusion` | 768x448, 121 frames | 12.8 s | 8.6 GiB |
| `diffusion` | 1024x576, 721 frames | About 2 min 20 s | About 34 GiB |

Both decoders keep a large allocator cache where memory allows, so the process uses more memory
than those MLX peaks. Judge the diffusion decoder on your own material: its quality on HDR and
broad content is not yet validated.

The diffusion decoder's fast route runs only in FP16 and on blocks large enough for its kernels;
other blocks take a slower exact path. The run record's `vae_decode` attention receipt says which
ran: its `backend` is `fp16-neighborhood-kernels`, `mlx-sdpa-exact-neighborhood`, or `mixed`, and
`fallback_reason` names why a block took the slow path.

## Decode precision

Both decoders decode in FP16 by default. Each decoded chunk is checked before its frames are
written. If the first one overflows, KinoMLX decodes again in BF16, which has FP32's range at half
its memory, and the run record notes the switch under `vae_decode.overflow_fallback`.

| `--vae-decode-dtype` | Behavior |
| --- | --- |
| `auto` (default) | FP16, with the BF16 fallback above |
| `float16` | FP16 only; an overflow stops the run |
| `bfloat16` | BF16 |
| `float32` | FP32 |

A later chunk that overflows, after frames are written, stops the run and suggests
`--vae-decode-dtype bfloat16`. A run saved with `--save-all-sidecars` can then
[decode again](#decode-again-without-regenerating) without regenerating.

FP16 is much faster:

| Decoder | Video | FP16 | BF16 | FP16 frames against BF16 |
| --- | --- | ---: | ---: | --- |
| `conv` | 768x448, 721 frames | 25.7 s | 54.6 s | About 0.73 levels apart on an 8-bit scale (first tile, mean) |
| `diffusion` | 768x448, 121 frames | 12.8 s | 63.5 s | Not measured |

The diffusion decoder runs its fast Metal kernels only in FP16; BF16 and FP32 take its slower
exact path. Decoder precision is independent of `--transformer-dtype`.

## Memory: tiling and cache

A long or large video can exceed memory if decoded at once, so the decoder splits it into tiles
when needed:

| `--vae-tiling` | Behavior |
| --- | --- |
| `auto` (default) | Choose the tiles and the allocator cache from free memory, preferring temporal tiles to spatial ones |
| `single` | Decode in one pass |
| `custom` | Use `--vae-temporal-tile-frames`, `--vae-temporal-overlap-frames`, `--vae-spatial-tile-pixels`, and `--vae-spatial-overlap-pixels` |

The FP16 diffusion decoder decodes in one pass whenever that fits. The FP16 convolutional
decoder uses temporal tiles of at most 128 frames even when one pass would fit, because longer
tiles measured slower.

`auto` sizes its plan from the memory MLX can use: the GPU limit, including any
`iogpu.wired_limit_mb` setting, minus memory already allocated, and no more than macOS reports as
free or reclaimable. The file cache is not counted as used memory. The log shows that ceiling,
the headroom, and the chosen plan, which the run record stores.

To repeat a plan, read its tile sizes from the run record and pass them with
`--vae-tiling custom`; an empty plan replays as `--vae-tiling single`.

In custom mode, `0` turns off tiling along that axis, and at least one axis must be tiled. In
TOML, giving tile sizes without a mode selects `custom`, and choosing `auto` or `single` clears
any stored sizes.

`--vae-cache-limit-gb` fixes the allocator cache during decoding, instead of choosing it with the
plan; the tiles then shrink to fit beside it. It does not change the cache the other phases use.

A plan the device cannot address, such as a tile larger than its maximum buffer, is refused before
decoding starts.

## Sharpen the diffusion decoder

`--vae-sharpen AMOUNT` makes the LTX-2.5 diffusion decoder draw sharper edges. It sharpens the
input of the decoder's last stage, so the decoder paints the detail itself instead of a filter
amplifying finished pixels and their grain.

```bash
kinomlx --model ltx2 --model-generation 2.5 --video-vae diffusion --vae-sharpen 2 \
  --prompt "A quiet coastal sunrise, cinematic light" --width 1024 --height 576
```

| Option | Default | Effect |
| --- | --- | --- |
| `--vae-sharpen` | `0` (off) | About 1 is subtle; 3 to 4 is strong. The right amount depends on the clip. |
| `--vae-sharpen-radius` | `2` | Blur radius in output pixels, at most 8. Larger radii also sharpen coarser shapes and can outline them. |

Sharpening adds little time and no memory:

| Video | Stock | Sharpened | MLX peak |
| --- | ---: | ---: | --- |
| 1024x576, 121 frames | 22.0 s | 22.1 s | No higher |
| 768x448, 721 frames | 88.7 s | 89.0 to 90.6 s | 27.0 GiB either way |

Sharpening is not part of the reference decoder, and it is refused with the convolutional decoder.
The run record lists both values under `vae_decode.context_sharpening`.

## Decode again without regenerating

Save the latents on the original run with `--save-all-sidecars` or `--save-latents`. A decode
restart then tries another decoder, precision, tiling, or sharpening in seconds to minutes instead
of a full generation:

```bash
kinomlx --restart outputs/source_run.json --video-vae diffusion --vae-sharpen 1 \
  --output outputs/redecoded.mp4
```

See [Restart from saved stages](restart.md).
