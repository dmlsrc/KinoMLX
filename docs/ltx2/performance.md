# LTX2 performance and memory

This page helps you estimate a run, fit it in memory, and speed it up. The memory controls every
model shares, such as the allocator cache, and the High Power energy mode are in [Performance and
memory](../PERFORMANCE.md).

- [Run times](#run-times)
- [Run on 32 GB](#run-on-32-gb)
- [Use less transformer memory](#use-less-transformer-memory)
- [Precision](#precision)
- [Sparse video attention](#sparse-video-attention)
- [Attention kernel](#attention-kernel)
- [Adapters and speed](#adapters-and-speed)

## Run times

Four LTX-2.5 runs of one prompt on an M1 Max with 64 GB, macOS 27.0.1, and MLX 0.32.3, at seed
42. They used the defaults plus generated audio: an FP16 transformer with every block loaded, no
adapters, the convolutional decoder, and HEVC output. MLX had a 1 GiB allocator cache and a
60 GiB wired limit, set with the [system override](../PERFORMANCE.md#optional-system-override).

| Phase | 768x448, 5 s | 768x448, 30 s | 1024x576, 20 s | 1024x576, 30 s |
| --- | ---: | ---: | ---: | ---: |
| Prompt encoding | 5.5 s | 5.5 s | 5.5 s | 5.5 s |
| Stage 1, 8 steps at half size | 37 s | 4 min 1 s | 4 min 21 s | 7 min 8 s |
| Latent upscale | 1.2 s | 4.7 s | 4.8 s | 7.1 s |
| Stage 2, 3 steps at full size | 54 s | 8 min 27 s | 10 min 14 s | 19 min 14 s |
| One stage-2 step | 18 s | 2 min 49 s | 3 min 25 s | 6 min 25 s |
| Audio decode | 0.5 s | 2.5 s | 1.7 s | 2.5 s |
| Video decode and encoding | 4.4 s | 29.6 s | 28.0 s | 42.0 s |
| Total | 1 min 45 s | 13 min 12 s | 15 min 16 s | 27 min 22 s |
| Denoising share of the total | 87% | 94% | 95% | 96% |
| Peak MLX memory | 37.4 GiB | 41.1 GiB | 41.8 GiB | 44.4 GiB |
| Peak during prompt encoding | 23.1 GiB | 23.1 GiB | 23.1 GiB | 23.1 GiB |
| Peak during decoding | 6.9 GiB | 8.7 GiB | 13.1 GiB | 13.1 GiB |

Memory across all four runs:

| Memory | Amount |
| --- | ---: |
| Transformer weights, most of every peak | About 35 GiB |
| System memory in use, peak | 48 to 53 GiB |
| Of that, other software | About 9 GiB |
| Swap growth | None |

- Denoising dominates every run, and its cost grows faster than the video: compare the stage-2
  step across sizes.
- The peak comes from stage 2: the transformer's weights plus activations that grow with the
  video. The text encoder is released before the transformer loads.
- The first run at a new size also builds GPU kernels for it, once per machine, which adds a few
  seconds.
- Expect a few percent of variation with the machine's temperature and power state.
- Runs this long can throttle after a few minutes in the Automatic energy mode; choose
  [High Power](../PERFORMANCE.md#energy-mode) if your Mac offers it.

Treat them as estimates, not guarantees: other hardware, a different macOS or MLX, or other GPU
work change them.

## Run on 32 GB

`--stream-transformer` keeps a pool of 8 of the transformer's 48 blocks in memory and streams all
48 through it every step, reading each from disk as it is needed. That makes LTX2 fit a 32 GB
Mac:

```bash
kinomlx --model ltx2 --model-generation 2.5 --stream-transformer \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

Streaming changes only the transformer's memory. Longer clips, larger frames, audio, and decoding
still need room, so start at a modest size.

Its output matches a resident run bit for bit when both use the same
[fusion precision](adapters.md#fusion-precision) and the same execution settings; compiled block
groups, for example, can differ.

Each step reads every block again. While the checkpoint fits in the macOS file cache that is
cheap; once it does not, each step waits on the drive. Adapters add a fusion per reloaded block,
and a large adapter stays loaded for the run:

| Streaming the 22B transformer | Size |
| --- | ---: |
| All 48 blocks, read every step | 37 GB |
| The rank-384 distilled LoRA, loaded for the whole run | 7.6 GB |

`--transformer-resident-blocks N` streams with N blocks loaded instead of 8. More resident blocks
use more memory without much gain, because reads already overlap computation.

### Keep some tensors loaded

`--transformer-retain` keeps chosen tensors loaded across steps while streaming, so they are not
read again each step. Selectors combine with commas:

| Selector | Keeps |
| --- | --- |
| `small` | Norms, biases, and modulation tables |
| `audio-attention` | Audio attention, including the audio-video attention |
| `video-attention` | Video attention |
| `audio-ff` | Audio feed-forward layers |
| `video-ff` | Video feed-forward layers |
| a glob such as `ff.project_out.weight` | Block tensors whose names match |

```bash
kinomlx --model ltx2 --model-generation 2.5 --stream-transformer \
  --transformer-retain 'small,audio-attention,*to_gate_logits.weight,ff.project_out.weight' \
  --transformer-resident-blocks 4 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 512 --height 512 --frames 49 --generate-audio
```

Retention needs streaming, and each selector must match a tensor the run uses; otherwise it is
refused.

This command shows the selectors; it does not reproduce the measurement below. On a 64 GB M1 Max
with 32 GB held back to imitate a smaller Mac, with two adapters at 512x512 and 49 frames, that
selection produced identical latents:

| Streaming | Run time | Stage-1 peak |
| --- | ---: | ---: |
| Default, 8 resident blocks | 88.9 s | 16.4 GiB |
| The selection above, 4 resident blocks | 70.2 s, 21% less | 21.8 GiB |

The gain depends on your drive and memory, so measure a complete run.

Retained tensors are live memory, separate from the allocator cache. Start with default
streaming, add a small selection, and measure.

Repeating the flag replaces its value. In TOML it is `transformer_retain = ["small", ...]` under
`[model_settings]`.

## Use less transformer memory

`--transformer-quantize mxfp8-blocks` packs the transformer's attention and feed-forward weights
into MXFP8 from an ordinary checkpoint:

```bash
kinomlx --model ltx2 --prompt "A painted landscape." --transformer-quantize mxfp8-blocks
```

At 512x512, 49 frames, with audio:

| Transformer | Run peak | Denoising memory | Run time |
| --- | ---: | ---: | ---: |
| Unquantized | About 37 GiB | Reference | Reference |
| `mxfp8-blocks` | About 23 GiB | About 17 GiB less | About 13% longer |

Quantized, the transformer no longer sets the peak; the text encoder does. Quantization changes
the weights, so it changes the video. `--transformer-quantize-exclude-layers` keeps chosen blocks
unquantized.

`mxfp8-blocks-pretranspose` stores the packed weights transposed. It is experimental; measure it
on your machine before relying on it for speed.

## Precision

The transformer runs in FP16 by default. Measured on both LTX-2.3 and LTX-2.5:

| `--transformer-dtype` | Forward error against FP32 | Speed |
| --- | --- | --- |
| `float16` (default) | 3 to 16 times smaller than BF16's | About 6% faster than BF16 |
| `bfloat16` | Reference | Reference |

FP16 has the narrower range, so a step that overflows stops the run with a message naming
`--transformer-dtype bfloat16`.

BF16 and FP16 use the same memory. A shared seed does not make FP16 and BF16 videos comparable
frame for frame, so judge the result as well as the clock.

## Sparse video attention

Video self-attention grows with the square of the video's size. On long or large clips it is
about half of each stage-2 step. `--stage2-sparse-attention` and `--stage1-sparse-attention`
replace it on chosen steps with Sol-Attn, a training-free approximation from NVIDIA's Sana
Sol-Engine. It changes the output, so it is off by default:

```bash
# Sol-Engine's published stage-2 schedule
kinomlx --model ltx2 --prompt "A quiet coastal sunrise, cinematic light" \
  --stage2-sparse-attention auto

# The last four stage-1 steps too, with the first and last layers kept exact
kinomlx --model ltx2 --prompt "A quiet coastal sunrise, cinematic light" \
  --stage1-sparse-attention 5-8=1.5 --stage2-sparse-attention 1.25 \
  --stage1-dense-layers 0,47 --stage2-dense-layers 0,47
```

Each stage takes a schedule of taus, one per step. A larger tau computes fewer key blocks exactly:
faster, and further from exact attention.

| Schedule | Meaning |
| --- | --- |
| `dense` | Every step exact (the default) |
| `auto` | Stage 2 only: tau 1, 1.25, and 1.5 on its three steps |
| `1.5` | Every step at tau 1.5 |
| `1,1.25,1.5` | One tau per step; the count must match the stage's steps |
| `dense,1,1.5` | The first step exact, then tau 1 and 1.5 |
| `5-8=1.5` | Steps 5 to 8, counted from 1, at tau 1.5; the rest exact |
| `2=1,4-5=0.75` | Several ranges, which must not overlap |

The stages have 8 and 3 steps unless a custom schedule changes them, and a schedule that does not
fit is refused before anything loads. `auto` needs the three-step stage 2.

`--stage1-dense-layers` and `--stage2-dense-layers` keep layers 0 to 47 exact on sparse steps, as
numbers and ranges such as `0,46-47`, or `none`. Both default to layer 0, which suffers most from
the approximation.

The saving grows with the clip; at 768x448 and 121 frames it is small. Measured on a stock run
with earlier attention kernels:

| 768x448, 721 frames | Stage 2 | Frames against the exact run |
| --- | ---: | --- |
| Exact attention | 594 s | Reference |
| `--stage2-sparse-attention auto` | 375 s | Same composition, 32.9 dB PSNR |

Judge the result on your material: a 121-frame clip with two adapters came out sharp and stable
but recomposed.

Text, audio, and audio-video attention stay exact, and an FP32 transformer is refused.

A stage-2 restart can change the stage-2 schedule and dense layers; the stage-1 ones are fixed
once stage 1 has run, and a decode restart changes neither.

## Attention kernel

LTX's attention runs on KinoMLX's own FP16 Metal kernel by default:

| Against MLX's attention | Time saved |
| --- | --- |
| One video attention call | 10 to 16% |
| A whole run | A few percent; more on long clips |

In FP16 its video output is bit-identical to MLX's. A BF16 transformer's attention inputs are
converted to FP16 for the kernel, saturating at FP16's largest value, so BF16 does not protect
attention from FP16's range. `--no-dense-attention-kernel` runs every call on MLX's attention
instead, in the transformer's own precision.

## Adapters and speed

Fusing adapters into the weights happens during the first denoising step, so that step is slower
than the rest; the run log shows it inside denoising time. Streaming repeats the fusion whenever a
block reloads, which is why `--adapter-accumulation auto` fuses in FP16 there; see
[fusion precision](adapters.md#fusion-precision).

The first-use weight check also runs inside the first step. Skipping it with
`--no-check-adapter-weights` saves time on a configuration you have already checked; see
[first-use checks](../ADAPTERS.md#first-use-weight-checks).

The [expert tuning arguments](arguments.md#expert-transformer-tuning), such as compile groups and
weight layouts, are for benchmarks; the defaults are the tuned configuration.
