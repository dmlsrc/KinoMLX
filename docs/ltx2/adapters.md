# Adapters with LTX2

LTX2 takes ordinary LoRA adapters, such as style, motion, and character LoRAs, for either
generation. Because it denoises in two stages, each adapter can have its own strength in each
stage. The [shared adapter guide](../ADAPTERS.md) covers formats and scaling; this page covers
what is specific to LTX2.

- [Add an adapter](#add-an-adapter)
- [Strength per stage](#strength-per-stage)
- [Exclude parts of the model](#exclude-parts-of-the-model)
- [Run a dev transformer](#run-a-dev-transformer)
- [Check what an adapter changed](#check-what-an-adapter-changed)
- [Fusion precision](#fusion-precision)
- [Config files and Python](#config-files-and-python)

## Add an adapter

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" \
  --adapter style.safetensors:0.8
```

Repeat `--adapter` to stack several. `--lora` is the same option. Add any trigger words the
author lists to your prompt; KinoMLX does not insert them.

LTX applies an adapter at the strength you give, with no `alpha/r` factor: `W = W_base + strength *
(B @ A)`. That matches Lightricks' loader and the community LoRAs made for it. The one exception is
a PEFT export with its `adapter_config.json`, which applies the scale its configuration declares;
see [scaling](../ADAPTERS.md#ltx2-direct-strength-except-for-peft-exports).

Strengths must be finite. Negative values subtract the adapter, and very large ones can
overwhelm the model or overflow an FP16 weight. Start from the author's recommendation.

## Strength per stage

One number sets both stages. Two numbers set stage 1, then stage 2, as absolute strengths, and
`0` skips the adapter in that stage:

```bash
--adapter style.safetensors:0.8           # 0.8 in both stages
--adapter style.safetensors:0.25:0.5      # 0.25 in stage 1, 0.5 in stage 2
--adapter detail.safetensors:0:0.8        # stage 2 only
```

Separate flags do the same for the latest `--adapter`; use whichever reads better in a long
command:

```bash
--adapter style.safetensors --adapter-strength 0.8
--adapter style.safetensors --adapter-stage1-strength 0.25 --adapter-stage2-strength 0.5
```

A stage strength replaces the common strength for that stage; it does not multiply it. A flag
affects only its own adapter, and setting a member twice, even through an alias or the shorthand,
is an error.

Stage 1 sets composition and motion at half size; stage 2 refines detail at full size, so the
two strengths control different things.

When the two stages use different adapter strengths, stage 2 starts from a fresh copy of the
base transformer, so stage-1 changes never leak into stage 2.

## Exclude parts of the model

`--adapter-exclude` leaves parts of the transformer unchanged by the latest adapter. A visual
style adapter, for example, can leave the audio branch alone:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" --generate-audio \
  --adapter style.safetensors:0.8 --adapter-exclude audio,cross
```

| Category | Leaves unchanged |
| --- | --- |
| `video` | The video branch |
| `audio` | The audio branch |
| `cross` | The attention between audio and video |
| `attn` | Attention projections (`to_q`, `to_k`, `to_v`, `to_out`) |
| `gate` | Attention gates (`to_gate_logits`) |
| `ff` | Feed-forward layers |
| `adaln` | The timestep modulation layers |
| `cross_control` | Text and audio-video attention plus modulation: `adaln`, `attn2`, `audio_attn2`, and `cross` |
| `distill_control` | `cross`, `gate`, and every modulation category |

Finer categories name single modules or projections:

- Modules: `attn1` (video self-attention), `attn2` (video text attention), `audio_attn1`,
  `audio_attn2`, `video_to_audio_attn`, `audio_to_video_attn`, `audio_ff`.
- Projections: `to_q`, `to_k`, `to_v`, `to_out`, `to_gate_logits`, `project_in`, `project_out`.
- Modulation: `prompt_adaln`, `scale_shift`, `prompt_scale_shift`, `gate_adaln`, and `av_ca` for
  the audio-video attention modulation.

Separate several categories with commas. An excluded target still counts in the coverage total,
so the `Targets` and `Coverage` lines always describe the same set.

## Run a dev transformer

KinoMLX runs only the distilled recipe, and a dev transformer was not distilled for it.
Lightricks' recipe pairs the dev transformer with its official distilled LoRA, at two-stage
strengths of 0.25 and 0.5. KinoMLX does not recognize that LoRA by name, so pass both strengths:

```bash
kinomlx --model ltx2 --model-generation 2.3 \
  --transformer-path /models/ltx23/ltx-2.3-22b-dev.safetensors \
  --adapter /models/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors:0.25:0.5 \
  --adapter /models/loras/style.safetensors:0.3:0.75 --adapter-exclude audio,cross \
  --prompt "A quiet coastal sunrise with a restrained visual treatment" \
  --width 768 --height 448 --frames 121
```

The LTX-2.3 checkpoint still supplies the VAEs and audio; only the transformer is replaced. The
LTX-2.5 equivalent uses the pack's dev transformer and its rank-450 LoRA:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --transformer-path /models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --adapter /models/ltx25/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors:0.25:0.5 \
  --adapter /models/loras/style.safetensors:0.35:0.75 --adapter-exclude audio,cross \
  --adapter /models/loras/motion.safetensors:0.8:0.3 --adapter-exclude audio \
  --prompt "A quiet coastal sunrise with a deliberate camera move" \
  --width 768 --height 448 --frames 121
```

## Check what an adapter changed

A file that parses is not necessarily made for your model. KinoMLX accepts same-generation LoRAs
and cross-generation ones whose targets fit, and warns when:

- fewer than half of the adapter's targets were placed;
- the adapter changes nothing;
- some of its operations or tensors match no weight.

The warnings appear on the console, in `--json`, and in the run record, whose adapter receipt
lists each placed and skipped target with the reason. Check the receipt and the video when
trying an unfamiliar adapter. Malformed pairs and other adapter formats are refused before any
weight changes.

After the first denoising step, adapter-changed weights are checked for nonfinite values. The
check covers only weights the run uses, so a video-only run does not check audio weights. See
[first-use checks](../ADAPTERS.md#first-use-weight-checks) before turning it off.

## Fusion precision

KinoMLX adds adapters into the transformer's weights in memory, for each load. The checkpoint
files never change. `--adapter-accumulation` sets the precision of that addition:

| Value | Precision | Use it when |
| --- | --- | --- |
| `auto` (default) | FP16 on a streamed FP16 transformer, FP32 otherwise | Normal runs |
| `float32` | FP32 everywhere | Comparing a streamed run with a resident one |
| `float16` | FP16 everywhere; needs an FP16 transformer | Matching a streamed run on a resident transformer |

FP16 fusion changes the video, not just the time. One measured comparison on the 22B model, at
the same seed:

| Fusion | Video against FP32 fusion |
| --- | --- |
| FP32 | Reference |
| FP16 | 26 to 27 dB PSNR |

Keep the setting fixed when checking reproducibility.

A weight whose adapter factor does not fit FP16 falls back to FP32, with a warning. Quantized
weights always fuse in FP32, before packing.

## Config files and Python

TOML takes an ordered `adapters` list under `[generate]`. Each record has `path`, `strength`,
`exclude`, `stage1_strength`, and `stage2_strength`:

```toml
[generate]
adapters = [
  {path = "style.safetensors", strength = 0.8, stage1_strength = 0.25},
  {path = "motion.safetensors", stage1_strength = 0, stage2_strength = 0.5, exclude = ["audio"]},
]
```

`KINO_LTX2_ADAPTERS` takes the same list. Change the second item with
`--set 'generate.adapters[1].stage2_strength=0.25'`, or clear the list with
`--set 'generate.adapters=[]'`. Python requests take
`LTX2Request(adapters=(LTX2AdapterConfig(...), ...))` from `kinomlx.models.ltx2`.

A [stage-2 restart](restart.md) keeps each adapter's stage-1 strength and may change only stage 2.
A decode restart keeps both.
