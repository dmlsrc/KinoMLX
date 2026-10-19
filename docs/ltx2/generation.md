# Generating video

This page covers the choices every LTX2 run makes: length, size, prompt, audio, and sampling.
Every argument it mentions is in [the catalog](arguments.md).

- [A first run](#a-first-run)
- [Length and frame rate](#length-and-frame-rate)
- [Size](#size)
- [Prompt and text encoding](#prompt-and-text-encoding)
- [Audio](#audio)
- [Samplers and noise](#samplers-and-noise)
- [Generated keyframes](#generated-keyframes)
- [Config files](#config-files)

## A first run

```bash
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 --generate-audio \
  --output-dir outputs --output-prefix coastal-sunrise
```

This makes a five-second 1024x576 video with sound. Run `kinomlx --model ltx2 --dry-run` first
to check that every model file is found; see [model files](downloads.md).

## Length and frame rate

Give the length one way: `--frames`, `--duration`, or, on LTX-2.5, `--auto-duration`.

| Option | Use it when |
| --- | --- |
| `--frames N` | You want an exact frame count. It must be `8*k+1`, such as 121 or 241. |
| `--duration SECONDS` | You think in seconds. KinoMLX rounds up to the next valid count: 10 seconds at 24 fps is 241 frames. |
| `--auto-duration` | LTX-2.5 should pick a natural length for the prompt, using its duration head. |

`--fps` sets the frame rate the model generates for and the video plays at, 24 by default. To
deliver another rate, interpolate afterward with [`--target-fps`](output.md#change-the-frame-rate).

Run time grows faster than length: in the [measured runs](performance.md#run-times), a
30-second 768x448 clip took about 7.5 times as long as a five-second one.

## Size

Width and height are multiples of 64; the default is 1024x576. Stage 1 runs at half that size,
and stage 2 at full size.

Memory and time grow with the pixel count. On a 32 GB Mac, start at 768x448 with
`--stream-transformer`. To deliver a larger video cheaply, generate smaller and upscale with
[`--vsr-spatial-mode`](output.md#upscale).

## Prompt and text encoding

LTX2 takes one prompt per run, from `--prompt` or `--prompt-file`; see
[prompts](../USAGE.md#prompts). LTX-2.3 encodes it with Gemma 3 and LTX-2.5 with Gemma 4. The
encoder is released before the transformer loads.

LTX-2.3 uses the QAT-unquantized Gemma 3 that Lightricks pairs with the release.
`--gemma-variant plain` uses Google's instruction-tuned Gemma 3 instead, which changes the
conditioning and therefore the video.

By default the encoder's input is padded to 1,024 tokens, as the reference implementation does.
`--compact-prompt` encodes only the prompt's own tokens; it also changes the conditioning.

### Reuse an encoded prompt

`--save-text-conditioning` saves the encoded prompt as `<stem>_text.safetensors`. Pass it back
with `--text-conditioning` to skip text encoding, for example to try new seeds:

```bash
kinomlx --model ltx2 --text-conditioning outputs/<stem>_text.safetensors --seed 7
```

The file records which encoder and settings produced it, and a mismatch with the current run is
refused.

## Audio

`--generate-audio` generates stereo 48 kHz sound together with the video, in the same denoising
run, so it follows the picture. Both versions read their audio VAE and vocoder from the model
files; LTX-2.5 needs the pack's audio VAE.

The audio covers the whole video. `--reference-aligned-audio` instead uses the reference
implementation's rounding, which can make the audio one latent shorter (126 instead of 127
latents at 121 frames); use it only for parity tests.

The audio is muxed as Apple Lossless by default. [Encoding](output.md#encode) covers
`--audio-codec`, the start-click repair, and the WAV sidecar.

## Samplers and noise

Each stage has its own sampler. `auto` runs what the checkpoint was distilled for:

| Version | Stage 1 | Stage 2 |
| --- | --- | --- |
| LTX-2.3 | Deterministic Euler | Deterministic Euler |
| LTX-2.5 | Ancestral RF Euler | Ancestral RF Euler |

`--stage1-sampler` and `--stage2-sampler` take `deterministic` or `ancestral` on either version,
for experiments. LTX-2.3 was not distilled for ancestral sampling. Each ancestral stage draws
fresh noise from its own stream, so the two stages never share noise.

```bash
kinomlx --model ltx2 --model-generation 2.5 --stage2-sampler deterministic \
  --prompt "A quiet coastal sunrise, cinematic light" --width 768 --height 448
```

A [stage-2 restart](restart.md) can change `--stage2-sampler` on one saved stage-1 latent, which
is the cleanest way to compare samplers. The run record lists each stage's sampler and noise
seed.

`--noise-backend torch-mps` reproduces PyTorch 2.13.0's MPS noise from the same seed, to compare
against tools built on PyTorch. It covers every random draw of a run: both stages' starting and
ancestral noise, restart positions, and the diffusion decoder's texture. It changes the sample,
not the model, and PyTorch is not needed. The default `mlx` is KinoMLX's own noise.

The noise levels of each stage are fixed by the distilled recipe. `--sigmas-stage-1` and
`--sigmas-stage-2` replace them; see [Denoise schedules](../SCHEDULES.md#ltx2-has-two-of-them).

## Generated keyframes

`--generated-keyframes N` adds N keyframes that LTX-2.5 generates itself, alongside the video.
The video attends to them while it denoises, which shapes its motion and composition. They are
not image conditions and are not written to the output.

```bash
kinomlx --model ltx2 --model-generation 2.5 --auto-duration --generated-keyframes 1 \
  --prompt "A quiet coastal sunrise, cinematic light" --width 1024 --height 576
```

Each keyframe costs as much as an [image keyframe](images.md#what-a-keyframe-costs): one latent
frame of extra tokens in both stages. Stage 2 starts its keyframes from stage 1's. The run record
lists their frame positions. LTX-2.3 has no generated keyframes.

## Config files

Any run can be a TOML file; see [Configuration](../CONFIG.md). This one generates with the dev
transformer and two adapters, as in [the dev-transformer recipe](adapters.md#run-a-dev-transformer):

```toml
model = "ltx2"

[model_settings]
model_generation = "2.3"
transformer_path = "/models/ltx23/ltx-2.3-22b-dev.safetensors"
text_encoder_path = "/models/gemma-3-12b-it-qat-q4_0-unquantized"
spatial_upscaler_path = "/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

[generate]
prompt = "A quiet coastal sunrise with a restrained visual treatment"
width = 768
height = 448
frames = 121
seed = 42
generate_audio = true

[[generate.adapters]]
path = "/models/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
stage1_strength = 0.25
stage2_strength = 0.5

[[generate.adapters]]
path = "/models/loras/style.safetensors"
stage1_strength = 0.3
stage2_strength = 0.75
exclude = ["audio", "cross"]

[output]
directory = "outputs"
prefix = "coastal-sunrise"
save_all_sidecars = true
```

This one animates between a first and a last frame:

```toml
model = "ltx2"

[generate]
prompt = "A slow cinematic push through the illuminated scene"
width = 768
height = 448
frames = 121
images = [
  {path = "first-frame.png"},
  {path = "last-frame.png", frame_index = "last", strength = 0.8, stage2_strength = 0.5},
]

[output]
directory = "outputs"
prefix = "first-frame-motion"
```

```bash
kinomlx --config coastal-sunrise.toml
kinomlx --config first-frame-motion.toml --set generate.seed=7
```

`kinomlx config init --model ltx2` writes a template with every field and its default.
