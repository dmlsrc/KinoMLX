# LTX2 Model Guide

Generate video from text or images with LTX-2.3 or LTX-2.5, optionally with stereo
audio. Both versions support adapters and restarting from saved intermediate
results. Their HDR inputs and automatic-duration features differ; use the
capability table below to choose a version.

Start with the [shared installation](../README.md#install), download the
[required model files](#ltx-model-downloads), then follow the [usage examples](#usage).
KinoMLX uses the distilled two-stage 8 + 3 recipe; this guide documents its controls
rather than a general LTX training or fine-tuning workflow.

In this guide: [model differences](#ltx-capability-matrix),
[downloads](#ltx-model-downloads), [generation](#usage), [adapters](#adapter-composition),
[streaming](#block-streaming-on-32-gb-systems), [restart](#restart-from-saved-stages),
[HDR](#hdr-generation), [Python API](#library-api), and [performance](#performance).

See [seed lists and model lifetimes](../README.md#seed-lists-and-model-lifetimes)
for ordered seeds, conditioning reuse, and image execution modes.

## Status

Pre-alpha: CLI and Python generation are available for both versions, but interfaces
can change. Output is HEVC MP4 with optional stereo audio. Streaming reduces the
transformer memory requirement; larger dimensions, longer clips, and decoding
still need additional memory. See [hardware](#hardware) and [performance](#performance).

### Public surface

| Model | Input and output | Public recipe | Runner |
| --- | --- | --- | --- |
| `ltx2` | Text/optional image -> video and optional audio | `generate_distilled()` | `LTX2Runner` |

### LTX capability matrix

| Capability | LTX-2.3 | LTX-2.5 | Public surface |
| --- | --- | --- | --- |
| Distilled SDR text/image-to-video | Implemented and tested | Implemented and tested | `generate_distilled()` / CLI |
| Joint stereo audio generation | Implemented and tested | Implemented and tested | `--generate-audio` |
| Reference audio latent length | Implemented and tested | Implemented and tested | `--reference-aligned-audio` |
| Torch-MPS-compatible normal noise | Implemented and tested | Implemented and tested | `--noise-backend torch-mps` |
| First-frame and arbitrary-frame image conditions | Implemented and tested | Implemented and tested | `--image`, `--image-frame-index` |
| Automatic natural duration | Not present in published artifacts | Implemented and tested | `--auto-duration` or `frames=None` |
| Generated stage-1 keyframe slots | Not present in the 2.3 graph | Implemented and tested | `--generated-keyframes N` |
| Same- or cross-generation community adapter (ordinary LoRA) | Implemented and tested | Implemented and tested | Repeat `--adapter`; every placeable target fuses |
| Temporal x2 latent upscaler | Component-ready when supplied | Component-ready and tested | Library component station; no DFR CLI claim |
| Diffusion video-VAE decoder | Compatible 2.5 decoder implemented | Implemented and tested | Select with `--video-vae diffusion`; convolution remains the default |
| Restart from saved stage products | Implemented | Implemented | `--restart` / `DistilledRestart` / `LTX2Runner.restart()` |
| SDR-to-HDR video conversion (HDR IC-LoRA) | Implemented and tested | Not applicable | `--hdr` + `--hdr-reference` + declared LogC3 adapter |
| Native HDR image-to-video | Not present in the 2.3 graph | Implemented and tested | `--hdr` + HDR EXR `--image` |
| Text-only HDR | Not supported (refused) | Not supported (refused) | See "HDR generation" below |

Low adapter coverage produces a warning rather than automatically rejecting the file. KinoMLX
warns below 50% and records placed/skipped targets, strengths, knockouts, generation metadata,
adapter type, and artifact identity in the run receipt. LTX-2 currently admits ordinary LoRA
weight adapters; malformed pairs, other parsed adapter types, and unsafe fusion math fail before
weights are changed.

Generated keyframe slots are fully denoised stage-1 tokens whose self-attention
can influence the ordinary stage-1 video latent. The standard distilled recipe
records their selected frame indices in `run.json`, then removes the slot tokens
at the stage boundary; it does not retain or emit separate keyframe sidecars.

## Hardware

Designed for Apple Silicon (M1 family or later). LTX generation needs sufficient
unified memory: Gemma 3/Gemma 4 and the selected transformer are loaded
sequentially, not resident together. 32 GB is the supported LTX minimum with
transformer streaming; 64 GB is recommended for the default resident-transformer
path and larger generation geometries.

## Install and configure LTX2

KinoMLX targets Python 3.14 on Apple Silicon.

```bash
git clone https://github.com/dmlsrc/KinoMLX.git
cd KinoMLX
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

Any PEP 517 installer works; `pip install -e .` into an activated
virtual environment is equivalent. Then:

```bash
kinomlx --help
kinomlx --model ltx2 --help
kinomlx config --help

# Write the annotated LTX-2 starter config to ./kino-config.toml.
kinomlx config init

# Resolve defaults, environment, TOML, CLI flags, and typed --set overrides
# into one reproducible invocation without loading model weights.
kinomlx --config run.toml --set generate.seed=7 --print-config

# Save that same input TOML without replacing an existing file.
kinomlx --config run.toml --set generate.seed=7 --save-config selected.toml

# Keep only values that differ from built-in defaults. Environment-derived
# differences are retained, so this remains a reproducible invocation.
kinomlx --config run.toml --save-config overrides.toml --only-non-defaults
```

The initializer documents every field's purpose, accepted values, built-in
default, and any associated environment variable. It refuses to replace an
existing path. Prompt text is rendered as an editable multi-line literal block;
KinoMLX strips only the framing whitespace when it resolves the invocation.
Use the generated template to find the correct TOML table and environment variable
for an option. Unknown fields and invalid choices are rejected before loading a
model, so a misspelled setting does not silently fall back to a default.

`--print-config` and `--save-config PATH` serialize canonical input TOML and exit
before model loading. They may be used together. Built-in defaults appear in a
commented reference section; they are not activated on reload. `--only-non-defaults`
omits that section. Both forms retain supplied defaults, explicit opt-outs, environment
selections and meaningful clearing. A duration remains a duration; calculated frame
counts belong in execution metadata. If TOML cannot represent a clear through a declared
alternative, export fails before creating the file instead of silently losing the clear.

### Optional Hugging Face CLI

KinoMLX does not download model weights at runtime. If you want Hugging Face to
manage the local cache, install the `hf` CLI as an isolated uv tool:

```bash
uv tool install hf
hf --help
```

The official [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
also documents the no-install `uvx hf` form. If `hf` is not on `PATH` after a
tool install, run `uv tool update-shell` and open a new shell. The CLI is needed
only for authentication and cache-managed downloads; KinoMLX itself loads local
files and cached snapshots directly. You may skip the CLI and download files
through the model pages instead.

### LTX model downloads

The model releases are gated on Hugging Face. Inspect their model cards, files,
and terms before downloading:

- [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) and its paired
  [QAT-unquantized Gemma 3](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
- [LTX-2.3 FP8](https://huggingface.co/Lightricks/LTX-2.3-fp8) for optional
  E4M3 transformer-source files
- [LTX-2.3 HDR IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
  for the optional SDR-video-to-HDR-video workflow
- [LTX-2.5 split component pack](https://huggingface.co/Lightricks/LTX-2.5)
- Optional [vanilla Gemma 3](https://huggingface.co/google/gemma-3-12b-it) for
  `--gemma-variant plain`

Before downloading by either method, sign in through each linked model page and
accept or request access for every repository shown as gated. CLI authentication
alone does not grant access to a gated repository.

#### Cache-managed download

After access is approved, authenticate the optional CLI and download:

```bash
hf auth login

hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-1.1.safetensors \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# Optional: full transformer plus its matching v1.1 distilled LoRA
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-dev.safetensors \
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors

hf download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized

# Optional: the FP8 sources, converted lazily to the execution dtype
hf download Lightricks/LTX-2.3-fp8 \
  ltx-2.3-22b-dev-fp8.safetensors \
  ltx-2.3-22b-distilled-fp8.safetensors \
  --local-dir /models/ltx23-fp8

# Optional: the LTX-2.3 SDR-to-HDR adapter
hf download Lightricks/LTX-2.3-22b-IC-LoRA-HDR \
  ltx-2.3-22b-ic-lora-hdr-0.9.safetensors

# Optional: the vanilla instruction-tuned encoder for --gemma-variant plain
hf download google/gemma-3-12b-it

# Download the LTX-2.5 release without the unused INT8 and NVFP4 assets.
hf download Lightricks/LTX-2.5 --exclude "*int8*" --exclude "*nvfp4*"

# Or preserve that pack in a directory you choose.
hf download Lightricks/LTX-2.5 --local-dir /models/ltx25 \
  --exclude "*int8*" --exclude "*nvfp4*"
```

For a cache-managed LTX-2.3 download, point `KINO_LTX2_WEIGHTS_PATH` (or
`--weights-path`) at `ltx-2.3-22b-distilled-1.1.safetensors`. The spatial
upscaler is found next to the checkpoint or in the local Hugging Face cache, and
the Gemma snapshot is resolved from the cache automatically - the
QAT-unquantized encoder paired with the official release by default, or the
vanilla release with `--gemma-variant plain` (`KINO_LTX2_GEMMA_VARIANT`).
`KINO_LTX2_GEMMA_PATH` and `KINO_LTX2_SPATIAL_UPSCALER_PATH` override automatic
resolution.

For a cache-managed LTX-2.5 download, pass `--model-generation 2.5`; KinoMLX
selects the prepared local snapshot, so an exact snapshot path is not required.
For a local-directory or browser download, pass the directory once as
`--weights-path /models/ltx25`. KinoMLX recognizes both the official
subdirectory tree and a flat directory containing the canonical filenames.
The older `--ltx-generation` spelling remains a compatibility alias. The
selected recipe requires only the components it consumes:

| LTX-2.5 component | Hub path | Requirement |
| --- | --- | --- |
| Distilled transformer and video/audio connectors | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | Required for the public distilled recipe |
| Gemma 4 text encoder and projection | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | Required |
| Convolutional video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | Required by the default `--video-vae conv` path |
| Spatial x2 latent upscaler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | Required by the two-stage distilled recipe |
| Audio VAE and vocoder | `vae/ltx-2.5-audio-vae-bf16.safetensors` | Required only with `--generate-audio` |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | Required only for `--auto-duration`; explicit `--frames` does not load it |
| Diffusion video VAE | `vae/ltx-2.5-video-vae-bf16.safetensors` | Optional alternative selected by `--video-vae diffusion` |
| Temporal x2 latent upscaler | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | Component-ready library station; no public DFR recipe or CLI claim |
| Dev transformer | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | Optional compatible transformer; the distilled recipe may consume it, but CFG/STG is not implemented |
| Official distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | Optional adapter; not required by the distilled transformer |

#### Manual browser download

You do not need the `hf` CLI or a Hugging Face cache layout. After access is
approved, open each model page's Files and versions tab, download the files into
any local folder, and pass their paths explicitly.

For LTX-2.3, download the distilled checkpoint and spatial upscaler from the
LTX-2.3 page. The dev transformer and matching distilled LoRA are optional files
from that same page. Download the paired Gemma repository contents into one
directory; that directory must contain `config.json`, the tokenizer files, and
all model files:

```bash
kinomlx --model ltx2 \
  --weights-path /models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upscaler-path /models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

For LTX-2.5, download the required rows from the component table into one
folder. You may preserve the repository subdirectories or place the files
directly in the folder without renaming them. Point KinoMLX at that folder once:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --weights-path /models/ltx25 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

The pack resolver finds the audio VAE for `--generate-audio`, the duration head
for `--auto-duration`, and the other canonical optional files when they are
present. For the diffusion decoder, select `--video-vae diffusion`; the same
pack root then selects `ltx-2.5-video-vae-bf16.safetensors`. Use the individual
`--transformer-path`, `--text-encoder-path`, `--video-vae-path`,
`--audio-vae-path`, `--spatial-upscaler-path`,
`--temporal-latent-upscaler-path`, and `--duration-head-path` options only for
renamed, scattered, or deliberately overridden components.

#### Override individual components

Component path overrides replace only the selected component. Missing LTX-2.5
components are discovered from the transformer's local pack, then a cached default
pack; explicit paths always win. A `--weights-path` pack directory selects the
baseline explicitly, and a monolithic baseline keeps supplying its unchanged parts.

Treat `--weights-path` as the baseline: an LTX-2.3 monolithic checkpoint or an
LTX-2.5 component-pack directory. Add any explicit component path to replace
only that component. For example, this keeps every other LTX-2.5 component in
the pack but swaps in the dev transformer:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --weights-path /models/ltx25 \
  --transformer-path \
    /models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

The same precedence rule applies to both generations and every component
override:

| Explicit option | What it replaces from the pack |
| --- | --- |
| `--transformer-path` | Transformer plus its video/audio connectors |
| `--text-encoder-path` | Prepared text encoder, tokenizer data, and any bundled projection |
| `--gemma-path` | LTX-2.3 Gemma source; use this for a full local Gemma directory |
| `--video-vae-path` | Selected video VAE |
| `--audio-vae-path` | Audio VAE and vocoder |
| `--spatial-upscaler-path` | Spatial x2 latent upscaler |
| `--temporal-latent-upscaler-path` | Temporal x2 latent upscaler |
| `--duration-head-path` | Automatic-duration head |

An explicit path always wins. For LTX-2.3, components without explicit paths
continue to come from the monolithic baseline, its neighboring files, or the
configured Hugging Face cache. For LTX-2.5, they continue to come from the pack
directory. Without `--video-vae-path`, the `--video-vae conv|diffusion` selector
chooses the compatible named decoder.

The loader inspects the effective transformer first, then checks every retained
or overridden component against it before loading weights. A mismatched text
width, latent channel count, connector shape, upscaler kind, duration-head
context, or selected generation fails explicitly.

Some subcomponents are bundled rather than independently selectable. The A/V
connectors normally travel with a compatible transformer checkpoint; tokenizer
and projection data travel with the prepared text artifact or baseline; and the
vocoder travels with the audio artifact. Those subcomponents therefore do not
have independent path options.

The LTX-2.3 and LTX-2.5 dev transformers are compatible with KinoMLX's public
distilled two-stage runner, but the runner does not implement a dev CFG/STG
recipe. To use an official dev-transformer distillation workflow, add the
matching official distilled LoRA as shown in
[adapter composition](#adapter-composition).

#### Component loading

KinoMLX loads the model components needed for each phase, then releases them before
moving on. Dtype conversion and weight-layout preparation are lazy, so the first
use of a component can include work that was not paid during resource inspection.
See [model loading](WEIGHTS.md#what-happens-during-loading) when interpreting phase
timings or diagnosing memory pressure.

Adapters work with both resident and streamed transformers. Streaming repeats
adapter fusion when blocks are reloaded, so it can make fusion precision a visible
performance choice:

| `--adapter-accumulation` | Behavior |
| --- | --- |
| `auto` | FP16 accumulation for ordinary LoRA on a streamed FP16 transformer; FP32 otherwise |
| `float32` | Explicit FP32 accumulation, useful for comparing resident and streamed output |
| `float16` | Explicit FP16 accumulation; requires `--transformer-dtype float16` |

FP16 accumulation can change the image, not just elapsed time. A recorded 22B
comparison measured 26-27 dB PSNR against FP32 fusion using the same seed. Keep
accumulation settings identical when checking reproducibility across streaming and
resident runs.

Quantized targets fuse adapters before packing and retain FP32 accumulation;
`auto` can still select FP16 for the dense targets in a quantized transformer.
If an adapter factor cannot fit in FP16, the affected weight falls back to FP32
fusion with a warning. This avoids silently overflowing during the factor cast.

Active floating-point adapted weights receive the optional first-use check;
packed targets rely on per-step latent checks. Streamed checks share a progress
bar and report their summed time, with individual batch timings in the run JSON.
See [adapter checks](ADAPTERS.md#first-use-weight-checks) before disabling the scan.

#### FP8 transformer sources and execution precision

KinoMLX accepts safetensors transformers containing E4M3 FP8 weights, including
the official [LTX-2.3 FP8 release](https://huggingface.co/Lightricks/LTX-2.3-fp8).
FP8 is an input-storage format here, not a runtime compute mode. During the
load, KinoMLX prepares dequantization into the selected compute dtype. This
conversion evaluates lazily during denoising. The default target is BF16; select FP16 explicitly
like this:

```bash
kinomlx --model ltx2 --model-generation 2.3 \
  --weights-path /models/ltx23-fp8/ltx-2.3-22b-distilled-fp8.safetensors \
  --gemma-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upscaler-path \
    /models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --transformer-dtype float16 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

Conversion, layout changes, and adapter fusion stay lazy until execution needs them.
Post-step latent checks catch non-finite results; optional first-use weight checks
inspect active adapter-modified weights.

This path supports safetensors E4M3, including scalar weight-scale companions.
E5M2 is rejected, and the LTX-2.5 Comfy INT8 and NVFP4 files are different
formats rather than substitutes for an FP8 source.

The transformer artifact carries the video/audio connectors, and the text artifact carries the
tokenizer data and projection, so those are not separate downloads. LTX-2.5 HDR image-to-video uses
the same model components and additionally requires a genuinely HDR EXR `--image`; no HDR LoRA is
used. KinoMLX rejects prompt-only, SDR-only, and mixed SDR/EXR HDR requests before model loading.

## Usage

```bash
# LTX-2.3 using the default monolithic distilled checkpoint family
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 \
  --generate-audio \
  --output-dir outputs --output-prefix coastal-sunrise

# LTX-2.5 with checkpoint-predicted duration and one generated keyframe slot
kinomlx --model-generation 2.5 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --auto-duration --generated-keyframes 1

# Optional override: use deterministic Euler in both LTX-2.5 stages
kinomlx --model-generation 2.5 --sampler deterministic \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121

# Select the official LTX-2.5 neighborhood-attention diffusion VAE explicitly
kinomlx --model-generation 2.5 \
  --video-vae diffusion \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

`--sampler auto` is the default. It gives published LTX-2.3 distilled
checkpoints deterministic Euler in both stages, while LTX-2.5 defaults to
ancestral RF Euler in stage 1 and deterministic Euler in stage 2.
`--sampler deterministic` is not the LTX-2.5 default; it is an explicit
all-deterministic override. `--sampler ancestral` is the corresponding explicit
stage-1 ancestral selection for controlled A/B runs.
The run receipt records the requested override, effective policy, and derived
ancestral seed. The sigma schedules do not change.

`--noise-backend mlx` is the default and preserves existing KinoMLX output.
`--noise-backend torch-mps` selects an MLX Metal implementation of PyTorch
2.13.0 MPS normal noise for comparisons with applications using that generator; PyTorch is not a
runtime dependency. The choice applies to every generation-time Gaussian
draw: shared stage-1/stage-2 initialization, the independent ancestral stream,
restart position recovery, and diffusion-VAE texture noise. The run receipt
records the backend, compatibility profile, seeds, draws, element counts, and
Philox-block positions. Gemma text conditioning is deterministic and consumes
no diffusion-noise draws.

KinoMLX normally rounds the audio latent count up so the causal decoder covers
the requested video timeline. `--reference-aligned-audio` instead reproduces
the reference implementation's nearest-integer rule for parity experiments.
At 121 frames and 24 fps, the default uses 127 audio latents and the reference
mode uses 126. The run receipt records the selected policy and latent shape.

`--video-vae conv` is the default. `--video-vae diffusion` discovers the
official neighborhood-attention artifact in a cached 2.5 snapshot, including
when the selected transformer is LTX-2.3. Lightricks explicitly preserves the
2.3 video-latent contract through 2.5, so a 2.3 encoder or saved latent can feed
either decoder. KinoMLX still reads and validates the actual graph from
checkpoint metadata; an explicit `--video-vae-path` remains the expert override
and always wins over the named selection.

### Invocation files

The same requests as TOML files. Run one with `kinomlx --config <file>`; explicit CLI
flags and `--set` overrides still take precedence, and `kinomlx config init --model ltx2`
writes an annotated template that documents every field and table.

A generation with adapters, matching the dev-transformer example under
[Adapter composition](#adapter-composition):

```toml
model = "ltx2"

[model_settings]
model_generation = "2.3"
transformer_path = "/models/ltx23/ltx-2.3-22b-dev.safetensors"
gemma_path = "/models/gemma-3-12b-it-qat-q4_0-unquantized"
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

An image-conditioned generation from a first frame:

```toml
model = "ltx2"

[generate]
prompt = "A slow cinematic push through the illuminated scene"
width = 768
height = 448
frames = 121
seed = 42

[generate.image]
path = "first-frame.png"
frame_index = 0
strength = 0.95

[output]
directory = "outputs"
prefix = "first-frame-motion"
save_all_sidecars = true
```

```bash
kinomlx --config coastal-sunrise.toml
kinomlx --config first-frame-motion.toml --set generate.seed=7
```

### Machine-readable run results

`--json` uses the same success envelope for every model command: `status`,
`model`, `action`, `outputs`, `elapsed_seconds`, `timings`, `diagnostics`,
`warnings`, and `sidecar_errors`, plus model-specific facts under `generation`.
When writing a script, read output paths from `outputs` and adapter application
details from `generation.adapter_receipts`. Older clients expecting `output` or
`generation.lora_receipts` need to use these current names.

Saved TOML uses ordered `generate.adapters` records; the API uses
`DistilledRequest(adapters=(LTX2AdapterConfig(...), ...))`. Both types are exported from
`kinomlx.models.ltx2`. The former parallel-array fields are no longer accepted. CLI `--lora*`
spellings remain aliases for the corresponding `--adapter*` flags.
LTX-only composition primitives also live with the model: import `LatentState`
from `kinomlx.models.ltx2.types` and `GaussianNoiser` from
`kinomlx.models.ltx2.noising`; the former generic import locations are not
retained as compatibility shims.

### Adapter composition

LTX uses direct-strength LoRA: `W_run = W_base + stage_strength * (B @ A)`.
Saved alpha is ignored, including for supported PEFT exports; no `alpha/r`
multiplier is applied. This differs from Krea and Klein's format-dependent
[alpha and scaling rules](ADAPTERS.md#strength-alpha-and-effective-scale).

Each `--adapter` starts an ordered record. Strength, stage-strength and exclusion modifiers
apply to the latest preceding CLI item. Controls do not broadcast; duplicate member assignments
are errors, including equal values and aliases. The common `--adapter-strength` defaults to `1.0`
for that item and supplies both stages unless an explicit stage strength overrides it.

The familiar `--lora`, `--lora-strength`, `--lora-stage1-strength`,
`--lora-stage2-strength`, and `--lora-exclude` spellings remain CLI aliases.
Each record owns `path`, `strength`, `exclude`, `stage1_strength` and `stage2_strength`.

Separate flags modify the latest preceding adapter:

```bash
--adapter style.safetensors --adapter-strength 0.8
--adapter style.safetensors \
  --adapter-stage1-strength 0.25 --adapter-stage2-strength 0.5
```

Or attach the strength to the path, which is the same thing written shorter:

```bash
--adapter style.safetensors:0.8
--adapter style.safetensors:0.25:0.5
```

The two-number suffix is LTX-2 only and reads stage 1 then stage 2, as absolute
strengths. A `0` skips that adapter in that stage, so `detail.safetensors:0:0.8`
applies in stage 2 alone. One numeric suffix supplies only common strength; two supply only
the two stage strengths. Neither form assigns defaults to sibling records. Reassigning a member
already supplied by shorthand is an error. Numeric suffixes after `.safetensors` are decoded
without filesystem probes. Nonnumeric suffixes remain literal; structured paths are always literal.
See [Adapters](ADAPTERS.md#inline-strengths) for the common rules.

```toml
[generate]
adapters = [
  {path="style.safetensors", strength=0.8, stage1_strength=0.25},
  {path="motion.safetensors", stage1_strength=0, stage2_strength=0.5, exclude=["audio"]},
]
```

`KINO_LTX2_ADAPTERS` accepts the same array. A higher-precedence collection replaces user adapters;
`generate.adapters=[]` clears them. `--set 'generate.adapters[1].stage2_strength=0.25'` patches the
second existing item. Replacing an entire item removes omitted controls. Stage-2 restart preserves
the effective first-stage profile, including common-strength fallback and zero filtering; decode
restart preserves both stage profiles. Whole-item replacement obeys the same locks as member edits.

One flag per adapter stays readable as a stack grows, so the examples below use it.

The official distilled LoRAs are intended for dev-transformer workflows. They
can be combined with additional compatible style, motion, or character LoRAs in
the same invocation. Here is an LTX-2.5 composition:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --transformer-path \
    /models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --adapter \
    /models/ltx25/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors:0.25:0.5 \
  --adapter /models/loras/style.safetensors:0.35:0.75 --adapter-exclude audio,cross \
  --adapter /models/loras/motion.safetensors:0.8:0.3 --adapter-exclude audio \
  --prompt "A quiet coastal sunrise with a deliberate camera move" \
  --width 768 --height 448 --frames 121
```

The distilled LoRA uses Lightricks' two-stage HQ profile, `0.25` then `0.5`. The
style LoRA is lighter in stage 1 and stronger in stage 2; the motion LoRA does the
opposite. Each exclusion modifies its preceding adapter; the distilled LoRA has no exclusions.
Strengths must be finite. Negative values
subtract the adapter delta, while unusually large magnitudes can overwhelm the
base model or overflow an FP16 fused weight; follow an adapter author's
recommendation unless you are deliberately experimenting.

A knockout omits matching transformer targets from one adapter before fusion. The
common broad categories are `video`, `audio`, and `cross` for branches, plus
`attn`, `gate`, and `ff` for operation families. More specific categories such
as `attn1`, `audio_ff`, `to_q`, `project_out`, `adaln`, `cross_control`, and
`distill_control` are also available; the annotated starter config written by
`kinomlx config init --model ltx2` lists the complete accepted set beside the
`adapters` record. Comma-separate multiple categories for one adapter. Omit the modifier
for no knockouts, or use `exclude=[]` to explicitly clear categories in a structured record.
Structural coverage is the fraction of all complete parsed target
operations that were placed. Excluded targets remain in that denominator, so
the displayed `Targets` and `Coverage` values always describe the same set.

Every adapter is mapped independently, then all selected deltas are accumulated
into the selected transformer's live weights. The source checkpoint and adapter
files remain unchanged. If the two stage profiles differ, KinoMLX closes the
stage-1 transformer and reloads a pristine base transformer before applying the stage-2
profile; stage-1 deltas therefore cannot leak or compound
into stage 2.

After the first step, adapted-weight validation checks only weights used by the
active modality paths. Video-only generation leaves unused audio and cross-modal
weights lazy. A disabled stream that supplies cross-attention context is still
accounted for. Previously inactive weights remain pending until their path is
used; changing the stage adapter profile creates a fresh validation lifetime.
`--no-check-adapter-weights` disables these scans while retaining per-step latent
finite checks.

Same-generation and structurally placeable cross-generation community LoRAs are
accepted, but compatibility is not an artistic-quality guarantee. KinoMLX warns
when fewer than half of all complete parsed targets are placed, when an adapter
has no effective merge, or when operations or tensors remain unmapped. Those
warnings appear in the console, `run.json`, and `--json` output. The run receipt
records source identity, declared and selected generations, strength, stages,
knockouts, placed targets, skipped reasons, effective status, warning strings,
and structural coverage for each adapter. Inspect that receipt and the generated
media when validating an unfamiliar adapter.

The LTX-2.3 form uses the monolithic checkpoint as its baseline for the video
VAE, audio VAE/vocoder, and other compatible embedded components while replacing
the transformer. This example pairs the 1.1 dev transformer workflow with the
matching official 1.1 distilled LoRA and one additional visual adapter:

```bash
kinomlx --model ltx2 --model-generation 2.3 \
  --transformer-path /models/ltx23/ltx-2.3-22b-dev.safetensors \
  --gemma-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upscaler-path \
    /models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --adapter \
    /models/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors:0.25:0.5 \
  --adapter /models/loras/style.safetensors:0.3:0.75 \
  --adapter-exclude audio,cross \
  --prompt "A quiet coastal sunrise with a restrained visual treatment" \
  --width 768 --height 448 --frames 121
```

Lightricks' current two-stage HQ pipeline gives the official distilled LoRA
separate defaults: `0.25` in stage 1 and `0.5` in stage 2. KinoMLX does not
identify a distilled LoRA from its filename, so pass both values explicitly as
shown. Only the additional style adapter changes to custom values and knocks
out audio/cross targets.

LTX adapter fusion accepts ordinary LoRA weight adapters with resident or streamed
transformers. Quantized transformers fuse adapters into source weights before packing.
Streamed blocks repeat adapter fusion when reloaded; this adds work per denoise step
that depends on the adapted weights, rather than output resolution. Active floating-point
adapter targets receive the optional first-use weight check.

### Block streaming on 32 GB systems

KinoMLX supports 32 GB unified memory as its minimum, while 64 GB is recommended
for the default fully resident transformer and larger geometries. On a machine
with less than 64 GB, add `--stream-transformer` and start with a modest output
geometry:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --stream-transformer \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

The preset keeps 8 of the 48 transformer blocks resident as two four-block groups,
run uncompiled so that streamed and resident outputs match bit for bit when their
accumulation and execution settings match (the default `auto` adapter accumulation
differs between the routes for float16 adapters; `--transformer-compile-group-size`
compiles groups on request and measured no faster). Each group is dispatched without
waiting, so its source reads and dtype/layout preparation overlap the previous group's
computation. The allocator cache limit is whatever you configured, defaulting to 1 GiB;
see [allocator caching](PERFORMANCE.md#allocator-cache) for the tradeoff.
It works with LTX-2.3 and LTX-2.5. It reduces the
transformer live set but does not make output geometry free: longer clips, larger
frames, audio, and decoder work still consume unified memory. Every denoising step
re-reads all 48 blocks (37 GB for the 22B checkpoint), so once the checkpoint no
longer fits the page cache the streamed step time follows the drive's sequential
read rate. Adapters add a fusion per reloaded block; on a float16 transformer the
default `--adapter-accumulation auto` runs that repeated fusion in float16 (a resident
transformer keeps float32), and large adapters stay resident for the run (the
rank-384 distilled adapter is 7.6 GB). See [weight loading](WEIGHTS.md) for shared
loading and memory behavior.

`--adapter-accumulation float32` forces FP32 fusion on both resident and streamed
transformers. `--adapter-accumulation float16` forces FP16 fusion and requires
`--transformer-dtype float16`. Keep this choice fixed when comparing streamed and
resident output. Adapted floating-point targets are checked once per loaded profile
after their first group execution and before eviction; latent checks remain active
if you disable the weight scan with `--no-check-adapter-weights`.

#### Retaining selected block tensors

Every streamed step re-reads all 48 blocks, so on a machine that cannot keep the
checkpoint in the filesystem cache the step time follows the drive. `--transformer-retain`
keeps chosen tensors resident across steps, removing their bytes from that per-step read
stream. Selectors are the class names `small`, `audio-attention`, `video-attention`,
`audio-ff` and `video-ff`, or globs over block-relative MLX module names such as
`ff.project_out.weight`. Combine them in one comma-separated value; repeating the flag
replaces the previous value rather than adding to it.

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --stream-transformer \
  --transformer-retain 'small,audio-attention,*to_gate_logits.weight,ff.project_out.weight' \
  --transformer-resident-blocks 4 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 512 --height 512 --frames 49 --generate-audio
```

Retention is off by default and requires streaming. Each selector must match an
active block tensor. `small` covers norms, biases, and modulation tables;
`audio-attention` includes both audio/video cross-modal paths. Retained tensors
belong to the loaded transformer profile and are rebuilt when its adapters change.
The allocator cache is separate: retained tensors are live allocations, so
`--mlx-cache-limit-gb 0` does not disable tensor retention.

Equivalent configuration and API forms are:

```toml
[model_settings]
stream_transformer = true
transformer_resident_blocks = 4
transformer_retain = ["small", "audio-attention", "*to_gate_logits.weight", "ff.project_out.weight"]
```

The environment binding is `KINO_LTX2_TRANSFORMER_RETAIN`, with the same comma-separated value
as the CLI. Python uses `LTX2Settings(stream_transformer=True, transformer_resident_blocks=4,
transformer_retain=("small", "audio-attention", "*to_gate_logits.weight",
"ff.project_out.weight"))`.

A recorded M1 Max comparison used LTX-2.5 dev weights, FP16 transformer and adapter
accumulation, a rank-450 distilled adapter plus a style adapter, 512x512 output,
49 frames with audio, and the 8+3 schedule. The 64 GiB host had an external 32 GiB
memory reservation; MLX used a 28 GiB wired limit and a 1 GiB allocator cache.
Each run started after cooling. This approximates constrained memory on that host;
it is not a measurement on a physical 32 GiB Mac.

| Configuration | Total time | Stage-1 active MLX peak |
| --- | --- | --- |
| Default streaming, 8 resident blocks, no retention | 88.91 s baseline mean | 16.41 GiB |
| Selection above, 4 resident blocks | 70.18 s | 21.80 GiB |

The retained run's stage-2 peak was higher, at 22.67 GiB. These active-allocation
peaks exclude the allocator cache and are not total system-memory measurements.
Stage-1 and final latent digests matched in that comparison. The approximately 21%
speedup applies to this workload and storage/memory setup, not every generation.
The command above demonstrates the selectors; it does not reproduce that adapter benchmark.

Start with default streaming, then add a small retained selection and measure a
complete run. Larger selections reduce repeated reads and preparation but increase
live memory. A smaller resident-block count can free headroom for retention.
Longer clips and larger frames add activation memory, so a selection that fits a
short clip may not fit another workload. See [Performance](PERFORMANCE.md) for
cache controls and comparison guidance.

Adapters work with the same strength and stage controls during streaming.
The automatic VAE tiler remains active independently and
chooses a decoder plan from the resolved geometry and dtype. Phase-completion
lines and `--save-run-log` report memory-boundary observations that can help
choose a smaller geometry if pressure remains high.

### Restart from saved stages

`--save-all-sidecars` on the original generation records every restart input.
The shortest restart decodes the saved final latent with the original resolved
settings as a lower-precedence base:

```bash
# Re-decode the final latent with another VAE and terminal media policy.
kinomlx --restart outputs/source_run.json \
  --video-vae diffusion --seed 43 \
  --vsr-spatial-mode balanced --target-fps 60 \
  --output outputs/redecoded.mp4

# Decode stage 1 directly, skipping the latent upscaler and stage 2.
kinomlx --restart outputs/source_run.json \
  --latent-stage stage-1 \
  --vsr-spatial-mode fast \
  --output outputs/stage1-diagnostic.mp4

# Resume with the saved stage-1 latent and encoded prompt, then run stage 2.
kinomlx --restart outputs/source_run.json \
  --restart-from stage-2 \
  --output outputs/stage2-rerun.mp4
```

The default is `--restart-from decode --latent-stage final`. A stage-1 direct
decode is half the source width and height; `--vsr-spatial-mode fast` is the
2x terminal mode that restores the original delivery dimensions. Stage-2
restart needs both the stage-1 latent and saved text conditioning.
`--restart-latents` and `--text-conditioning` can substitute sidecars from
another run. Schema-1 stage-1 sidecars have no saved noise position and are
treated as legacy MLX runs. They can resume with the default `mlx` backend;
using `torch-mps` requires regenerating stage 1 so its Philox position is
recorded.

You can deliberately substitute a compatible sidecar when restarting. KinoMLX
records its hash and warns if it differs from the parent run, but a hash or
metadata difference alone does not reject the replacement. The tensor contents
still must be readable, finite, and compatible in shape and dtype. A failed hash
read does not bypass those loading checks.

Settings for work already represented by a saved result cannot be changed at a
later restart point. Decoder seed, VAE/tiling, post-processing, encoding, and the
output path remain selectable when restarting decoding.
Changing the seed or noise backend rerolls stochastic diffusion-VAE texture but
has no effect on the convolutional decoder.

When `--output` is omitted, KinoMLX writes
`<output-dir>/<output-prefix>_YYYYMMDD_HHMMSS.mp4`, following the
[shared app output naming policy](../README.md#output-names-and-sidecars).
Same-second collisions add `_2`, `_3`, and so on before the extension. The defaults are
`outputs/` and `ltx2-distilled-seed<seed>`, so neither option is required. `KINO_LTX2_OUTPUT_DIR`
takes precedence over the shared `KINO_OUTPUT_DIR` fallback.
Pass `--output /exact/path.mp4` when an
exact filename is wanted; it overrides the directory and prefix. The encoder
writes one standard HEVC MP4, so an exact path must end in `.mp4` (any case);
another suffix is refused before generation starts. All
requested sidecars use the same resolved directory and timestamped stem.

Each execution-sidecar toggle is unset by default and inherits
`--save-all-sidecars`; the negative form (`--no-save-run-log`) keeps that one
category off while the rest of the group stays on. `--save-vae-frames` and
`--save-hdr-heic-frames` are outside the group and are never selected by
save-all. The Python `OutputConfig` record resolves the same way, so
`save_all_sidecars=True` alone selects the inherited categories.

`--save-effective-config` writes the fully resolved invocation to
`<video-stem>_config.toml` before model execution. This sidecar includes values
resolved from environment variables and the final output path, so it remains
useful when generation fails. `--save-all-sidecars` enables it by default;
`--no-save-effective-config` is the corresponding category opt-out. The
effective-config writer refuses to replace an existing file. As with the run
and console logs, a sidecar write failure is recorded without aborting an
otherwise viable primary output.
Reloading an effective config that names an already materialized video also
refuses before generation; select a new `--output` or remove the old video
deliberately.

`--save-vae-frames` writes `<video-stem>_vae_frames/frame_000000.png`, and so
on, directly from the normalized SDR VAE frame stream before VSR, frame-rate
conversion, or video encoding. The directory also contains `manifest.json`
with the source signal and exact float-to-8-bit mapping. PNG is losslessly
compressed, but an image sequence has no temporal compression and can still be
large. This diagnostic output is deliberately not enabled by
`--save-all-sidecars`. It is an SDR inspection path; future unclipped HDR or
scene-linear frame delivery belongs to the typed EXR path rather than this
flag.

`--save-media-conditioning` writes the exact ordered VAE-encoded image,
arbitrary-frame keyframe, and HDR-reference inputs for both model stages as
`<video-stem>_stage1_conditioning.safetensors` and
`<video-stem>_stage2_conditioning.safetensors`. Each file records the tensor's
source, family, strength, placement, stage geometry, and frame rate without
retaining the half-resolution tensor until stage 2. When a request has a media
condition, `--save-all-sidecars` enables both files along with the existing
text-conditioning sidecar. Generated stage-1 keyframe slots are internal
denoising state rather than VAE-encoded caller input and remain intentionally
transient.

For HDR generations, `--save-hdr-heic-frames` additionally writes
`<video-stem>_heic/frame_00000.heic`, and so on. These are user-viewable 10-bit
BT.2100 PQ images encoded at quality 0.95. KinoMLX maps scene-linear `1.0` to
the 203-nit HDR reference white before ST-2084 encoding and records the mapping
in the sequence manifest. HEIC is a lossy display encoding for Preview, Quick
Look, and Photos; the mandatory half-float EXR sequence remains the HDR master.
Because a second image sequence is large, this flag is deliberately not
enabled by `--save-all-sidecars`.

Use `--duration SECONDS` instead of calculating `--frames`; KinoMLX rounds up
to the next valid `8*k+1` count. `--audio-onset-trim auto|off|MS` controls the
sync-safe sequence-start click mitigation; a millisecond count may be written bare
or quoted in TOML (`audio_onset_trim = 120` or `"120"`), and the saved configuration
stores it as the string.
Saved text-conditioning sidecars can
be reused with `--text-conditioning PATH` to skip Gemma and connector compute.
Native post-processing is selected with `--vsr-spatial-mode` and
`--target-fps`. When balanced VSR or frame-rate conversion retains adjacent-
frame history, lightweight scene-cut detection is enabled by default so a
generated cut cannot leak the previous scene into the next one. Use
`--cut-detect-mode off|simple|hist` and `--cut-detect-threshold FLOAT` to
override that terminal policy; detection does no frame work when the active
native chain is stateless.
`--vae-tiling auto|single|custom` controls decoder memory policy, while
`--vae-decode-dtype auto|bfloat16|float16|float32` controls decoder compute precision.
In TOML, tile geometry without an explicit mode selects `custom`; this makes
intentional geometry active instead of silently ignoring it. Selecting `auto`
or `single` clears stored geometry, including geometry inherited from a lower-
precedence source and inactive defaults emitted by older full config dumps.
Custom mode requires at least one positive temporal or spatial tile size; zero
disables that axis.
The dtype default is recipe-aware: BF16 for SDR and FP32 for HDR.
With `--video-vae diffusion`, `--vae-decode-dtype float16` enables experimental FP16
decoding. One 512x512, 49-frame decode comparison on M1 Max took about 9% less time
than BF16, with similar peak memory and small pixel differences. Broader content
and HDR quality are unvalidated; keep `auto` for the default precision behavior.
This option changes the video decoder independently of `--transformer-dtype` and
does not support the convolutional VAE. API requests use
`DistilledRequest(vae_decode_dtype="float16", ...)`; TOML uses
`[generate].vae_decode_dtype`.

The automatic tiler accounts for the resolved dtype. Both convolutional and diffusion VAE
auto plans try a full decode first, then all temporal-only options; spatial
tiling is the last resort when temporal-only decoding cannot fit. They use Metal's
reported GPU limit, including `iogpu.wired_limit_mb`, minus live MLX allocations.
This decode headroom is bounded by macOS's estimated free plus reclaimable
file-backed and purgeable memory. Unused MLX buffers are released first;
filesystem cache is not treated as permanently occupied RAM. No fixed 16 GB
cap or half-budget reserve is applied. The log distinguishes the GPU ceiling
from current estimated reclaimable headroom; neither is an allocation guarantee.
`--stream-transformer` enables the constrained-memory 8-resident
block preset. Phase-completion lines include the peak MLX allocator memory for
that phase. `--save-run-log` (also enabled by `--save-all-sidecars`) records the
same non-synchronizing boundary samples, resetting the peak counter after each
sample, plus the VAE-entry memory accounting and resolved tile geometry. The
empty resolved plan replays with `--vae-tiling single`; otherwise the recorded
temporal/spatial tile and overlap values can be passed back with
`--vae-tiling custom`. The diffusion decoder additionally records its seed,
decoder-load outcome, and exact bounded-attention working-set maxima. Its
stage-4/stage-5 auto tiler uses the same measured budget, with dtype-aware
workspace and output-assembly estimates; the earlier deterministic stages still
see the complete latent. Every option
is also available through TOML and typed `--set`.
Infrastructure values use `[settings]`; checkpoint and transformer policy use
the selected model's `[model_settings]` table, and opt-in model tensor outputs
use `[model_artifacts]`. For example,
`--set model_settings.transformer_dtype="float16"` changes only LTX-2 policy.
Stale transformer keys under `[settings]` are rejected instead of translated.

### HDR generation

An HDR output setting alone cannot generate HDR from a text prompt. It requires
the supported HDR conditioning inputs for the selected version. These are the
currently validated paths; combinations outside them are refused rather
than approximated, and text-only HDR is refused for both generations
(there is no validated way to anchor the HDR signal from a prompt
alone):

- **LTX-2.3 converts SDR video to HDR video.** The HDR IC-LoRA is a
  video-to-video converter: `--hdr` plus `--hdr-reference <sdr video>`
  and a LoRA whose safetensors metadata declares `hdr_transform=logc3`.
  The reference must cover the full requested frame count - the fused
  adapter runs off-distribution on frames without a reference track, so
  a short reference is refused.
- **LTX-2.5 generates HDR video from an HDR still.** `--hdr` plus an
  EXR `--image` establishes the ACEScct working signal. The model
  preserves the condition's exposure distribution, so the EXR must
  genuinely carry above-reference-white content: an SDR-derived linear
  EXR yields an effectively SDR result in an HDR container (KinoMLX
  warns when it sees one). `kinomlx --model gmnet` produces suitable
  HDR plates from SDR stills; see the [GMNet model guide](GMNET.md).

Both paths write the half-float EXR master beside the video and a
BT.2020/HLG encode; `--save-hdr-heic-frames` adds viewable 10-bit
BT.2100 PQ frames.

#### LTX-2.3 HDR IC-LoRA

The supported adapter is Lightricks'
[LTX-2.3 22B IC-LoRA HDR](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR).
It is a separate gated repository from the LTX-2.3 base model, so open that
page while signed in, review its model card and LTX-2 Community License, and
accept or request access before downloading. The file KinoMLX uses is
`ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`:

```bash
hf download Lightricks/LTX-2.3-22b-IC-LoRA-HDR \
  ltx-2.3-22b-ic-lora-hdr-0.9.safetensors \
  --local-dir /models/ltx23-hdr
```

The `hf` CLI is optional. You can instead use the model page's Files and
versions tab to download that file into any folder. There is no adapter install
step: keep the safetensors file wherever you store models and pass it to
`--adapter`. The current official file already declares `hdr_transform=logc3` and
`reference_downscale_factor=1`; KinoMLX verifies those facts before loading the
base model. The repository's separate `scene-emb` file is not consumed by this
supported full-reference workflow.

Run the conversion with the LTX-2.3 base checkpoint, the adapter, and an SDR
video whose frame sequence aligns with the desired output:

```bash
kinomlx --model ltx2 --model-generation 2.3 \
  --adapter /models/ltx23-hdr/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors \
  --hdr-reference /videos/source-sdr.mov \
  --hdr SRGB_LINEAR \
  --prompt "Preserve the source scene with natural motion and detailed highlights" \
  --width 768 --height 448 --frames 121 \
  --output-dir outputs --output-prefix source-hdr
```

`SRGB_LINEAR` writes scene-linear Rec.709 EXR masters; use `ACESCG` when an
ACEScg EXR master is preferable. The reference must contain at least the
requested number of frames. KinoMLX reads the first requested frames,
aspect-preserving-resizes and pads them to the generation geometry, and refuses
a short reference rather than running the adapter without coverage. This
validated adapter recipe is video-only: do not add `--generate-audio`, image
conditioning, generated keyframes, adapter target exclusions, or transformer
streaming/quantization. The default adapter and reference strengths are `1.0`;
`--adapter-strength` and `--hdr-reference-strength` expose deliberate overrides.

#### LTX-2.5 native HDR

LTX-2.5 does not use the HDR IC-LoRA. Give the ordinary LTX-2.5 pack a genuinely
HDR EXR still as its image condition:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --image /images/hdr-plate.exr --hdr ACESCG \
  --prompt "A slow cinematic push through the illuminated scene" \
  --width 768 --height 448 --frames 121 \
  --output-dir outputs --output-prefix hdr-plate-motion
```

## Denoise schedules

The distilled pipeline denoises in two stages with independent pinned schedules:
nine literal sigma values drive stage 1's eight steps, and four drive stage 2's
three steps after the spatial upscaler. Both are reported in the sampling section
and recorded as `stage_1_sigmas` and `stage_2_sigmas` in the run metadata.

Each stage is overridden on its own:

```bash
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --sigmas-stage-1 "1,0.99,0.98,0.96,0.94,0.88,0.7,0.4,0" \
  --sigmas-stage-2 "0.9,0.72,0.42,0"
```

Overriding one stage leaves the other on its pinned values. An array holds one
more value than the steps it drives, so it also sets that stage's step count: the
nine values above keep stage 1 at eight steps while respacing them.
Values must be finite, lie between 0 and 1, and decrease strictly.

Stage 2 legitimately starts below 1.0: it continues from stage 1's tail rather
than from pure noise, which is why its pinned first value is `0.909375`. Keep that
in mind when writing a replacement - starting stage 2 at 1.0 would treat an
already-denoised latent as pure noise.

See [Denoise schedules](SCHEDULES.md) for the full contract, and for varying an
adapter's strength across a run.

## Library API

Use the Python API when you want to consume frames directly or integrate generation
with your own output handling. Preparing resources selects model files and settings
without keeping the models resident. Consume the output inside its context manager
so its frames and resources are released correctly:

```python
from kinomlx import Settings
from kinomlx.models.ltx2 import (
    DistilledRequest,
    LTX2Settings,
    generate_distilled,
    prepare_resources,
)

resources = prepare_resources(
    LTX2Settings.from_env(),
    infrastructure=Settings.from_env(),
)
request = DistilledRequest(prompt="A quiet coastal sunrise, cinematic light")
with generate_distilled(request, resources) as output:
    for frame in output.frames:
        consume(frame)
```

Applications that host multiple recipe calls can retain the same prepared
resources without retaining model weights:

```python
from kinomlx.models.ltx2 import LTX2Runner

runner = LTX2Runner(resources=resources)
with runner.run(generate_distilled, request) as output:
    consume(output)
```

Use `LTX2Runner.restart()` to decode saved latents or rerun stage 2 from Python.
Choose the corresponding `DistilledRestart` constructor and provide the saved
inputs required by that operation:

```python
from pathlib import Path

from kinomlx.models.ltx2 import DistilledRequest, DistilledRestart

request = DistilledRequest(
    width=768,
    height=448,
    frames=121,
    seed=43,
    generate_audio=True,
)

with runner.restart(
    request,
    DistilledRestart.decode(Path("outputs/source.safetensors")),
) as output:
    consume(output)

with runner.restart(
    request,
    DistilledRestart.stage_2(
        Path("outputs/source_stage1.safetensors"),
        text_conditioning=Path("outputs/source_text.safetensors"),
    ),
) as output:
    consume(output)
```

For the full explicit component schedule and a complete VideoToolbox write,
see [`examples/ltx2_distilled.py`](../examples/ltx2_distilled.py). Saved text
conditioning records the model and text-encoding settings needed to check replay
compatibility. Direct conditioning replay requires matching identities; explicit
restart can accept a replacement with compatible tensors and records the differences.
When supplying your own `GenerationSink`, pass an `OutputColorPlan` that describes
the actual signal. Changing its label does not convert SDR pixels into HDR or
transform one color space into another.
`VideoToolboxGenerationSink(vae_frame_directory=...)` exposes the same
pre-terminal PNG sequence to library hosts without involving CLI sidecars.

## Performance

### Transformer quantization

Use `--transformer-quantize mxfp8-blocks` to reduce transformer weight memory at
the cost of slower generation. It packs supported attention and feed-forward
weights throughout the LTX transformer into MXFP8, while `--transformer-dtype`
selects the compute dtype. It works with regular BF16 checkpoints; no separately
quantized checkpoint is required.

```bash
kinomlx --model ltx2 --model-generation 2.3 --prompt "A painted landscape." \
  --transformer-dtype float16 --transformer-quantize mxfp8-blocks
```

In measured M1 Max runs at 512x512, 49 frames, with audio and FP16 transformer
compute, whole-block MXFP8 reduced overall MLX peak memory from about 37 GiB to
23 GiB, with roughly 13% longer total runtime. Denoising memory fell by about
17 GiB; the text encoder then set the overall peak. These results cover LTX-2.3
and LTX-2.5, including an adapter on the latter. Savings and speed costs depend
on the model, output geometry, and hardware.

Quantization changes the weights and can change the generated result. Use it
when the memory saving is worth the speed and quality tradeoffs. Adapters are
fused into affected weights before packing. Leave quantization at its default,
`off`, for unquantized transformer execution.

The same setting is available as `transformer_quantize="mxfp8-blocks"` in
`LTX2Settings` and the `[model_settings]` TOML table, or as
`KINO_LTX2_TRANSFORMER_QUANTIZE=mxfp8-blocks` in the environment. The other modes are
`off` (dense weights) and `mxfp8-blocks-pretranspose` (packed weights with transposed
storage). The latter is experimental; benchmark it on your hardware before using it
as a speed optimization.

### Recorded generation and decode timings

On an M1 Max (64 GB), matched LTX-2.3 and LTX-2.5 768x448, 121-frame joint
audio/video runs each complete in about 2 minutes 30 seconds end to end. A
one-off LTX-2.3 1024x576, 481-frame production run measured about 23 minutes
wall.

For the same saved LTX-2.5 768x448x121 final latent in the decoder comparison,
the automatic convolutional decoder took 10.5 seconds and peaked at 28.7 GiB;
the optimized diffusion decoder took 73.5 seconds and peaked at 8.0 GiB. They
implement different published output priors and are not expected to be
bit-identical: convolution is the speed default, while diffusion is an explicit
lower-peak alternative whose output should be judged for the intended material.

## Limitations

- Distilled LTX-2.3/LTX-2.5 only: the two-stage 8 + 3 pipeline. Dev CFG/STG,
  DFR, general retake/video-to-video, and alternative public recipes remain out
  of scope. The one implemented video-to-video profile is LTX-2.3 SDR-to-HDR
  conversion with a full-length aligned SDR reference and the declared LogC3
  HDR IC-LoRA.
- Apple Silicon Macs only.
- Frame counts are `8*k+1`; `--duration` rounds up to the next valid
  count.
- Pre-alpha: interfaces and defaults can change between commits.

## Acknowledgments

KinoMLX builds on Apple's MLX framework and on the officially published
implementations of the model architectures it runs.

### Framework / tooling

- [MLX](https://github.com/ml-explore/mlx) by Apple - array/numerical framework (MIT)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Gemma 3 and Gemma 4 topology (MIT)

### Reference implementations

- [diffusers](https://github.com/huggingface/diffusers) - LTX-2 pipeline
  and model modules, co-authored by Lightricks and Hugging Face (Apache-2.0)
- [Transformers](https://github.com/huggingface/transformers) - Gemma 3 and Gemma 4
  reference implementation (Apache-2.0)

See [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for full license texts.

## Model Weights

KinoMLX is an inference engine. Model weights are **not** redistributed;
users download them from Hugging Face under their respective licenses.
KinoMLX contains no code from the Lightricks LTX-2 repository; model
behavior is implemented from the Apache-2.0 diffusers implementation,
checkpoint metadata, and the published architecture.

- **LTX-2.3** - by [Lightricks](https://huggingface.co/Lightricks/LTX-2.3),
  governed by the LTX-2 Community License Agreement. Non-commercial use
  is free; entities with annual revenue >= $10 M require a separate paid
  commercial license from Lightricks.
- **LTX-2.3 HDR IC-LoRA** - by
  [Lightricks](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR),
  distributed from its own gated model repository under the LTX-2 Community
  License Agreement.
- **LTX-2.5** - by [Lightricks](https://huggingface.co/Lightricks/LTX-2.5),
  governed by its published model license.
- **Gemma 3 12B** - by Google; KinoMLX defaults to Lightricks'
  [paired QAT-unquantized
  repository](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
  and can use Google's [vanilla repository](https://huggingface.co/google/gemma-3-12b-it).
  Both are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Gemma 4 12B** - by
  [Google](https://huggingface.co/google/gemma-4-12B-it); LTX-2.5 supplies its
  prepared text-encoder artifact in the split component pack. It is governed by
  the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## License

KinoMLX's own code is [MIT-licensed](../LICENSE). Model weights remain
governed by their respective upstream licenses (see above).

### Memory policy for API calls

The runner and public recipe functions apply the resource inventory's shared
memory policy before model work. Set `mlx_cache_limit_gb` and
`mlx_wired_limit_gb` through infrastructure `Settings`. Reusing resources
preserves those choices; each invocation reapplies them. For standalone
component loading, call `kinomlx.apply_memory_policy(resources.memory_policy)`
before allocating inputs or loading weights. See
[public API memory policy](PERFORMANCE.md#public-api-memory-policy).
