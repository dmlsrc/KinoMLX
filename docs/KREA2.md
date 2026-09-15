# Krea 2 Model Guide

Use Krea 2 to generate PNG or TIFF images from text, or edit one or two reference
images with a compatible edit adapter. Turbo defaults to eight guidance-free steps;
Raw defaults to 52 steps with guidance. Both run from local model files.

You can combine style adapters, select an optional 2x decoder, and save intermediate
results to rerun denoising or decoding without repeating earlier work. This guide
covers downloads, generation, editing, output choices, and Python usage.

Use `--prompt-file "prompts/edit.txt"` instead of `--prompt` to read one complete
UTF-8 prompt, including multiple lines. This works for generation and editing.
See [prompt-file behavior](../README.md#prompt-files) for validation, TOML, and
precedence. Literal `--prompt` text is never interpreted as a file path.

In this guide: [downloads](#model-access), [generation](#text-to-image-generation),
[editing](#image-editing), [restart](#restart-from-saved-stations),
[adapters](#adapters), [precision](#denoising-mode-and-precision), and [Python API](#python-api).

See [seed lists and model lifetimes](../README.md#seed-lists-and-model-lifetimes)
for ordered seeds, conditioning reuse, and image execution modes.

## Model access

Inspect the model cards, repository files, licenses, and access terms while
signed in:

- [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)
- [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw)

Accept or request access for each repository shown as gated. Authentication by
itself does not grant access to a gated repository. KinoMLX does not download
weights at runtime or redistribute them.

After access is approved, download the repository with the Hugging Face CLI,
excluding only the root monolith:

```bash
hf download krea/Krea-2-Turbo --exclude turbo.safetensors
hf download krea/Krea-2-Raw --exclude raw.safetensors
```

These commands omit only the root native monolith, the same transformer in its
native layout; the remaining repository files are small. Download the monolith
separately only when you intend to select it as an explicit transformer override
or use it in an external comparison.

## Local source selection

KinoMLX resolves the local `main` snapshot exactly and performs no network fallback.
The default variant is Turbo. Without a prompt, the command inspects resource metadata:

```bash
kinomlx --model krea2 --variant turbo
kinomlx --model krea2 --variant raw
kinomlx --model krea2 --source-path /path/to/Krea-2-Turbo
```

Replace a compatible logical component with `--tokenizer-path`, `--text-encoder-path`,
`--transformer-path`, or `--vae-path`. Select transformer and decoder precision
independently using `--transformer-dtype` and `--vae-dtype`; each accepts `bfloat16`,
`float16`, or `float32`. Reference denoising requires a BF16 or FP32 transformer.
Components load from their original source files and prepare their execution layouts
lazily, without generating an intermediate model file.

### Optional aesthetic 2x VAE decoder

KinoMLX supports the image-only
[spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)
decoder because Wan 2.1 and Krea/Qwen use the same 16-channel latent convention.
Download exactly the top-level checkpoint:

```bash
hf download spacepxl/Wan2.1-VAE-upscale2x \
  Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors
```

The repository also carries
`diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1/diffusion_pytorch_model.safetensors`.
It is the same decoder in Diffusers packaging; KinoMLX needs one representation,
not both. The bounded command above downloads the top-level file that KinoMLX
can normalize directly.

After downloading, select and discover it from the ordinary Hugging Face cache:

```bash
kinomlx --model krea2 --vae-decoder upscale2x \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 1024 --seed 42
```

An explicit normal VAE replacement is autodetected from its tensor schema and
12-channel decoder output, so this is equivalent without the selector:

```bash
kinomlx --model krea2 \
  --vae-path /path/to/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 1024 --seed 42
```

`--width` and `--height` remain the Krea denoising dimensions. The decoder
pixel-shuffles its packed RGB output at the end, so a 1024x1024 request saves a
2048x2048 image. Automatic and custom tiling account for this larger output;
custom tile size and stride continue to describe the pre-2x sample space.

Treat this much like an aesthetic adapter. It was trained primarily for
perceptual quality on real images and may make skin, hair, and local texture look
better. It can also introduce slight color shifts, excessive sharpness, or
invented texture, and its model card warns that anime, line art, and text may be
weak. It is not a neutral reconstruction oracle or a general-purpose
super-resolution guarantee. The stock `--vae-decoder qwen` path remains the
default.

All components use the shared [weight-loading contract](WEIGHTS.md).
Raw and Turbo read their original tokenizer, text-encoder, VAE, and transformer
sources.

## Text-to-image generation

### Four-step Turbo adapter

The optional community four-step Turbo adapter uses four denoise steps instead of the
standard eight-step Turbo recipe. Download it once into the Hugging Face cache,
or fetch the file by hand from
[lvladikov/Krea2-Turbo-Distill-4step-LoRA](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-4step-LoRA):

```bash
hf download lvladikov/Krea2-Turbo-Distill-4step-LoRA krea2_turbo_4step_rank_64_lora.safetensors
kinomlx --model krea2 --turbo-4step --prompt "A painted landscape."
```

KinoMLX resolves the exact release filename from the cached main revision; it
does not download adapters during generation. The profile supplies four steps,
guidance 0, mu 1.15, and strength 1 **as defaults**, filling only what the request
leaves unset. An explicit `--steps`, `--guidance`, `--mu`, or `--sigmas` is honored
and announced as a deviation from the distilled recipe, because leaving that
recipe is a legitimate thing to explore and a thing worth knowing you did. This
does not change the default eight-step Turbo recipe.

Strength and exclusions belong to this adapter independently of additional
style adapters:

```bash
kinomlx --model krea2 --turbo-4step --turbo-4step-strength 0.75 --turbo-4step-exclude output --adapter style.safetensors:0.8 --prompt "A painted landscape."
```

`--turbo-4step-exclude` accepts the same target categories as
`--adapter-exclude`; `none` disables exclusions. Exclusions deliberately change
the trained recipe. The corresponding request and `[generate]` TOML fields are
`turbo_4step`, `turbo_4step_strength`, and `turbo_4step_exclusions`.
Environment controls are `KINO_KREA2_TURBO_4STEP`,
`KINO_KREA2_TURBO_4STEP_STRENGTH`, and `KINO_KREA2_TURBO_4STEP_EXCLUSIONS`.
CLI and TOML values override environment defaults.

Raw is allowed with a warning: also supply a **standard 8-step Turbo adapter**
through `--adapter`. KinoMLX does not guess whether another adapter supplies
that behavior. The profile also works with `--source-image` and the edit adapter. Turbo adapter strength
and exclusions remain separate from `--edit-adapter-strength` and
`--edit-adapter-exclude`.

The selected repository, revision, filename, strength, exclusions, and actual
adapter placements are recorded. A denoise restart refuses a different selected
release; decode-only restart does not require the Turbo adapter.

### SDA diversity adapter

Turbo's distillation collapses sampling diversity: the same prompt tends toward
one composition across seeds. The community SDA adapter restores it. Download it
once and switch it on, or fetch the file by hand from
[F16/krea2-turbo-sda](https://huggingface.co/F16/krea2-turbo-sda):

```bash
hf download F16/krea2-turbo-sda krea2_turbo_sda_v1.0_comfy.safetensors
kinomlx --model krea2 --turbo-sda --prompt "A still life with fruit on a table."
```

The profile pins the trainer's own flattened export. The same repository ships a
diffusers-style conversion that mis-splits two text-fusion module names and
silently loses 32 of its 264 targets, so the switch cannot select it by accident.

The adapter is trained at a single high-noise level and degrades detail if it
stays active for the whole trajectory, so it is **gated by default** to the band
it records for itself, `0.93` upward. On the stock eight-step schedule that
selects steps 0 and 1, which is exactly the gate its model card prescribes:

```text
step 0 @ 1.0000   active
step 1 @ 0.9567   active   <- the level it was trained at
step 2 @ 0.9045   off
```

Because the gate follows the noise level rather than a step index, it keeps
tracking that band when the step count changes. Override it either way:

```bash
--turbo-sda-sigma-window "0.93:0.97"      # the literal training band, step 1 only
--turbo-sda-step-strengths "1,1,0,0"      # name the steps instead
--turbo-sda-strength 1.5                  # magnitude inside a window gate
```

A step array carries its own magnitude, so `--turbo-sda-strength` is refused
beside `--turbo-sda-step-strengths`: write the magnitude into the array. The
four-step profile follows the same rule.

`--turbo-sda-exclude` takes the same categories as `--adapter-exclude`. TOML
fields are `turbo_sda`, `turbo_sda_strength`, `turbo_sda_exclusions`,
`turbo_sda_step_strengths`, and `turbo_sda_sigma_window`. The four-step profile
gained the same two gate controls, so either adapter can be scheduled.

### Combining the two profiles

Both profiles can run together, and the combination is **not validated by
anyone**. The four-step adapter is distilled for its own schedule, and no level of
that schedule reaches the SDA's training band:

```text
4-step levels   1.0000, 0.9045, 0.7595, 0.5128
SDA band        0.93 to 0.97          <- nothing lands inside
```

KinoMLX says so on the console rather than running it quietly. Treat the
following as starting points to explore, not as recommendations:

| Combination | What it does |
| --- | --- |
| `--turbo-sda` alone | The adapter's native configuration: eight steps, gate on its own band. Everything its card measured. |
| `--turbo-sda --turbo-4step` | Default gate narrows to step 0 only, since step 1 falls below the band. |
| `--turbo-sda --turbo-4step --turbo-sda-step-strengths "1,1,0,0"` | The pairing reported to work in the adapter's own discussion thread: both early steps, straddling the trained level. |
| raise it to `--turbo-sda-step-strengths "1.5,1.5,0,0"` | Pushes magnitude instead of widening the gate. |
| add `--sigmas "1,0.9567,0.7595,0.5128,0"` | Moves step 1 onto the trained level, keeping the distilled tail. Buys diversity, and costs sharpness. |

On a small local panel the last row measured the most seed-to-seed diversity and
the least high-frequency detail, and a schedule-only control showed **most of that
detail cost came from moving the schedule, not from the adapter**. A smaller move
to `0.93`, just far enough to enter the band, cost nothing measurable on its own.
None of that was a large enough panel to rank the middle options, and the gain
depends strongly on the prompt: a wide-open prompt has composition for the
adapter to vary, while one that already pins subject, wardrobe, and location has
little. Measure on your own prompts before settling.

### Standard recipes

Turbo is the default variant. Its default recipe uses eight denoise steps,
guidance `0.0`, and fixed scheduler shift `mu=1.15`:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 1024 \
  --seed 42
```

Raw defaults to 52 steps, guidance `3.5`, and a resolution-dependent dynamic
shift. Its unconditional branch uses an empty negative prompt unless one is
provided:

```bash
kinomlx --model krea2 --variant raw \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --negative-prompt "low contrast" \
  --width 1024 --height 1024 \
  --seed 42
```

Override the recipe with `--steps`, `--guidance`, and `--mu`. Guidance zero
skips the negative text branch and every unconditional transformer call.
Requested dimensions are rounded up to Krea's 16-pixel execution alignment and
cropped back exactly at output. VAE decoding defaults to `--vae-tiling auto`:
use a full image when its estimated working set fits, otherwise choose the
largest fitting spatial tile. LTX2 and Krea 2 share the same memory budget:
Metal's reported GPU limit minus live MLX allocations, bounded by macOS's
estimated free plus reclaimable file-backed and purgeable memory. This honors
`iogpu.wired_limit_mb`, releases unused MLX buffers before measuring, and does
not count filesystem cache as permanently occupied RAM. The estimate does not
include swap capacity or guarantee that every file-backed page is immediately
discardable.

Use `--vae-tiling single` to force full-image decoding, or
`--vae-tiling custom --vae-tile-size 256 --vae-tile-stride 192` for an explicit
tile layout. Size and stride apply only to custom mode. The log shows the GPU
ceiling, live allocations, estimated reclaimable headroom, and chosen plan.

Repeat `--prompt` to form an ordered batch. A single `--negative-prompt` is
broadcast across the prompt batch, while repeated negative prompts must match
the positive batch. `--images-per-prompt` adds independently seeded variations;
image `i` uses `seed + i`.

### Setting the schedule explicitly

Both variants derive their schedule by shifting a uniform 1-to-0 ramp. Supply the
noise levels yourself to replace that:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --sigmas "1,0.9567,0.9045,0.8403,0.7595,0.6547,0.5128,0.3109,0"
```

The array holds one more value than the steps it drives, so it also fixes the
step count: nine values run eight steps, whatever `--steps` resolved to. `mu`
shapes a generated schedule only, so it has no effect here and the sampling
section reports it as unused. `--turbo-4step` brings its own schedule, and an
explicit array is an advanced override that replaces it: the pairing is accepted
and the array decides the levels and the step count. A step array written for the
profile's own step count still applies per step. TOML uses `sigmas`.

Values must be finite, lie between 0 and 1, and decrease strictly. A final value
above zero is allowed and simply stops the denoise early, leaving residual noise
in the latent.

Every run reports its resolved schedule in the sampling section and records it
under `denoise.sigmas` in the run metadata, whether the schedule was generated or
supplied.

See [Denoise schedules](SCHEDULES.md) for the full contract, and for varying an
adapter's strength across a run.

### Regular text system instruction

`--system-prompt "Describe the visible objects and lighting."` overrides the Qwen system
instruction for regular text generation. The matching TOML field is `[generate].system_prompt`;
the Python API accepts `Krea2Request(system_prompt=...)`. Omit it to keep Krea's training default.
An explicit empty string (`--system-prompt ""`) keeps an empty system message. Whitespace in a
custom instruction is preserved. Image editing uses the separate `--grounding-system-prompt`;
the regular option is rejected for image editing to avoid an ineffective selection.

KinoMLX measures the complete custom prefix with the selected tokenizer, retains the complete
instruction, and recalculates prefix removal, middle padding, masks, and cumulative positions.
The denoiser still receives 512 conditioning positions. A longer instruction increases the text
encoder's sequence length and work. Custom role boundaries are tokenized separately from prompt
text so leading prompt whitespace cannot change the prefix slice. Default encoding is unchanged.
Instructions use the same 65,536-character input limit as prompts, and the complete sequence must
fit the bundled Qwen encoder's 262,144-token context.

Saved text conditioning records the instruction and prefix token count. Reusing a custom text
artifact requires the same `system_prompt`; run-record restart inherits that selection. Changing
the instruction requires fresh text encoding and is rejected at a saved denoise/decode boundary,
including when other provenance is treated as advisory. Custom instructions are experimental:
correct bookkeeping does not make an instruction equivalent to the one used during training.

The default `--noise-backend mlx` is the normal generation path. The optional
`torch-mps` choice implements the pinned PyTorch MPS random stream directly in
MLX Metal so a seed can replay oracle noise exactly. Both produce standard
normal noise; changing the backend changes the seeded sample, not the model's
compatibility or training distribution.

Use `--output image.png` for an exact single-image path. For a batch, that path
becomes the stem for numbered siblings. `--output-dir` and `--output-prefix`
select a timestamped bundle instead, following the
[shared app output naming policy](../README.md#output-names-and-sidecars).
The default stem is `krea2-<variant>-seed<seed>_YYYYMMDD_HHMMSS`; same-second
collisions add `_2`, `_3`, and so on. All images and sidecars use that one stem.
Existing explicit targets are refused unless `--force` is set; generated names
always select a fresh bundle. PNG, TIF, and TIFF are the supported lossless formats.

`--save-all-sidecars` enables the reusable Qwen text conditioning, unpacked
initial noise, final VAE latents, effective TOML, human-readable console log,
and structured run JSON. Use `--save-text-conditioning`,
`--save-initial-noise`, or `--save-final-latents` to select tensor products
individually. Tensor sidecars are separate for each output image, with the same numbered
and seed-labeled stem as that image. No combined multi-seed tensor file is written.
The console shows stage times and peak MLX memory, one denoise bar
with average time per step and ETA, and the selected configuration. Transformer
binding and optional adapter fusion stay lazy until the first denoise step. VAE
load includes weight materialization.

The run JSON additionally records the exact variant recipe, seeds, geometry,
noise compatibility profile, and allocator samples. Each timed phase's
`advances` list contains its completed count and wall-clock interval. Denoise
advances occur after MLX evaluation, so each interval includes the complete
step (both conditional and unconditional forwards when CFG is enabled).
Nested stage times overlap their parent and should not be added together.

## Image editing

Download the community adapter separately, or fetch the file by hand from
[conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit):

```bash
hf download conradlocke/krea2-identity-edit krea2_identity_edit_v1_2_r128.safetensors
```

Editing automatically resolves this file from the exact cached `main` snapshot
under the configured `HF_HOME`. It does not download missing files or pick an
arbitrary older snapshot. Supply a source image and instruction:

```bash
kinomlx --model krea2 \
  --source-image source.png \
  --prompt "Change the jacket to blue, preserving the person and background." \
  --output edited.png
```

Use `--edit-adapter /path/to/another-edit-adapter.safetensors` to override the default.
Use `--edit-adapter-strength` to control that adapter independently (default 1).
Use `--edit-adapter-exclude text_fusion,to_k` for edit-only knockouts. It accepts the
same categories as `--adapter-exclude`, works with cached or explicit edit adapters,
and does not change additional adapters. `none` disables these knockouts.
Explicit adapters go through the same structural validation and fusion. The run
records the selected default repository, exact revision, filename, and adapter
receipt. Default selection belongs to the Krea 2 edit recipe, not to the model's
encoder or attention code.

Additional style/character adapters use `--adapter`; they do not disable or change
the selected edit adapter. Their strengths default to 1. Each `--adapter-strength`
applies to the `--adapter` it follows, so give one value per additional adapter in
matching order; repeating the flag for the same adapter is refused as a duplicate
member assignment. Ordinary adapter updates are summed with their individual
strengths, not averaged. Listing the edit adapter again under `--adapter` is rejected
to prevent applying it twice.

```bash
kinomlx --model krea2 \
  --source-image source.png \
  --prompt "Restyle the scene while preserving the subject." \
  --adapter /path/to/style.safetensors \
  --adapter-strength 0.35 \
  --output edited.png
```

This uses the cached edit adapter at strength 1 and the style adapter at 0.35.

Turbo defaults to eight steps and guidance zero. For the community Raw editing
recipe, add `--variant raw --steps 20 --guidance 2`. Kino guidance is an offset:
`conditional + guidance * (conditional - unconditional)`, so guidance 2 is the
equivalent of conventional CFG 3. Each guided branch sees the reference images.

Repeat `--source-image` for two references, in scene then subject order. Use
`--reference-strength` for the sole reference or the second reference, and
`--scene-reference-strength` for the first of two references. Both default to 1.
These control target-to-reference attention; `--edit-adapter-strength` separately
controls the weight update. Output dimensions use the normal `--width` and
`--height` options; one output image is supported per edit call.

`--match-source-aspect` keeps the source image's aspect ratio instead of the
square default. Its value names the dimension that is pinned: `long` (the
source's longer side, also what the bare flag means), `short`, `width`, or
`height`. The pinned dimension gets the one size you pass with `--width` or
`--height`, or 1024 with neither; the other dimension follows the source. For
`width` and `height` the size must come from that same option; for `long` and
`short` either option supplies the number. Passing both sizes is an error. Both
sizes round to the 16-pixel grid, the pinned one up and the derived one to
nearest. A 720x1280 phone photo edits at 576x1024 with the bare flag instead of
being centered in a 1024x1024 canvas with outpainted sides, and each denoising
step costs about a third less because the reference and the target both shrink;
`short` would make it 1024x1824 instead.

The default `--reference-fit fit` preserves source aspect ratio and centers its
latent positions in the target grid. Almost-matching aspect ratios fill the
target after a small center crop. Use `--reference-fit crop` for legacy adapters
trained with a full target-sized center crop. The visual language path caps the
longest source side at 768 pixels; `--grounding-px 0` disables this preliminary cap.

No new VAE weights are required: the original Krea VAE checkpoint already contains
the encoder. Retain the complete original `vae/` and `text_encoder/` components;
editing uses the image encoder and vision tower as well as the language model.
Their arrays are released before denoising. Output decoding loads the selected
decoder when needed. A decoder override affects output decoding; source encoding
uses the baseline Krea VAE.

`--save-all-sidecars` retains initial noise, final latents, and execution records
for edits. Final-latent decode restart works as usual. Grounded text alone cannot
restart editing because it lacks the clean reference latents, so edit runs omit
the text-only restart artifact and reject `--save-text-conditioning` by itself.
Source-image paths and reference settings are recorded in the run metadata.

Use `--reference-boost-mask subject-mask.png` with `--reference-strength` to restrict the last
reference's attention adjustment to selected regions. White selects a region; black leaves its
attention unchanged. RGB masks use channel mean, and alpha is ignored. The mask must match the
last source image dimensions. It is cropped with the same source-image crop box, then
area-resized to the reference token grid and thresholded strictly above 0.5. This intentionally
differs from the reference node so source-painted masks and fitted-view coordinates align. The
first of two references still uses the whole-image `--scene-reference-strength`. Selections at
strength 1 are rejected; set `--reference-strength` to a value other than 1. Missing mask files
are rejected before resource preparation. This is a reference attention mask, not a target
inpainting mask or a guarantee that other output regions stay unchanged.

Geometric regions can replace or supplement a mask image. Repeat
`--reference-boost-region` to combine regions by union (including any image-mask
selection). All numbers are percentages from 0 to 100; `%` suffixes are optional.
Coordinates start at the top-left of the fitted reference view seen by the
transformer, after source fitting/cropping. Shapes select reference token centers;
very small regions can contain no token centers.

| Shape | Example | Coordinates |
| --- | --- | --- |
| Rectangle | `--reference-boost-region "rect:35,15,65,55"` | left, top, right, bottom |
| Circle | `--reference-boost-region "circle:50,30,15"` | center x, center y, radius as percent of the shorter side |
| Ellipse | `--reference-boost-region "ellipse:50,30,15,20"` | center x, center y, horizontal radius as percent of width, vertical radius as percent of height |
| Polygon | `--reference-boost-region "polygon:30,15;70,15;80,60;20,60"` | x,y vertices separated by semicolons; closes automatically |

`rectangle` and `poly` are aliases for `rect` and `polygon`. Quote region strings,
especially polygons, so the shell does not interpret semicolons. Polygons accept
clockwise or counterclockwise vertices and concave outlines; self-intersections
use the even-odd fill rule. Portions of circles/ellipses outside the view are clipped.
Like image masks, regions affect only the last reference, need a reference strength
other than 1 to have an effect, and do not restrict where the output can change.
Shapes are rasterized directly to a compact token selection once before sampling,
without loading or allocating a full-resolution mask image.

Use `--grounding-system-prompt "Describe facial identity and clothing details."`
to override the Qwen grounding system instruction. Empty or whitespace-only text
uses the training default. The same instruction applies to positive and negative
grounded conditioning. It steers language conditioning rather than changing the
vision tower. Both options are recorded in config/run sidecars.

```bash
kinomlx --model krea2 \
  --source-image source.png \
  --prompt "Change the jacket to blue." \
  --reference-strength 1.5 \
  --reference-boost-mask subject-mask.png \
  --grounding-system-prompt "Describe facial identity and clothing details." \
  --output edited.png
```

Direct preencoded `source_latent` inputs are not exposed. The recipe restricts
output to one image per call and does not expose ComfyUI's arbitrary
sampler/partial-denoise composition.

The Python API selects editing through `Krea2Runner.generate(Krea2Request(...,
source_images=(Path("source.png"),)))`; supply adapter selections as for ordinary
generation. `generate_krea2_edit` is also exported as an explicit recipe. Image
encoding, visual features, reference geometry, and attention have no adapter-name
checks. Adapter dialect validation and fusion stay in the adapter loader. Future
native editing recipes can reuse those components without pretending that a
future model has the Krea 2 architecture.

## Invocation files

The same requests as TOML files. Run one with `kinomlx --config <file>`; explicit CLI
flags and `--set` overrides still take precedence, and `kinomlx config init --model krea2`
writes an annotated template that documents every field and table.

A Turbo generation with the four-step profile and a style adapter, matching the
[four-step Turbo adapter](#four-step-turbo-adapter) example:

```toml
model = "krea2"

[model_settings]
variant = "turbo"

[generate]
prompt = "A painted landscape."
width = 1024
height = 1024
seed = 42
turbo_4step = true

[[generate.adapters]]
path = "style.safetensors"
strength = 0.8

[output]
directory = "outputs"
prefix = "painted-landscape"
save_all_sidecars = true
```

An edit that restyles a source image, matching the [image editing](#image-editing) example:

```toml
model = "krea2"

[generate]
prompt = "Restyle the scene while preserving the subject."
seed = 42
source_images = ["source.png"]
match_source_aspect = "long"

[[generate.adapters]]
path = "style.safetensors"
strength = 0.35

[output]
path = "edited.png"
save_all_sidecars = true
```

```bash
kinomlx --config painted-landscape.toml
kinomlx --config restyle.toml --set generate.seed=7
```

## Restart from saved stations

Krea has one denoising stage, so it has two meaningful restart boundaries:

- `denoise` consumes saved Qwen text conditioning, skips tokenization and text
  encoding, then repeats noise creation, transformer denoising, and VAE decode.
- `decode` consumes saved final latents and repeats only VAE decode. It is the
  default restart phase.

A source run must have the artifact needed by the selected boundary. Save both
boundaries and the authoritative run record with `--save-all-sidecars`, or use
the individual flags:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --save-text-conditioning \
  --save-final-latents \
  --save-run-log
```

### Resume after text encoding

Use the run JSON to inherit the source prompt batch and select its saved text
product:

```bash
kinomlx --model krea2 \
  --restart /path/to/source_run.json \
  --restart-from denoise
```

This skips tokenizer and Qwen execution. Seed, noise backend, dimensions,
steps, guidance, scheduler shift, denoiser adapters, transformer dtype, VAE,
and output settings can be changed. The source variant and prompt batch remain
fixed because the saved file already contains their positive and optional negative
conditioning. A replacement sidecar with
the same batch size can be selected explicitly:

```bash
kinomlx --model krea2 \
  --restart /path/to/source_run.json \
  --restart-from denoise \
  --text-conditioning /path/to/replacement_text.safetensors \
  --seed 99
```

Without a run record, `--text-conditioning` is also the direct LTX-style replay
surface. The artifact header supplies its prompts, so `--prompt` can be omitted:

```bash
kinomlx --model krea2 \
  --text-conditioning /path/to/source_text.safetensors \
  --seed 99
```

A run that produced several images records one path per image for every sidecar, so a
restart selects which image to resume with `--restart-index` (one-based, matching the
output number in the filename). The resumed request inherits that image's prompt and
seed alone; without an index a multi-image run is refused with the range to choose from.
`--restart-latents` still substitutes a different tensor file for the selected image.

### Restarting a run you interrupted

A run stopped with Ctrl-C publishes every image it had already finished, and its run
record states how many images it set out to write. Such a record is restartable for
each of those finished images, exactly like a completed run: select one with
`--restart-index` and the saved digests still match, because they were keyed by the
planned count rather than by what the run managed to publish.

Selecting an image the run never reached is refused by name, for example
`image 2 of 3 was not published before cancellation`, and a run interrupted before its
first image is refused the same way. A run that *failed* is not restartable at all: it
stopped for a reason it cannot describe, so nothing it left behind is offered to a
restart.

Each generated image has its own saved conditioning sidecar. Select that file with
`--text-conditioning` and use a seed or seed list to generate new variations. Earlier
multi-row conditioning files remain readable: omit `--prompt` and the request inherits
the file's saved prompt rows, one image per row, then a seed list multiplies each row.
A replay must still expand to exactly the saved row count, so passing one `--prompt`
with `--images-per-prompt` greater than one against a multi-row file is rejected;
inherit the saved rows or select the per-image sidecar you want instead.

Ordinary direct replay requires the saved tokenizer and Qwen source, graph, dtype, and selected
layers to match the prepared resources. The recorded Raw or Turbo variant and denoiser recipe
are descriptive because they do not participate in text encoding; a Raw artifact's negative
branch is ignored when guidance is disabled, while a guidance-enabled replay still requires one.
An explicit nonempty `--negative-prompt` is rejected when guidance is zero. In the library API,
`request.prompts` and `request.negative_prompts` expose the resolved rows; omitted `prompt` and
`negative_prompt` inputs remain `None`, so replacing a request preserves whether those rows came
from the saved product. Run-based restart lets you deliberately supply replacement conditioning.
Source identity differences are recorded rather than automatically rejected, but the replacement
still needs compatible tensors: batch and token counts, selected layers, hidden width, dtypes,
positions, and finite hidden states must match the required input. An arbitrary embedding file
is not interchangeable with Krea text. A run that did not save `outputs.text_conditioning`
cannot use `--restart-from denoise` unless `--text-conditioning` supplies a compatible
replacement. A replacement can also be used after the original sidecar has been deleted: parent
prompt checks use the run's recorded generation context. Older records without both prompt
metadata fields still need the original sidecar for those checks.

The denoise restart writes beside the source run with a `_denoise` suffix and a
fresh timestamp by default. Saved text is an input to this phase rather than a
new output, so `--save-text-conditioning` is rejected for run-based denoise
restart. Newly generated initial noise and final latents can still be saved.
`--save-all-sidecars` selects those two tensors plus the run log, console log,
and effective configuration. Individual sidecar overrides still apply.

### Decode final latents

Use the run JSON as the authoritative source for the saved latent and original
request geometry:

```bash
kinomlx --model krea2 \
  --restart /path/to/source_run.json \
  --vae-decoder upscale2x
```

This loads the recorded `final_latents` safetensors artifact, skips
tokenization, text encoding, transformer loading, and denoising, then decodes
through the selected VAE. The prior invocation is the lower-precedence config
layer, so omitting `--vae-decoder` reuses its decoder selection. Use
`--vae-decoder qwen` or `--vae-decoder upscale2x`, `--vae-path` for an explicit
compatible replacement, and the normal `--vae-dtype` and `--vae-tiling`
options to change decoding without rerunning denoising.
A newly supplied `--vae-path` uses decoder autodetection unless this invocation
also explicitly selects a decoder. An explicitly conflicting path and decoder
selection is rejected.

By default, the restarted image is written beside the source run with a
`_final_decode` suffix and a fresh timestamp. `--output`, `--output-dir`, and
`--output-prefix` retain their normal meanings. To pair the run configuration
with a compatible edited or copied sidecar, substitute it explicitly:

```bash
kinomlx --model krea2 \
  --restart /path/to/source_run.json \
  --restart-latents /path/to/edited_final_latents.safetensors \
  --output /path/to/decoded.png
```

Restart validates the safetensors envelope before materialization and requires exactly one
finite BF16, FP16, or FP32 `final_latents` tensor with the batch, channel, and latent geometry
recorded by the source request. A changed sidecar hash or metadata is recorded but does not
alone prevent loading; incompatible tensor structure or nonfinite values do. Initial noise and
saved text are not produced by a decode-only restart, but the final latents it decoded are
re-emitted, so the bundle it writes can be decode-restarted in turn. Here
`--save-all-sidecars` selects the run log, console log, effective configuration, and those
final latents. A denoise restart likewise re-emits the saved text it read back. Explicit
requests for unavailable tensor sidecars are rejected.

A run produced by text-conditioning replay can be decode-restarted even after
the original text sidecar is removed: its run record retains the resolved
prompt batch. Restart records can also seed another restart. When the selected
artifact is absent from a parent's outputs, its recorded consumed input is
used, without requiring earlier run records. `--restart-latents` still takes
precedence over both sources.

Successful tensor publication records output SHA-256 fingerprints in the run
log. A later restart compares its consumed input with the parent's recorded
hash when available; mismatches remain advisory. Older run records without
output fingerprints remain usable, and a fingerprinting error does not discard
successfully published outputs.

## Python API

The runnable [public API example](../examples/krea2_generate.py) prepares local
resources, calls `Krea2Runner.generate()`, and publishes PNG/TIFF images through
`Krea2OutputSink` before closing the result. Run it from a source checkout with
KinoMLX installed in your environment:

```bash
python examples/krea2_generate.py "A quiet coastal sunrise" outputs/turbo.png
python examples/krea2_generate.py "A quiet coastal sunrise" outputs/raw.png --variant raw
```

The example uses the variant's normal step count and guidance. Local Hugging Face
discovery works as described above; `--source-path /path/to/model` selects an
explicit local source. `--width`, `--height`, and `--seed` control the request.
Existing output files are refused. Library hosts can reuse `generate_images()`
with a typed request, optional model settings, and a progress reporter; batched
requests receive numbered output paths. The example is included in the source
distribution and can also run against an installed wheel outside the checkout.

The same recipe is available as a closeable library result:

```python
from kinomlx.models.krea2 import Krea2Request, Krea2Runner, Krea2Settings

runner = Krea2Runner(Krea2Settings(variant="turbo"))
request = Krea2Request(
    "A quiet coastal sunrise, cinematic light",
    width=1024,
    height=1024,
    seed=42,
)

with runner.generate(request) as result:
    print(result.images[0].pixels.shape)
    print(result.metadata["recipe"])
```

The corresponding typed restart entry points are:

```python
from pathlib import Path

from kinomlx.models.krea2 import Krea2Request, Krea2Restart

with runner.restart(
    request,
    Krea2Restart.decode(Path("source_final_latents.safetensors")),
) as result:
    print(result.images[0].pixels.shape)

saved_text = Path("source_text.safetensors")
replay_request = Krea2Request(text_conditioning=saved_text, seed=99)
with runner.restart(
    replay_request,
    Krea2Restart.denoise(saved_text),
) as result:
    print(result.metadata["execution_mode"])
```

Constructing or copying `Krea2Request` validates individual input fields without
opening source files. Generation, restart and output planning validate the final
combination and resolve saved conditioning before consuming it. To inspect inherited
prompt rows explicitly before execution:

```python
from kinomlx.models.krea2.request_sources import resolve_request_sources

resolved_request = resolve_request_sources(replay_request)
print(resolved_request.prompts)
```

Saved prompt rows appear in the resolved request's `prompts` and `negative_prompts`;
omitted `prompt` and `negative_prompt` inputs remain omitted. Copies contain their
own input-derived rows; resolve the copied request when inspecting saved context.

`generate_krea2(request, resources, ...)` is the stateless public recipe. Both
it and `Krea2Runner` accept an injected `Krea2Components` aggregate and an
injected `Krea2TextConditioner` for host integration and testing.

## Adapters

Use a local safetensors adapter with the normal generation command:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --adapter /path/to/style.safetensors
```

[Krea's fine-tuning guidance](https://github.com/krea-ai/krea-2#finetuning-krea-2)
recommends training adapters on **Raw** and running them on **Turbo**. That is
the intended pairing, not a mismatch. The log and run JSON distinguish this
pairing when the adapter declares its training model. A filename alone does
not establish the training variant.

Repeat `--adapter` to combine ordered adapter records. Each modifier applies to the latest
preceding CLI item. Omitted strength defaults to `1.0` for that item; strengths and exclusions do
not broadcast. The familiar `--lora`, `--lora-strength`, and `--lora-exclude` spellings are aliases.
A duplicate member assignment is an error, including aliases or equal values. Negative and zero
strengths are accepted:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --adapter /path/to/style.safetensors --adapter-strength 0.8 \
  --adapter /path/to/detail.safetensors --adapter-strength 0.3
```

`--adapter-exclude` accepts comma-separated target categories for the current item.
For example, `--adapter-exclude text_fusion`
leaves the denoiser's text-fusion weights unchanged. Categories are
`transformer`, `text_fusion`, `attn`, `ff`, `gate`, `norm`, `bias`, `modulation`,
`input`, `output`, `to_q`, `to_k`, `to_v`, `to_out`, and `to_gate`.
`gate` covers both attention `to_gate` projections and feed-forward gates.
`transformer` covers the 28 main blocks only; text-fusion, input, and output
projections have their own categories.
Krea has one denoise stage, so there are no per-stage adapter strengths.

TOML and ENV use one ordered `generate.adapters` collection. The API uses
`Krea2Request(adapters=(Krea2AdapterConfig(...), ...))`; both types are exported from
`kinomlx.models.krea2`. The former parallel-array fields are no longer accepted.

```toml
[generate]
adapters = [
  {path="style.safetensors", strength=0.8, exclude=["text_fusion"]},
  {path="detail.safetensors"},
]
```

Each item owns `path`, `strength`, `exclude`, `step_strengths` and `sigma_window`. A nonempty step
array supplies absolute per-step strengths and conflicts with an explicit strength or window on
that same item. A higher-precedence collection replaces the user collection; `generate.adapters=[]`
clears it while keeping independently enabled four-step, SDA and edit profiles. Their ordering,
defaults and pairing guidance still apply.

Use `--set 'generate.adapters[1].strength=0.25'` to edit the second existing item. Replace an entire
item to remove an optional control: `--set 'generate.adapters[1]={path="detail.safetensors"}'`.
Empty step arrays and empty windows are invalid. `KINO_KREA2_ADAPTERS` accepts the same array syntax.
Numeric `.safetensors` suffixes are CLI-only shorthand, independent of file existence; structured
paths remain literal. See the [shared adapter guide](ADAPTERS.md#inline-strengths) for examples.

Adapters apply to the Krea denoiser, including its text-fusion blocks, not the
separate Qwen text encoder or VAE.

The [shared adapter guide](ADAPTERS.md) lists the supported LoRA, LyCORIS, and
PEFT exports, their required saved state, and explicit conversion boundaries.
This includes low-rank, magnitude, Kronecker, rotation, sparse, shared-basis,
spectral, and full-weight formats. It does not treat every file with a LoRA
filename as the same algebra.

Standalone community adapters do not generally need a separate JSON file.
For a standard PEFT export, keep `adapter_config.json` beside
`adapter_model.safetensors`; Krea also accepts supported configuration embedded in
the safetensors file. See [required export files](ADAPTERS.md#keep-the-export-self-contained)
for the distinction. If loading reports missing bases, indices, or mappings,
obtain a complete export from the author; increasing strength will not repair
missing parameters.

### Alpha, strength, and unexpectedly strong adapters

User strength multiplies the adapter's format-defined update. The saved `alpha`
is not universally another multiplier. Dense/dense AI Toolkit and LyCORIS LoKr
exports use unit intrinsic scaling even when they record a very large alpha;
PEFT LoKr instead uses its configured alpha/r scaling. Treating these exports
identically would produce a different image or numerical overflow.

Saved alpha `0` does not disable a rank-scaled LoRA, LoHa, or LoKr update; use user
strength `0` for that. See the shared [scaling rules](ADAPTERS.md#strength-alpha-and-effective-scale)
for tensor/configuration precedence, saved scalars, rank-stabilized LoRA, and
numerical examples. Start with the author's strength recommendation rather than
compensate for an apparent alpha value yourself. Add any required trigger words
to your prompt; they are not inserted automatically.

### Coverage warnings and first-step cost

A coverage warning can explain why an adapter has less effect than expected.
Coverage below 50% warns, and 0% is explicitly reported. Unmapped updates warn even
when aggregate coverage is high. Check the model pairing and exclusions before
raising strength. The run JSON's `generation.adapters` records which tensors were
applied, excluded, unmapped, shared, or training-only.

Original checkpoint files remain unchanged. Adapter math is evaluated when the
denoiser first uses its weights, so denoising time includes materialization and
merging that the preparation timer does not capture. Both normal and reference
denoising support adapters. The transformer is released before VAE decoding to
reduce peak memory.

Adapter-modified active weights are checked after first use by default.
`--no-check-adapter-weights` disables that scan while per-step latent checks remain.
For format support, error handling, and what the checks establish, see
[the adapter guide](ADAPTERS.md#first-use-weight-checks).

## Denoising mode and precision

The normal denoiser reduces work on padding and prepares selected weight layouts
for faster matrix multiplication. This mode is recorded as
`compact_unmasked_pretransposed`. It can change the generated image relative to
reference execution because different matrix shapes change floating-point rounding.
Use the normal mode for generation; use reference mode when comparing against the
original padded execution.

The optimization keeps only valid text rows, including the five template suffix
tokens, removes denoiser sequence-alignment padding and masks, and pretransposes
the main feed-forward down-projection weights. Pretransposition preserves weight
values; removing padded rows is what changes matrix shapes. These details matter
when comparing timings or trying to reproduce a reference image from the same seed.

BF16 remains the default dtype. FP16 is an opt-in speed setting, not a memory
saving: both types use two bytes per value. For example:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --transformer-dtype float16
```

Use `--reference-denoise` to restore the original padded text/combined sequence,
masked attention, and untransposed projections:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --reference-denoise
```

Reference mode rejects a `float16` transformer before loading: masked padding
rows can overflow FP16 and contaminate valid attention outputs. Use BF16 or
FP32 for that mode. The setting is also available as
`Krea2Settings(reference_denoise=True)`, `[model_settings].reference_denoise`,
and `KINO_KREA2_REFERENCE_DENOISE`. `--no-reference-denoise` overrides an enabled
config/environment setting. Effective TOML, console output, and the run JSON
record the selected mode and precision.

In either mode nonfinite denoising results raise `Krea2DenoiseError` instead of
being decoded into an invalid image. If FP16 overflows, use BF16 or FP32 and check
adapter strengths. Disabling adapter weight scans does not disable latent checks.

Compaction changes matrix shapes and their floating-point rounding, so the
denoising trajectory and image may differ from reference mode even with the
same seed and noise backend. FP16 can introduce additional differences.
Pretransposing a weight does not change its values. Neither mode changes Qwen
tokenization/encoding, scheduler/CFG math, image-token geometry, or VAE decoding.
Prompts retain their original output order. Each seed runs separately through the
denoiser; `--seed-mode` controls model lifetimes between those calls. One image per
denoiser call is not a throughput cost at these token counts: the transformer is
already compute-bound at batch one, so per-image execution matches a combined tensor
batch while holding less activation memory.

The loader prepares the measured pretransposed projections lazily. Reference mode
restores source orientation in memory. Every selected dtype converts from the original
source, without an intermediate BF16 rounding step.

## Native text conditioning

Use `NativeKrea2TextConditioner` when a Python application needs text embeddings
separately from image generation. It accepts one prompt, a prompt batch, and an
optional same-sized negative-prompt batch. Positive and negative branches share
one Qwen load and execute sequentially to limit memory. The returned arrays are
evaluated, and the live Qwen model is released before the call returns.

```python
from kinomlx import Settings, apply_memory_policy
from kinomlx.models.krea2 import (
    Krea2Settings,
    NativeKrea2TextConditioner,
    prepare_resources,
)

infrastructure = Settings.from_env()
resources = prepare_resources(
    Krea2Settings(variant="turbo"),
    infrastructure=infrastructure,
)
apply_memory_policy(resources.memory_policy)
conditioning = NativeKrea2TextConditioner()(
    "A quiet coastal sunrise, cinematic light",
    resources,
)

print(conditioning.positive.hidden_states.shape)
# (1, 512, 12, 2560)
print(conditioning.provenance.to_metadata())
```

Passing `negative_prompts=` produces a second product during the same Qwen
lifetime. Supply negative prompts explicitly when encoding them separately. During complete
generation, `generate_krea2()` selects negative conditioning according to the Raw
or Turbo guidance settings.

Each prompt product contains:

- twelve selected Qwen hidden-state taps with shape
  `[batch, sequence, tap, width]`;
- the post-prefix attention mask and three-axis position IDs;
- the original prompt batch and a deterministic token identity.

The returned metadata identifies the prompts, model variant, text source, dtype,
and selected layers. Keep it with saved conditioning so a later replay can check
compatibility. It contains no copy of the model weights.

## Prompt length and text encoding

Long prompts can lose content during tokenization even though the prompt file was
read completely. The default conditioning length has 507 positions for prompt
tokens plus a five-token suffix. Prompt strings over 65,536 characters are rejected
before tokenization; that character limit is not a promise that all those characters
fit in the model's token budget. Put important instructions early in a long prompt.

The text template also matters for reproducibility. With default conditioning
length, Qwen receives 546 positions arranged as:

```text
[34-token instruction prefix | prompt | middle padding | 5-token suffix]
```

Padding stays between the prompt and suffix. The 34-token prefix is removed only
after encoding, leaving 512 conditioning positions. Removing padding or moving
the suffix before Qwen runs would change the text embeddings. API callers reusing
embeddings must preserve the accompanying masks and positions, not just the visible
prompt text.

For precise cross-runtime comparisons, KinoMLX applies NFC normalization, Qwen's
Unicode split, ByteLevel mapping, BPE merges, and literal added-token handling.
Text-only conditioning uses hidden states at layers 2, 5, 8, 11, 14, 17, 20, 23,
26, 29, 32, and 35; it does not run unused language-model output layers. Editing
also uses the vision components described in [image editing](#image-editing).
Matching the prompt string alone is insufficient when another implementation uses
a different template, tokenizer configuration, or hidden-state selection.

The tokenizer tables are derived from the model's original JSON files in memory.
You do not need to install Transformers or Tokenizers for generation, and changing
a generic chat-template file does not replace Krea's prompt layout. Use the
[system-prompt option](#regular-text-system-instruction) for supported instruction changes.

## Native transformer, decoder, and still output

For custom Python decoding or a custom public recipe, `NativeKrea2Components`
provides the denoiser and Qwen Image decoder. Each call returns a `ComponentLease`.
To avoid keeping multiple large models resident, close one lease
before opening the next and do not retain a separate reference to its model.
Use `lease.value(...)` for calls so closing the lease releases its ownership
before the next station loads. The provider itself retains no model.

The decoder accepts a normalized 16-channel Krea latent in BCHW order and
returns `Krea2DecodedRGB`: an evaluated, unclamped floating RGB product in BHWC
order. Pass this product directly to the output sink: it performs the `[-1, 1]` to
display-RGB transfer, clamp, byte conversion, and ImageIO encoding. Applying that
conversion yourself first would transform the pixels twice. PNG, TIFF, and TIF are accepted;
lossy still formats are refused.

This lower-level example starts from an already unpacked final latent when a
host needs direct component and output-station control below the restart recipe:

```python
from pathlib import Path

from kinomlx import Settings, apply_memory_policy
from kinomlx.models.krea2 import (
    Krea2StillOutputPlan,
    Krea2Settings,
    Krea2StillOutputSink,
    Krea2VAETilingConfig,
    NativeKrea2Components,
    prepare_resources,
)

resources = prepare_resources(
    Krea2Settings(variant="turbo", vae_dtype="bfloat16"),
    infrastructure=Settings.from_env(),
)
apply_memory_policy(resources.memory_policy)
components = NativeKrea2Components()
plan = Krea2StillOutputPlan(Path("image.png"))

with plan.reserve() as reservation:
    decoder_lease = components.image_decoder(resources)
    try:
        decoded = decoder_lease.value(
            final_latent,
            tiling=Krea2VAETilingConfig(),
        )
    finally:
        decoder_lease.close()
    Krea2StillOutputSink(plan).write(decoded, reservation=reservation)
```

Omit `tiling=` for full-image decode. The default tiling contract uses a
256-pixel tile with a 192-pixel stride and activates only when either output
dimension exceeds the tile size. It follows the upstream overlap blend and is
an explicit memory-quality tradeoff: tiled decoding bounds activation memory,
while full decoding retains global bottleneck-attention context.

Decode-only calls do not load the VAE encoder or unused temporal-resampling
weights. This lowers the memory needed to retry decoding a saved image latent.
Still decoding uses the equivalent current-frame convolution taps because there
is no preceding video frame; it does not require constructing a temporal history.

## Current limits

- Adapter support is limited to the denoiser and the serialized formats above.
  It does not imply support for every PEFT tuner, dynamic/timestep-dependent
  adapters, or arbitrary trainer configuration.
- Reference editing supports one or two source images with a compatible edit adapter.
  It does not provide a separate masked-inpainting workflow.
- Interfaces are pre-alpha and may change between commits.

### Memory policy for API calls

The runner and public recipe functions apply the resource inventory's shared
memory policy before model work. Set `mlx_cache_limit_gb` and
`mlx_wired_limit_gb` through infrastructure `Settings`. Reusing resources
preserves those choices; each invocation reapplies them. For standalone
component loading, call `kinomlx.apply_memory_policy(resources.memory_policy)`
before allocating inputs or loading weights. See
[public API memory policy](PERFORMANCE.md#public-api-memory-policy).
