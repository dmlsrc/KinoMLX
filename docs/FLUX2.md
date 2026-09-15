# FLUX.2 Klein 9B

KinoMLX generates base, distilled, and KV still images through the CLI and Python API, including
ordered reference editing and static adapters. Common image flags match Krea 2: prompts, source
images, fitting, output paths, sidecars, and saved-station restart use the same
spellings. Model-specific guidance and reference behavior are described below.

In this guide: [generation and editing](#generate-and-edit),
[downloads](#local-checkpoint-selection), [saved results](#saved-stations),
[precision and loading](#weight-loading-and-configuration),
[Python API](#native-image-api), and [reference KV memory and reuse](#reference-kv-lifetime).

See [seed lists and model lifetimes](../README.md#seed-lists-and-model-lifetimes)
for ordered seeds, conditioning reuse, and image execution modes. A saved-station restart
resumes one image; select it from a multi-image run with `--restart-index`.

## Generate and edit

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window." --seed 42
kinomlx --model flux2 --variant kv --source-image reference.jpg \
  --prompt-file edit.txt --match-source-aspect --height 1024 --save-all-sidecars
kinomlx --model flux2 --variant base --prompt "A painted landscape." \
  --negative-prompt "blurry" --guidance 4
```

### Base with a Turbo LoRA and a style adapter

A **Turbo LoRA** is a LoRA adapter trained to enable a faster base sampling recipe.
Use a Turbo LoRA trained for Klein base 9B and follow its recommended step count and guidance.
For a Turbo LoRA that supports four steps at CFG 4:

```bash
kinomlx --model flux2 --variant base --prompt "A painted landscape." \
  --negative-prompt "blurry" --guidance 4 --steps 4 \
  --adapter klein_turbo_lora.safetensors --adapter-strength 1 \
  --adapter style.safetensors --adapter-strength 0.8
```

Replace both adapter filenames with your actual local files. The four-step recipe depends on
the Turbo LoRA; `--steps 4` by itself does not turn the base model into a four-step model and is
not a suitable ordinary base recipe. Without a Turbo LoRA, keep the base default of 50 steps.
The separately distributed four-step model is selected with `--variant distilled`; "Turbo LoRA"
names an adapter, not another model variant.

### Common image controls

Normal output defaults to `KINO_FLUX2_OUTPUT_DIR`, then `KINO_OUTPUT_DIR`, then `outputs`, with
a timestamped `flux2-<variant>-seed<seed>` name. Use `--output image.png` for an exact PNG/TIFF
path, `--output-dir`, or `--output-prefix`. Existing outputs are protected unless an exact
`--output` is paired with `--force`; source files and path aliases remain protected even with
replacement. The complete image/tensor bundle is staged before publication, and ordinary
publication failures restore prior files. Generated-name collisions receive a new suffix.

Repeat `--prompt` for multiple prompts and use `--images-per-prompt` for sequential images with
consecutive seeds for each prompt. `--negative-prompt` accepts one value or one per prompt.
Prompt files resolve
to literal text before execution, so the effective config does not depend on later file edits.
Within one precedence layer, `--prompt` and `--prompt-file` are mutually exclusive.

`--source-image` is repeatable and reference order matters. With one source and no explicit
output grid, Klein matches its aspect ratio and fits the output grid automatically.
`--match-source-aspect` without a value means `long`; `long`, `short`, `width`, and `height`
have the same meaning as Krea. Supply at most one output dimension when matching aspect. With
both dimensions explicitly set, Klein preserves that grid. `--reference-fit fit|crop` controls
reference fitting, not output cropping. Multiple references use independent capped native grids
unless fitting is explicitly selected. Aspect matching rounds the derived grid to 16-pixel
alignment; an explicit width/height pair must already be aligned. The header reports the
effective canvas, ordered reference grids, crop boxes, H/W offsets, and bicubic resizing. In
receipts, `resized_size` is the intermediate image size before `crop_box`; `reference_size` is
the final resampled image supplied to the VAE. Very different source/canvas aspects can leave
substantial empty space with `fit`; use `crop` when filling the canvas matters more than
retaining the complete source.

### Adapter records

Each `--adapter` (alias `--lora`) starts a new item. Strength, exclusion, step-strength and
sigma-window modifiers apply to the latest preceding item. A modifier requires a local adapter
declaration; assigning the same member twice is an error, including aliases and equal values.
Omitted strength defaults to `1.0` for that item. Controls do not broadcast to other adapters.

`--adapter style.safetensors:0.8` is shorthand for a path plus common strength. Numeric suffixes
are recognized after `.safetensors` without checking whether the file exists. Nonnumeric suffixes
such as `:latest` remain literal path text. Klein accepts one numeric suffix; stage pairs are not
supported. Structured paths are always literal, so they can name a file ending in a numeric suffix.

TOML, ENV and `--set` use one ordered `generate.adapters` collection:

```toml
[generate]
adapters = [
  {path="style.safetensors", strength=0.8, exclude=["modulation"]},
  {path="detail.safetensors"},
]
```

The API uses `Flux2Request(adapters=(Flux2AdapterConfig(...), ...))`; both types are available from
`kinomlx.models.flux2`. Each record owns `path`, `strength`, `exclude`, `step_strengths` and
`sigma_window`. A supplied step array must be nonempty and match the final step count. A supplied
window must contain valid noise-level bands. A step array conflicts with an explicit strength or
window on the same item. Omitted strength remains omitted in saved configuration.

A higher-precedence collection replaces the lower collection. `generate.adapters=[]` clears user
adapters. An indexed edit such as `--set 'generate.adapters[1].strength=0.25'` changes the second
existing item and preserves its siblings. Replace an entire item to remove an inherited schedule:
`--set 'generate.adapters[1]={path="detail.safetensors"}'`. Repeated paths retain their order.
`KINO_FLUX2_ADAPTERS` accepts the same TOML array of records. The former parallel-array configuration
and API fields are no longer accepted.

Static LoRA, LoKr, LoHa, and the shared supported adapter catalog use fresh per-generation fusion.
Turbo LoRAs are base-model recipes; distilled and KV do not accept external CFG.
A filename alone does not determine adapter type. Unsupported or incomplete exports
produce a loading error; use the [adapter guide](ADAPTERS.md) to check export requirements.
Embedded metadata from both Diffusers model-level and pipeline-level LoRA exports is supported,
including the pipeline's `transformer.` configuration prefix. Configured PEFT LoRA uses
`alpha/r` (or `alpha/sqrt(r)` for rank-stabilized LoRA) before the selected strength.
Other exports can use unit scaling, particularly dense/dense LyCORIS or AI Toolkit LoKr.
See [strength, alpha, and effective scale](ADAPTERS.md#strength-alpha-and-effective-scale)
for the exact rules, saved-zero-alpha handling, and examples.
Klein's `transformer` exclusion covers the entire denoising transformer; `double` and `single`
select block families, `norm` selects Q/K normalization, and `modulation` includes the output
AdaLN projection. These categories describe Klein's graph, so their target sets differ from Krea.

Base CFG defaults to `--cfg-mode sequential`, which evaluates positive and negative branches
separately to limit activation memory. `--cfg-mode batched` evaluates the two branches together;
this can change memory and numerical rounding. It does not batch separate requested images.
Prompts exceeding 512 templated tokens are truncated with a notice, matching the reference path.

The `torch-mps` noise backend reproduces the Torch/MPS stream. An external tool using a
CPU generator will produce different noise from the same seed. To compare implementations,
use the same explicit noise and conditioning tensors rather than assuming seed equality
means input equality.

## Denoise schedules

Klein derives its schedule from an empirical shift over the target token grid, so
the same step count produces different noise levels at different resolutions. Both
the resolved schedule and the controls that produced it appear in the sampling
section, and the schedule is recorded per image in the run metadata.

Supply the noise levels to replace that derivation:

```bash
kinomlx --model flux2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --sigmas "1,0.85,0.6,0.3,0"
```

An explicit array ignores the token grid entirely and fixes the step count: five
values run four steps. Values must be finite, lie between 0 and 1, and decrease
strictly. A denoise-phase restart may change the schedule, as it may change the
step count.

See [Denoise schedules](SCHEDULES.md) for the full contract, and for varying an
adapter's strength across a run.

## Saved stations

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window." --save-all-sidecars
kinomlx --model flux2 --restart previous_run.json --restart-from denoise
kinomlx --model flux2 --restart previous_run.json --restart-from decode
kinomlx --model flux2 --restart previous_run.json --restart-latents replacement.safetensors
kinomlx --model flux2 --text-conditioning previous_text.safetensors --seed 43
```

Use `--save-all-sidecars`, or select `--save-initial-noise`, `--save-final-latents`,
`--save-text-conditioning`, `--save-run-log`, `--save-console-log`, and `--save-effective-config`.
Each selector also has a negated form, such as `--no-save-initial-noise`.
Initial-noise files include the Euler schedule; final
latents record their packed output grid. Run records include resource identities, adapter receipts,
artifact fingerprints, timings, and sampled memory. Output pixels are HWC RGB in `[0,1]`.

A conditioning artifact contains all 512 positive text states and, for active base CFG, the
negative states. For editing it also contains ordered reference latents, their positions, and
complete geometry. Saving an edit's conditioning therefore supports denoise replay without
reopening source images. Reference KV is never serialized; replay extracts fresh KV on step one.

Run-record restart resumes one image, selected from a multi-image source run with
`--restart-index`. Decode is the default and needs only final
latents and the recorded VAE source. Denoise needs the complete conditioning station and
the recorded VAE and transformer sources. It
permits new seed, steps, guidance within the saved branch contract, and adapters. Prompt context,
model settings, and edit geometry cannot change at this boundary. A replacement conditioning
artifact must match the saved prompt context. A replacement final-latent artifact must carry the
Klein tensor schema and a valid packed grid. Restart revalidates the required source and
preparation identities; those sources must still be available and unchanged.
Decode restart does not need the transformer source.

A run stopped with Ctrl-C publishes every image it had already finished, and its record
states how many it set out to write. Such a record is restartable for each of those
finished images, exactly like a completed run: select one with `--restart-index` and the
saved digests still match, because they were keyed by the planned count rather than by
what the run managed to publish. Selecting an image the run never reached is refused by
name, for example `image 2 of 3 was not published before cancellation`, and a run
interrupted before its first image is refused the same way. A run that *failed* is not
restartable at all: it stopped for a reason it cannot describe, so nothing it left behind
is offered to a restart.

Decode restart saves only artifacts it produces; save-all omits initial noise and conditioning.
Arbitrary mid-step restart, VAE tiling, and Krea-specific grounding/region controls are
unsupported. Unsupported flags fail rather than being silently ignored. Klein 4B, FLUX.2-dev,
inpainting masks, the FP8 text encoder, prompt upsampling/Mistral, built-in moderation and
watermarking, cross-generation reference KV, and simultaneous multi-image batches are outside
this implementation. Multiple requested images execute sequentially with batch one.
Reference-token count and estimated KV payload are reported before generation. The estimate is
not a guarantee that the entire generation fits in memory: weights, activations, attention
workspace, and other applications also consume memory. Large VAE attention calls use bounded
query chunks that retain the complete key context; the VAE still decodes the whole image and
convolution peaks remain.

## Local checkpoint selection

The profiles use the bundled Diffusers components from the official model repositories:

| Variant | Source | Sampling |
| --- | --- | --- |
| `base` | [Klein base 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | 50 steps, CFG 4 |
| `distilled` | [Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 4 steps |
| `kv` | [Klein 9B KV](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) | 4 steps; reference K/V reused within one generation |

Review the applicable model card and weight license before acquiring weights. KinoMLX only reads
existing local files; it never downloads missing checkpoints during inspection or preparation.
The 9B weights use BFL's [FLUX Non-Commercial
License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md). Section
2(e) requires content filtering or review of outputs for unlawful/infringing content before
distribution, display, transmission, or dissemination, and AI-generation/modification disclosure
to the extent required by applicable law. These are user-side requirements; KinoMLX does not
bundle a content filter. Consult the license attached to the checkpoint for its complete terms.
After accepting the applicable repository access conditions on Hugging Face, authenticate with
`hf auth login` if needed. Download the profile you want with the Hugging Face CLI:

```bash
hf download black-forest-labs/FLUX.2-klein-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-base-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-9b-kv --exclude "flux-2-klein-*.safetensors"
```

These commands omit only the duplicate BFL single-file transformer checkpoint. KinoMLX reads
the Diffusers component folders; the remaining README, license, and sample
images are small and stay in the download. The commands populate the ordinary Hugging Face cache
selected by `HF_HOME`; use `hf download --local-dir` only when you intend to pass that directory
as `--source-path`. Downloads require network access; generation and preparation remain local.

The default variant is `distilled`. Sources resolve through the selected repository's local
Hugging Face `refs/main`, using `--hf-home` or `HF_HOME`.

```bash
kinomlx --model flux2 --variant kv --dry-run
kinomlx --model flux2 --variant kv
kinomlx --model flux2 --variant distilled --transformer-dtype float16 --dry-run
kinomlx --model flux2 --variant base --transformer-dtype bfloat16 --dry-run
kinomlx --model flux2 --variant base --source-path /path/to/bundled-checkpoint --dry-run
```

A bundled source contains `model_index.json`, `scheduler/scheduler_config.json`, `tokenizer/`,
`text_encoder/`, `transformer/`, and `vae/`. Tensor components require their corresponding
`config.json` and safetensors file or shard index. Use the bundled BF16 9B components;
incompatible shapes, storage dtypes, and model configurations are rejected rather than converted
into another architecture. The selected variant controls sampling identity. Structurally
identical transformer layouts do not prove whether arbitrary replacement weights are base,
distilled, or KV; select their variant explicitly. An explicit contradictory `is_distilled`
declaration is rejected.

Use `--tokenizer-path`, `--text-encoder-path`, `--transformer-path`, and `--vae-path` to replace
individual components. Explicit overrides are inspected before default discovery. Supplying all
four overrides permits preparation without a baseline snapshot and uses the selected profile's
scheduler contract. A supplied `--source-path` still validates that bundle's pipeline and scheduler.

## Weight loading and configuration

Klein loads the original component files. The default transformer dtype is `bfloat16`,
preserving the bundled weights' precision. `--transformer-dtype float16` is an optional
speed optimization that changes numerical range and rounding; it does not reduce weight
memory compared with BF16. Text and VAE execute in BF16. Tokenizer tables
are derived in memory. The transformer binds lazily and materializes during denoising.
The measured FF-down and combined-output projections use contiguous pretransposed
weights; other projections retain source orientation. Dtype and layout transforms do
not produce files on disk.

Components load only the weights used for inference and are released between phases.
This reduces memory compared with keeping the text encoder, transformer, and VAE alive
together. Adapter and reference-KV memory still add to the denoising working set. See
[weight loading](WEIGHTS.md) for first-step costs and loading-error guidance.

```bash
kinomlx config init --model flux2 --output klein.toml
kinomlx --model flux2 --config klein.toml --variant kv --dry-run
kinomlx --model flux2 --variant kv --print-config --only-non-defaults
```

Use `--dry-run` to inspect local resources without loading weights. Normal generation
requires a prompt or saved station. Environment, TOML, CLI, and `--set` precedence,
`--quiet`, `--json`, and MLX memory limits use the shared infrastructure settings.

### Invocation files

The same requests as TOML files. Run one with `kinomlx --config <file>`; explicit CLI
flags and `--set` overrides still take precedence, and the annotated template above
documents every field and table.

A base generation with a Turbo LoRA and a style adapter, matching the example under
[Base with a Turbo LoRA and a style adapter](#base-with-a-turbo-lora-and-a-style-adapter):

```toml
model = "flux2"

[model_settings]
variant = "base"

[generate]
prompt = "A painted landscape."
negative_prompt = "blurry"
guidance = 4.0
steps = 4
seed = 42

[[generate.adapters]]
path = "klein_turbo_lora.safetensors"
strength = 1.0

[[generate.adapters]]
path = "style.safetensors"
strength = 0.8

[output]
directory = "outputs"
prefix = "painted-landscape"
save_all_sidecars = true
```

A KV reference edit:

```toml
model = "flux2"

[model_settings]
variant = "kv"

[generate]
prompt = "Turn the ceramic vase into blue glass, preserving the window and light."
seed = 42
source_images = ["reference.jpg"]
match_source_aspect = "long"
height = 1024

[output]
directory = "outputs"
prefix = "vase-edit"
save_all_sidecars = true
```

```bash
kinomlx --config painted-landscape.toml
kinomlx --config vase-edit.toml --set generate.seed=7
```

## Native image API

The public host follows the same pattern as Krea, LTX2, and GMNet: a typed request,
prepared resources or model settings, a reporter, and a typed generation result.

```python
from kinomlx.models.flux2 import Flux2Request, Flux2Runner, Flux2Settings

runner = Flux2Runner(Flux2Settings(variant="distilled"))
output = runner.generate(
    Flux2Request(
        prompt="A ceramic vase beside a window.",
        width=512,
        height=512,
        seed=42,
    )
)
image = output.images[0]
```

`Flux2Runner(resources=prepared, reporter=reporter)` uses already prepared resources
without re-inspection or preparation. Settings and prepared resources are mutually
exclusive. `run(recipe, request)` accepts a custom typed recipe; `generate(request)`
uses `generate_flux2`. The runner retains resource descriptions and reporting only,
not model tensors, results, or reference KV. Each generation owns component lifetimes.

`Flux2GenerationOutput.images` contains `Flux2ImageResult` objects in prompt/variation
order, with HWC RGB pixels in `[0,1]`, final packed latents, dimensions, reference
geometry, adapter receipts, complete conditioning, initial noise, and the Euler schedule.
The [typed example](../examples/flux2_generate.py) reserves a transactional output
bundle before constructing the runner and writes results through `Flux2OutputSink`.

For saved-station replay, call `runner.restart(request, Flux2Restart(...))`:

```python
from pathlib import Path
from kinomlx.models.flux2 import Flux2Request, Flux2Restart

output = runner.restart(
    Flux2Request(),
    Flux2Restart(
        phase="denoise",
        text_conditioning=Path("previous_text.safetensors"),
    ),
)
decoded = runner.restart(
    Flux2Request(),
    Flux2Restart(
        latents=Path("previous_final_latents.safetensors"),
    ),
)
```

Restart uses the runner's prepared resources. A complete conditioning station supplies
its saved prompt context; an explicit conflicting prompt is rejected. Decode uses only
the VAE and final latents. The CLI additionally resolves run records, fingerprints, and
recorded resource identities before calling the same runner recipes. `Flux2Error` is the
typed operational error boundary; missing files and cancellation retain their normal
exception types.

`generate_image(resources, prompt, ...)` remains the lower-level single-image function
for explicit tensors and sampling controls. It accepts `capture_stations=True` and
`conditioning=`; `decode_latents()` takes the VAE source files and a final latent/grid.
Transformer weights bind lazily and evaluate during denoising after its progress event
starts. Each text, reference VAE, transformer, and decode VAE lifetime ends before the
next model is loaded.

In `Flux2Request`, use `source_images`, `steps`, `guidance`, and `cfg_mode` for editing
and recipe controls. The following details describe the lower-level function's arguments.

For editing, pass `images=(Path("reference.jpg"),)` and, for example, `height=1024`.
A single reference uses source-aspect sizing and fits the output grid by default. Explicit
`match_source_aspect` and `reference_fit` use the shared image-model semantics described below.
Multiple images preserve order and independent reference grids; only target latents are updated.

Select `variant="base"` in the resource settings for default 50-step CFG at scale 4. A
`SamplingRecipe` passed as `recipe=` can override steps, guidance, and `cfg_mode` (`sequential`
or `batched`); its variant must match the resources. Distilled defaults to four steps without
external CFG. Sequential CFG is the default and materializes the positive branch before the
negative branch. The default negative prompt is empty text. `negative_prompt=` changes it;
`negative_embeddings=` accepts an explicit `[1,512,12288]` tensor instead. Negative conditioning
requires active base CFG and cannot combine a nonempty negative prompt with explicit embeddings.

`noise=` accepts shared packed `[1,H/16 * W/16,128]` input for cross-runtime comparisons.
Otherwise `seed` and `noise_backend` use the central KinoMLX normal-noise stream. Matching seed
numbers across different RNG implementations do not imply matching tensors. Transformer FP16
execution preserves the separately validated BF16 text/VAE policies and casts at their boundaries.

## Native text and VAE components

Use the component APIs when your application needs embeddings or VAE conversion without
running a complete generation. Loading a component does not load the transformer or other
models. VAE `mode="encode"` or `mode="decode"` avoids allocating the unused half; use
`mode="both"` only when both directions are needed. Apply the memory policy before loading
components and release each model when you have finished using it.

```python
import mlx.core as mx

from kinomlx.models.flux2.loading import component_weight_files, load_text_encoder, load_tokenizer
from kinomlx.models.flux2.resources import inspect_resources
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.flux2.text import prepare_text
from kinomlx import apply_memory_policy
from kinomlx.settings import Settings

resources = inspect_resources(Flux2Settings(variant="kv"), infrastructure=Settings.from_env())
apply_memory_policy(resources.memory_policy)
tokenizer = load_tokenizer(resources.component("tokenizer"))
batch = prepare_text(tokenizer, ("A glass sphere on a wooden table.", ""))
encoder = load_text_encoder(component_weight_files(resources.component("text_encoder")))
conditioning = encoder(batch.input_ids, batch.attention_mask)
mx.eval(conditioning)
del encoder
```

For reproducible embeddings, preserve the template and padding: text uses the user-message
template with thinking disabled, right padding/truncation to 512 tokens, and raw hidden-state
taps 9, 18 and 27. `conditioning` is `[batch, 512, 12288]`. Keep all 512 states, including
padding states. `batch.text_ids` provides `[batch, 512, 4]` THWL coordinates. An empty or
negative prompt uses the same conditioner as the positive prompt.

`load_vae(files, mode="both")` returns a BF16 `Flux2VAE`. `encode()` accepts BCHW RGB pixels in
`[-1,1]` with dimensions divisible by 16 and returns the raw 32-channel posterior mean.
`encode_normalized()` additionally packs 2x2 spatial positions into 128 channels and applies
stored inference normalization. `decode_normalized()` reverses those operations and returns
unclipped BCHW pixels; clip and map to `[0,1]` at the image-output boundary. Posterior sampling
and tiling are not part of this component path. `dtype=mx.float32` is a diagnostic execution
option that expands the bundled BF16 weights; it does not recover original FP32 weights.

`image_processing.output_size()` resolves single-reference aspect defaults and explicit sizing.
`reference_geometry()` records integer crops, fitted sizes and floating H/W token offsets;
`load_reference_pixels()` applies EXIF orientation, native bicubic resampling and normalization.
Single references fit the resolved output grid by default. Multiple references retain independent
native grids with the 1024-square area cap unless fit/crop is explicitly selected. Native grids
require source sides of at least 64 pixels and aspect ratio at most 8. `spatial_ids()` assigns
target time zero or ordered reference time `10 * (index + 1)` without losing half-token offsets.

If an external implementation uses Lanczos while KinoMLX uses native bicubic resizing,
identical image dimensions can still produce different conditioning pixels. Supply the same
preprocessed tensors when isolating model differences in a comparison.

## Reference KV lifetime

The KV variant can reduce repeated reference-image work during editing. Use
`--variant kv` in the CLI or `Flux2Settings(variant="kv")` in Python. The first
step computes the reference representation; later steps reuse it, so the first
step and the remaining steps can have different timings. With no references it
uses full attention, and a one-step recipe has no later step to benefit from reuse.

The speed benefit costs memory. For this 32-block 9B model, BF16 or FP16 reference
keys and values require 512 KiB per reference token: 2 GiB for 4,096 tokens or
4 GiB for 8,192 tokens, in addition to model weights and active computation.
Reference dimensions and count therefore matter even when output size is unchanged.
Klein does not silently shrink references to fit an additional KV memory cap.

Reference KV lasts for one generation. Another call extracts it again, including
when you reuse a Python runner; it is cleared on success, failure, or cancellation.
The cache does not act as a persistent reference-image library.

### How reuse preserves the reference calculation

Reference tokens use fixed timestep-zero modulation and attend only to the
reference sequence. Text and target tokens attend to all tokens. Each block saves
the reference keys after rotary positioning and the reference values, then later
steps inject them into attention while updating text and target tokens. This is
why those reference values can be reused without recomputing them at each step.
It is a property of the KV model's computation, not a switch that makes an ordinary
distilled checkpoint equivalent to the KV variant.

`--recompute-reference-kv` (TOML `recompute_reference_kv`, KV variant only)
recomputes that same fixed-reference calculation every step, at the cost of one
extra reference forward per step. It is required when a per-step adapter schedule
or a sigma window edits with references, because the reused cache would otherwise
hold values from the first step's weights, and it is useful for checking the reuse
optimization; it is never a faster recipe. The Python API exposes the same switch as
`SamplingRecipe(Flux2Variant.KV, recompute_reference_kv=True)`. With it off, model
weights, including adapters, must remain unchanged during the call: changing them
would invalidate the saved reference representation. The standard adapter loader
applies fused adapters before reference extraction.

### Memory policy for API calls

The runner and public recipe functions apply the resource inventory's shared
memory policy before model work. Set `mlx_cache_limit_gb` and
`mlx_wired_limit_gb` through infrastructure `Settings`. Reusing resources
preserves those choices; each invocation reapplies them. For standalone
component loading, call `kinomlx.apply_memory_policy(resources.memory_policy)`
before allocating inputs or loading weights. See
[public API memory policy](PERFORMANCE.md#public-api-memory-policy).
