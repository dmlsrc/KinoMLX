# KinoMLX

Native multimodal inference on Apple Silicon through
[MLX](https://github.com/ml-explore/mlx). KinoMLX provides four model families
through one CLI and Python APIs:

- **LTX2**: LTX-2.3 and LTX-2.5 text/image-to-video generation, optional stereo
  audio, model-specific HDR workflows, adapters, and staged restart.
- **GMNet**: SDR-to-HDR still expansion with realworld and synthetic variants,
  EXR and PQ HEIC output, and optional gain maps.
- **Krea 2**: Raw and Turbo text-to-image generation, reference editing with a
  compatible adapter, lossless PNG/TIFF output, and saved-station restart.
- **FLUX.2 Klein 9B**: base, distilled, and KV text/image generation, ordered
  reference editing, adapters, lossless PNG/TIFF output, and saved-station restart.

KinoMLX is an independent implementation built from published architecture descriptions and
implementations, checkpoint metadata, and independent validation. You do not need to install the
original model implementations to run KinoMLX.

## Status

Pre-alpha. All four families have end-to-end CLI and Python workflows.
Interfaces and defaults can change between commits. Capabilities, model access,
and limits are documented in each model guide.

| Model | Input and output | Python generation entry point | Guide |
| --- | --- | --- | --- |
| `ltx2` | Text/optional image -> video and optional audio | `generate_distilled()` / `LTX2Runner` | [LTX2](docs/LTX2.md) |
| `gmnet` | SDR still -> HDR still artifacts | `expand_gmnet()` / `GMNetRunner` | [GMNet](docs/GMNET.md) |
| `krea2` | Text/optional references -> PNG/TIFF stills | `generate_krea2()` / `Krea2Runner` | [Krea 2](docs/KREA2.md) |
| `flux2` | Text/ordered references -> PNG/TIFF stills | `generate_flux2()` / `Flux2Runner` | [Klein](docs/FLUX2.md) |

Use the CLI for generation commands or a model runner to generate from Python.
The examples linked below show how to select settings, run a model, and save its
output. Each model guide describes the inputs it supports and the available controls.

## Requirements

- Apple Silicon (M1 family or later)
- macOS
- Python 3.14
- Unified memory appropriate to the model, precision, and output geometry

LTX supports a streamed 32 GB profile; 64 GB is recommended for its default
resident-transformer path and larger geometries. Krea and Klein use sequential
component lifetimes, but their working sets still depend on image size and
reference count. GMNet has a much smaller working set that scales primarily
with source dimensions. Consult the model guides for validated configurations.

## Install

```bash
git clone https://github.com/dmlsrc/KinoMLX.git
cd KinoMLX
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
kinomlx --help
```

KinoMLX does not download model weights at runtime or bundle them in its
repository, source distribution, or wheel. LTX2, Krea 2, and Klein use Hugging
Face repositories; GMNet publishes its checkpoints on GitHub.

For cache-managed downloads, optionally install the `hf` CLI as an isolated uv tool:

```bash
uv tool install hf
hf --help
```

The official [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
also documents `uvx hf`. If `hf` is not on `PATH`, run `uv tool update-shell`
and open a new shell. Inspect model cards, licenses, and access terms before
downloading. CLI authentication alone does not grant access to a gated repository;
accept or request access on the model page first, then use `hf auth login`.
Alternatively, download the files into any local folder through the model pages.

## Quick start

Once the corresponding local model files are available:

```bash
# LTX-2.3 distilled video with optional generated audio
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 --generate-audio

# GMNet SDR-to-HDR still expansion using converted local weights
kinomlx --model gmnet \
  --image photo.jpg --output-dir hdr/ --save-gain-map

# Krea 2 Turbo text-to-image generation
kinomlx --model krea2 \
  --prompt "A ceramic cup on a walnut table, soft window light" \
  --width 1024 --height 1024 --seed 42

# FLUX.2 Klein distilled text-to-image generation
kinomlx --model flux2 \
  --prompt "A ceramic cup on a walnut table, soft window light" \
  --width 1024 --height 1024 --seed 42
```

Use `kinomlx --model MODEL --help` for any of the four model names above.
`kinomlx config init --model MODEL --output settings.toml` writes an annotated
starter configuration for that model. `kinomlx weights --help` describes conversion.

`kinomlx config inspect --model MODEL` prints the registered configuration inventory as JSON.
Add `--field generate.seed` to inspect one field. The inventory shows typed ownership, CLI/ENV
bindings, normalization rules, public API parameter roles, and each field's restart decision with
the reason for it. A model without a saved restart operation reports `no_restart` and states why.
This inspects declarations without reading invocation files or weights.

## LTX2 model access

Inspect the [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) or
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) model repository. The
[LTX download guide](docs/LTX2.md#ltx-model-downloads) lists required text and VAE
components, optional FP8 sources, and exact download commands for each version.

LTX-2.5 resolves its component pack from the same cache:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --auto-duration --generated-keyframes 1
```

Automatic duration and generated keyframe slots are LTX-2.5 features. LTX-2.3
has a validated SDR-video-to-HDR profile; LTX-2.5 supports native HDR-still-to-video.
Both support adapter composition and staged restart. See the
[LTX2 guide](docs/LTX2.md) and [Python example](examples/ltx2_distilled.py) for
component overrides, streaming, HDR, and explicit pipeline composition.

## GMNet model access and current usage

Download a published realworld or synthetic checkpoint from the
[GMNet weight guide](docs/GMNET.md#model-weights). Convert it once with the
restricted, torch-free model converter:

```bash
kinomlx weights convert gmnet G_realworld.pth
kinomlx weights convert gmnet G_synthetic.pth
```

The guide documents source placement and converted-weight discovery. Select
`--variant synthetic` for that checkpoint; `realworld` is the default. GMNet
accepts display-referred SDR stills and publishes a half-float EXR master,
10-bit PQ HEIC, and optional gain-map sidecar as one transaction. It does not
perform temporal video expansion or accept scene-linear EXR input.

See the [GMNet guide](docs/GMNET.md) and
[Python example](examples/gmnet_expand.py) for checkpoint selection, output
controls, and `GMNetRunner` integration.

## Krea 2 model access and current usage

Inspect the [Turbo](https://huggingface.co/krea/Krea-2-Turbo) and
[Raw](https://huggingface.co/krea/Krea-2-Raw) model cards. Download either or both:

```bash
hf download krea/Krea-2-Turbo --exclude turbo.safetensors
hf download krea/Krea-2-Raw --exclude raw.safetensors
```

The excludes omit duplicate root transformer monoliths. Run `kinomlx --model krea2`
without a prompt to inspect Turbo resources, or add `--variant raw` for Raw. An
ordinary directory works with `--source-path /path/to/Krea-2-Turbo`.

Turbo defaults to eight guidance-free steps; Raw defaults to 52 steps with CFG.
Reference editing uses a compatible edit adapter. Transformer FP16 is selectable
with `--transformer-dtype float16`; BF16 remains Krea's default. An optional 2x VAE
decoder changes output size and appearance and is described in the model guide.

The optional `--turbo-4step` profile selects the community four-step Turbo adapter;
see the [recipe and download instructions](docs/KREA2.md#four-step-turbo-adapter).

See the [Krea guide](docs/KREA2.md) and
[Python example](examples/krea2_generate.py) for component overrides, edit
conditioning, adapters, decoder choices, and `Krea2Runner` integration.

## FLUX.2 Klein model access and current usage

Inspect the [distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
[base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), and
[KV](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) model cards.
Download only the variants you intend to use:

```bash
hf download black-forest-labs/FLUX.2-klein-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-base-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-9b-kv --exclude "flux-2-klein-*.safetensors"
```

These commands omit duplicate root transformer monoliths and retain the split
runtime components. Run `kinomlx --model flux2 --dry-run` to inspect the
default distilled resources; select `--variant base` or `--variant kv` as needed.
`--source-path` also accepts an ordinary local component directory.

Distilled and KV use four steps; base defaults to 50 steps with CFG. A suitable
Turbo LoRA can enable a faster base recipe while allowing style-adapter stacking.
KV accelerates reference reuse within one generation, including ordered
multireference editing; its cache is not retained across generations.

See the [Klein guide](docs/FLUX2.md) and
[Python example](examples/flux2_generate.py) for reference fitting, Turbo LoRAs,
FP16/BF16 execution, and `Flux2Runner` integration.

## Weight loading

LTX2, Krea 2, and Klein load original local checkpoints, derive tokenizer tables in
memory, and prepare dtype and layout transforms lazily. Each phase releases its
components when they are no longer needed. See the
[weight-loading guide](docs/WEIGHTS.md) for component lifetimes, FP8 sources,
validation, and storage.

## Shared generation controls

Registered model options accept environment defaults named
`KINO_<MODEL>_<FIELD>`, using the model ID and registered configuration field
name in uppercase. The field can differ from its CLI spelling: `--output` uses
`KINO_KREA2_OUTPUT_PATH`. Similarly, `--model ltx2 --vae-decode-dtype float16` can
use `KINO_LTX2_VAE_DECODE_DTYPE=float16`, and Klein's `--seed` can use
`KINO_FLUX2_SEED=42`. Shared infrastructure options retain names such as
`KINO_MLX_CACHE_LIMIT_GB`. Generated configuration templates list each binding.
TOML configuration, explicit CLI options, and `--set` override environment defaults.
The output directory also accepts the shared `KINO_OUTPUT_DIR` fallback described below.

Older `KINO_KREA_*` and unscoped LTX model variables have been replaced by
`KINO_KREA2_<FIELD>` and `KINO_LTX2_<FIELD>`. Update shell exports to those names;
`KINO_OUTPUT_DIR` remains the shared output-directory fallback.

### Prompt files

LTX2, Krea 2, and Klein accept `--prompt-file` instead of `--prompt`. Each UTF-8
file supplies one complete prompt, preserving internal line breaks. Empty files,
invalid UTF-8, NUL characters, and missing files produce errors. Configs and run
records store the resolved text, so replay does not need the original file.

Use `--prompt-unconditional` with LTX2, Krea 2, or Klein to deliberately generate
with an empty user prompt. Normal text encoding, model templates, adapters, and
any supplied image conditioning still apply. This does not disable guidance or
replace text embeddings with zeros. Nonempty `--prompt` or `--prompt-file` input
conflicts with this option, including text inherited from a config file; use
`--no-prompt-unconditional` when overriding such a config with a new prompt.

```bash
kinomlx --model flux2 --prompt-unconditional --seed 42
```

The same control is `[generate].prompt_unconditional = true` in TOML and
`prompt_unconditional=True` on `DistilledRequest`, `Krea2Request`, or
`Flux2Request`. Krea's `--system-prompt` remains independent. Saved conditioning
must contain an empty user prompt when this option is selected. Omitting both
prompt input and this option retains each model's existing preparation or
missing-input behavior.

### Memory and performance

The shared allocator-cache default is **1 GiB**, including LTX streaming.
`--mlx-cache-limit-gb 0` disables caching of unused allocations; it does not cap
live model memory or mean unlimited caching. `--mlx-wired-limit-gb` independently
controls how much MLX memory can remain resident and defaults to the device's
recommended working set.

See [Performance](docs/PERFORMANCE.md) for configuration examples, the different
meanings of zero, memory troubleshooting, and timing comparisons. LTX-specific
[streaming](docs/LTX2.md#block-streaming-on-32-gb-systems),
[tensor retention](docs/LTX2.md#retaining-selected-block-tensors), and
[quantization](docs/LTX2.md#transformer-quantization) are covered in the model guide.

#### Public API memory policy

Runners and public recipes apply the memory policy selected by infrastructure
`Settings` before model work. Low-level component callers apply it explicitly.
See [API memory policy](docs/PERFORMANCE.md#public-api-memory-policy) for examples,
`None` semantics, and process-wide lifetime rules.

### Adapters and saved stations

LTX2, Krea 2, and Klein also accept `--adapter style.safetensors:0.8`, with `--lora` as an alias.
LTX2 additionally accepts `--adapter style.safetensors:0.25:0.5` for stage 1 and
stage 2; zero skips that adapter in the corresponding stage. See the
[adapter guide](docs/ADAPTERS.md) for supported formats and separate strength flags.
Adapter-modified weights are checked once after their first denoise step.
`--no-check-adapter-weights` skips that scan; per-step latent checks remain enabled.

Krea and Klein share image-output, source-aspect, reference-fit, and saved-station
flag spellings. Repeat `--prompt` and use `--images-per-prompt` for variations.
`--save-all-sidecars` saves the model's applicable execution records and reusable
stations. `--restart previous_run.json --restart-from denoise` reuses saved
conditioning; `--restart-from decode` reuses final latents. Edit replay needs a
complete conditioning bundle. LTX has its own staged restart contract, and GMNet
has no denoising stages. See each model guide for supported restart boundaries.

Saved restart uses run schema version 2. Its required configuration record preserves canonical
inputs and supplied fields; effective execution values are recorded separately. Older run records
and records without this configuration format are rejected, with no automatic conversion.

## Seed lists and model lifetimes

LTX, Krea, and Klein accept an ordered seed list. Space-separated and comma-separated
spellings are equivalent; `--seeds` is an alias for `--seed`:

```bash
kinomlx --model ltx2 --prompt-file scene.txt --seed 42 123 999
kinomlx --model krea2 --prompt-file scene.txt --seed 42,123,999 --seed-mode station
kinomlx --model flux2 --prompt-file scene.txt --seed 42 123 999 --seed-mode resident
```

Order and duplicates are preserved. Each prompt uses the same selected seeds, with
results ordered by prompt and then seed. `--seed 42 --images-per-prompt 4` is shorthand
for consecutive seeds 42, 43, 44, and 45 for each image prompt. An explicit seed list
cannot be combined with `--images-per-prompt` greater than one. Seeds must be unsigned
64-bit integers; invalid lists fail before generation.

Image models support three execution modes:

| `--seed-mode` | Execution and model lifetime |
| --- | --- |
| `sequential` (default) | Finish one seed through all stations with normal model loading and unloading, then start the next. |
| `station` | Load the transformer, denoise each seed one at a time, and unload it. Then load the VAE, decode each seed one at a time, and unload it. |
| `resident` | Finish each seed through all stations while keeping the transformer and VAE loaded between seeds. Release both at the end of the request. |

All three modes process one seed at a time in each model call. `station` retains the
intermediate latents until decoding; it does not combine seeds into a tensor batch.
`resident` needs room for the transformer and VAE together, along with working tensors
and retained results. The selected mode is never silently changed to another mode.
Klein's `--cfg-mode` independently controls how positive and negative guidance branches
are evaluated.

Every model reuses seed-independent conditioning between seeds. The text encoder loads
once per request, encodes every prompt row, and is released before the denoiser loads;
reference encoders are released the same way, and their resulting tensors remain available
for the request. Initial noise and generation-specific state remain separate for every seed.
Krea image editing also reuses source-image encodings and visual grounding. Model retention
ends on completion, cancellation, or failure, including in `resident` mode.

LTX processes complete videos sequentially and retains text and source-media conditioning
at both stage geometries. It keeps its normal transformer lifetime within each video
and unloads before decoding. Image `--seed-mode` options do not apply to LTX. LTX saved-run
restart currently accepts one seed; use ordinary generation for seed lists. Each LTX seed
writes its own video, sidecar bundle, and run record; `--json` still emits exactly one
record for the command, with a `runs` array holding one entry per seed.

Whenever a run produces more than one image, its filenames carry an output index and the
seed that produced it, so repeated seeds and consecutive `--images-per-prompt` seeds both
get distinct, self-describing paths. Each completed image and requested sidecar is published
immediately at its final path. Ctrl-C or a later failure stops remaining work and preserves
completed files. Partial console logs and the latest valid run record also remain; graceful
failure or cancellation records `failed` or `aborted`. LTX also preserves completed stage
sidecars and saves each completed video before starting the next. Krea and Klein write
separate tensor sidecars for each output image. Run metadata records the ordered selection,
image execution mode, and completed outputs.

Canceling a run reports `canceled` with the outputs it had already published, in both
the human and `--json` surfaces, and exits 130. Every model reports it the same way.

A run that produced several images records one artifact path per image, so a saved-station
restart selects which one to resume with `--restart-index` (one-based, matching the output
number in the filename). Krea and Klein both accept it, and both narrow the resumed request
to that image's prompt and seed:

```bash
kinomlx --model krea2 --prompt "A lighthouse at sunset" --seed 42 99 --save-all-sidecars
kinomlx --model krea2 --restart previous_run.json --restart-from decode --restart-index 2
```

A single-image source run needs no index. A multi-image run without one is refused with the
range to choose from, and `--restart-latents` still substitutes a different tensor file for
the selected image. A run you stopped with Ctrl-C is restartable for each image it had
already published, exactly like a completed run; selecting an image it never reached, or
restarting a run that failed, is refused.

Because completed files are published as they finish, a multi-image run that replaces
existing targets is not all-or-nothing: if replacement is enabled and the run stops partway,
the finished outputs are already replaced while the rest still hold their previous contents.
Write to a fresh output bundle when you need the old set kept intact until a run completes.

Use `seeds` and `seed_mode` in TOML:

```toml
[generate]
prompt = "A lighthouse at sunset"
seeds = [42, 123, 999]
seed_mode = "station" # Image models only.
```

Use `seed` for one value or `seeds` for an ordered list in config. `seed` and `seeds` are
mutually exclusive within a configuration layer; a higher-precedence selection replaces the
earlier one. Python callers can pass `seeds=(42, 123, 999)` to `Krea2Request`, `Flux2Request`,
or `DistilledRequest`, and `seed_mode="station"` to image requests. Existing scalar `seed=`
callers remain valid.

`LTX2Runner.generate(request)` yields one `GenerationOutput` per seed. Consume and close
each output before advancing. Use `contextlib.closing` around the iterator when stopping
early, so shared conditioning is released promptly:

```python
from contextlib import closing

from kinomlx.models.ltx2 import DistilledRequest, LTX2Runner

runner = LTX2Runner()
request = DistilledRequest(prompt="A lighthouse at sunset", seeds=(42, 123, 999))
with closing(runner.generate(request)) as results:
    for result in results:
        with result:
            # Consume result.frames or pass result to your output writer here.
            for frame in result.frames:
                process_frame(frame)
```

## Output names and sidecars

Without `--output`, filenames include a local timestamp:

| Model | Default generated filenames |
| --- | --- |
| LTX2 | `ltx2-distilled-seed<seed>_<timestamp>.mp4` |
| GMNet | `<input-stem>_<timestamp>.exr` and `<input-stem>_<timestamp>.heic` |
| Krea 2 | `krea2-<variant>-seed<seed>_<timestamp>.png` |
| Klein | `flux2-<variant>-seed<seed>_<timestamp>.png` |

The directory defaults to `KINO_<MODEL>_OUTPUT_DIR`, then the shared
`KINO_OUTPUT_DIR` fallback, then `outputs/`
(for example, `KINO_LTX2_OUTPUT_DIR`). Use `--output-dir`
and `--output-prefix` to choose a directory and prefix. Generated names use
`YYYYMMDD_HHMMSS` and acquire `_2`, `_3`, and so on when occupied.

An exact `--output` has no automatic timestamp. Krea and Klein batches add
`-001`, `-002`, and so on before the image extension. Existing explicit targets
are refused; Krea, Klein, and GMNet offer `--force` for deliberate replacement.
`--force` requires an exact `--output` path across all three models.
Generated names always select a fresh bundle. Sidecars share the resolved stem;
individual save flags can select or exclude categories from `--save-all-sidecars`.

## Documentation

- [LTX2 guide](docs/LTX2.md) and [API example](examples/ltx2_distilled.py):
  downloads, video/audio, HDR, adapters, staged restart, and performance.
- [GMNet guide](docs/GMNET.md) and [API example](examples/gmnet_expand.py):
  conversion, variants, HDR still artifacts, gain maps, and transactional output.
- [Krea guide](docs/KREA2.md) and [API example](examples/krea2_generate.py):
  Raw/Turbo generation, reference editing, component overrides, and saved stations.
- [Klein guide](docs/FLUX2.md) and [API example](examples/flux2_generate.py):
  base/distilled/KV generation, ordered references, adapters, and saved stations.
- [Weight loading](docs/WEIGHTS.md): source selection, lazy preparation,
  component lifetimes, and validation.
- [Performance](docs/PERFORMANCE.md): memory controls, precision tradeoffs, troubleshooting,
  and comparable timing measurements.
- [Adapter formats](docs/ADAPTERS.md): shared parsing, model support, and strength syntax.
- [Denoise schedules](docs/SCHEDULES.md): reporting and setting sigmas, and varying
  adapter strength by step or by noise level.
- [Project structure](docs/STRUCTURE.md): package map, responsibilities, and public APIs.
- [Project conventions](docs/CONVENTIONS.md): contributor rules for architecture, code,
  configuration, documentation, testing, and performance measurements.
- [Changelog](CHANGELOG.md): user-facing changes in each release.

## Performance and limits

Performance and memory depend on model, precision, geometry, references, and
sampling recipe. Model-guide measurements describe their tested configuration;
video timings do not predict still-image timings. LTX supports transformer
streaming, Krea provides compact denoising and selectable precision, Klein uses
FP16/BF16 execution and per-generation reference KV, and GMNet scales with input size.

The public LTX recipe is the distilled two-stage 8 + 3 pipeline; general retake
and alternative public recipes remain out of scope. Frame counts are `8*k+1`.
GMNet expands SDR stills only. Krea editing requires a compatible adapter.
Klein run-record restart resumes one image of a multi-image run, selected with
`--restart-index`, and offers no arbitrary mid-step resume or VAE tiling. The model
guides own the detailed capability limits.

## Tests

Install the test tooling with the `dev` extra, then run the normal suite:

```bash
uv pip install -e ".[dev]"
python -m pytest tests
```

The suite runs in parallel by default (`-n auto`, one worker per physical core),
so every test in it must tolerate arbitrary interleaving. Pass `-n 0` for a single
process when debugging, or under `coverage run`, which only observes the controller
process.

Tests marked `heavy` load real checkpoints or run long GPU work. They are excluded
by default. To select them deliberately:

```bash
python -m pytest tests --run-heavy -m heavy
```

A heavy run always uses one process, whatever `-n` says: these tests load
multi-gigabyte checkpoints and wire MLX memory, and parallel workers would compete
for the same unified memory. A heavy test that reaches a parallel worker some other
way refuses to start rather than loading a checkpoint there. `slow` labels the heavy
tests that take minutes, so the quick checkpoint checks alone are:

```bash
python -m pytest tests --run-heavy -m 'heavy and not slow'
```

Heavy is about cost, not about real files. A test that reads checkpoint headers,
checks a tokenizer against its reference, or loads one small component runs in the
default suite; it finds its input the way the product does (the model's environment
settings first, then the local Hugging Face cache) and skips with the reason when the
cache does not hold it.

Heavy tests require locally available checkpoints and can take substantial time and
temporary disk space. Individual full-pipeline gates may require additional opt-in
environment variables. Test file writes use per-test temporary directories. Pytest
may retain recent temporary directories after a run according to its retention policy.

## Attribution and license

KinoMLX builds on Apple's [MLX](https://github.com/ml-explore/mlx) framework and
published architecture implementations and checkpoints from Lightricks, the
GMNet authors, Krea, Black Forest Labs, Hugging Face, and Google. Model weights
are not redistributed and remain governed by their upstream terms. Inspect the
model cards and the [LTX2](docs/LTX2.md), [GMNet](docs/GMNET.md),
[Krea](docs/KREA2.md), and [Klein](docs/FLUX2.md) model guides.

KinoMLX's own code is [MIT-licensed](LICENSE). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for included third-party
license texts and notices.
