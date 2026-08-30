# KinoMLX - Project Structure

A map of every directory and module, what it owns, and why it lives
where it does. Read this when you're not sure where to put new code.

## Overview

KinoMLX is a focused multimodal inference engine on Apple Silicon via MLX. It
currently ships distilled LTX-2.3/LTX-2.5 video generation and GMNet SDR-to-HDR
still expansion. The project is designed to be **modular by purpose** and
**extensible to new models and modalities**:

- Each model is a self-contained subpackage under
  `kinomlx/models/<name>/`. No cross-model imports.
- Model-agnostic infrastructure (samplers, LoRA fusion, I/O, UX,
  profiling, settings) sits at the top level of `kinomlx/`.
- The CLI dispatches inference through `--model`; model-neutral utilities such
  as checkpoint conversion have their own command tree.
- A pure-function bias drives the pipeline composition: classes only
  for genuine state (loaded weights, KV caches, file handles); pipeline
  orchestration is composition of pure functions.
- Soft ~500-line ceiling per file.

## Top-Level Tree

```text
KinoMLX/
|-- LICENSE                          # MIT, copyright dmlsrc 2026
|-- README.md                        # landing page, quick start, documentation map
|-- THIRD_PARTY_LICENSES.md          # MIT/Apache borrows attribution
|-- pyproject.toml                   # deps, entry point, ruff/pytest config
|-- .gitignore                       # Python template + project-specific ignores
|-- docs/                            # human-readable project documentation
|   |-- GMNET.md                     # gmnet weights, usage, output, and API guide
|   |-- LTX2.md                      # ltx2 setup, generation, restart, and API guide
|   `-- STRUCTURE.md                 # this file
|-- examples/                        # public recipe/runner compositions
|   |-- gmnet_expand.py              # small still-expansion + transaction example
|   `-- ltx2_distilled.py            # explicit distilled ownership reference
|-- tests/                           # pytest tests (mirrors package structure)
|-- scripts/                         # small project utilities and demonstrations
|-- weights-src/                     # collected upstream source checkpoints (untracked; see its README)
|-- debug/                           # local runtime sidecar output (gitignored)
`-- kinomlx/                         # the importable Python package
```

## Package Tree (`kinomlx/`)

```text
kinomlx/
|-- __init__.py                      # version, infrastructure + lazy model API exports
|-- artifacts.py                     # model-neutral tensor-artifact envelope and sink
|-- errors.py                        # typed operational error base
|-- reporting.py                     # host-neutral Reporter protocol + test sinks
|-- output.py                        # typed generation sink + VideoToolbox adapter
|-- types.py                         # shared dataclasses (VideoLatentShape, LatentState, ...)
|-- settings.py                      # infrastructure settings + env/CLI field machinery
|
|-- config/                          # generic TOML config primitives
|   |-- load.py                      # filename-aware TOML loading/composition
|   |-- merge.py                     # recursive table merge
|   |-- overrides.py                 # TOML-typed --set
|   |-- validate.py                  # dataclass schema validation + suggestions
|   |-- dump.py                      # deterministic round-trippable TOML
|   `-- registry.py                  # validated global model config contributions
|
|-- cli/                             # split-up model-neutral and model-owned CLI
|   |-- __init__.py                  # namespace only; entry point targets main.py
|   |-- _registry.py                 # file-loaded CLI + lazy runner/recipe registry
|   |-- main.py                      # entry, utility/model dispatch, typed error boundary
|   |-- args.py                      # LTX parser + compact model-neutral root help
|   |-- common.py                    # shared model-selection/error/elapsed CLI vocabulary
|   |-- config.py                    # one precedence path to typed runtime config
|   |-- config_file.py               # --config <kino.toml> support (TOML runner config)
|   |-- config_records.py            # runtime-light shared output config records
|   |-- config_schema.py             # one shared infrastructure-settings contribution
|   |-- config_init.py               # model-specific starter-config utility
|   |-- config_output.py             # shared full/sparse print and no-clobber save
|   |-- config_templates.py          # readable TOML renderer over the global registry
|   |-- restart.py                   # prior-run manifest, sidecar selection, identity receipt
|   `-- output.py                    # machine JSON + VideoToolbox output adapter
|
|-- weights/                         # generic safe checkpoint-conversion substrate
|   |-- __init__.py                  # lazy public converter exports
|   |-- cli.py                       # `weights convert` + model-specific dispatch
|   |-- host.py                      # shared settings/progress/allocator converter host
|   |-- output.py                    # no-clobber verified publication transaction
|   |-- pickle_scan.py               # exact symbolic pickle-global scanner
|   |-- torch_checkpoint.py          # bounded restricted tensor-only reader
|   `-- convert.py                   # value/layout-preserving generic conversion
|
|-- samplers/                        # MODEL-AGNOSTIC sampling math
|   |-- __init__.py
|   |-- noise.py                     # centralized MLX / Torch-MPS normal streams
|   |-- steps.py                     # Euler step
|   `-- noisers.py                   # Gaussian / channel-wise normalized noisers
|
|-- lora/                            # generic LoRA infrastructure
|   |-- __init__.py
|   |-- fusion.py                    # W += alpha * (B @ A) math
|   `-- loading.py                   # parse LoRA safetensors -> (key, A, B, alpha) tuples
|
|-- debug/                           # opt-in reproducibility artifacts
|   |-- __init__.py                  # exported orchestration-owned artifact writers
|   |-- metadata.py                  # effective TOML, execution logs, run records
|   `-- sidecars.py                  # generic tensor-artifact safetensors sink
|
|-- profiling/                       # opt-in Instruments instrumentation
|   |-- __init__.py
|   |-- signpost.py                  # Apple os_signpost wrapper
|   `-- _signpost.c                  # native signpost shim source
|
|-- ui/                              # centralized terminal output
|   |-- __init__.py
|   |-- console.py                   # shared Rich stderr console
|   |-- logging.py                   # human logging + isolated machine stdout
|   `-- bars.py                      # progress bars + RichReporter adapter
|
|-- io/                              # input / weight I/O
|   |-- __init__.py
|   |-- atomic.py                    # atomic file publication primitives
|   |-- fingerprints.py              # streamed file SHA-256 receipt identity
|   |-- image.py                     # native ImageIO/CoreImage -> MLX tensor (image conditioning)
|   |-- reservation.py               # hidden exclusive target-reservation peers
|   `-- safetensors.py               # mx.load weight I/O + native header metadata reader
|
|-- media/                           # model-neutral frame/signal boundary contracts
|   |-- __init__.py                  # public media vocabulary
|   |-- frames.py                    # closeable single-consumer frame stream
|   `-- signals.py                   # signal, color-plan, and delivery specifications
|
|-- videotoolbox/                    # FIRST-CLASS Apple VideoToolbox subsystem
|   |-- __init__.py
|   |-- errors.py                    # typed native operation errors + PyObjC classification
|   |-- encode.py                    # AVAssetWriter HEVC/H.264 encode (no ffmpeg)
|   |-- writer.py                    # AVWriter - AVAssetWriter file output
|   |-- pixel_buffers.py             # CVPixelBuffer wrapping (zero-copy mx <-> IOSurface)
|   |-- images.py                    # native still-image I/O (ImageIO/CoreImage)
|   |-- frame_dump.py                # opt-in post-VAE diagnostic PNG sequence
|   |-- exr.py                       # native half-float HDR master sequence
|   |-- hlg.py                       # scene-linear RGB -> BT.2020/HLG P010
|   |-- heic.py                      # scene-linear RGB -> BT.2100 PQ HEIC previews
|   |-- audio.py                     # AudioTrack PCM -> CMSampleBuffer (muxing)
|   |-- vsr.py                       # VideoToolbox Super Resolution
|   |-- temporal.py                  # VTFrameRateConversion (VtfrcSession)
|   |-- cut_detect.py                # scene-cut policy for history-sensitive native stages
|   `-- comparison.py                # side-by-side A/B composite
|
|-- kernels/                         # custom Metal kernels
|   |-- __init__.py
|   `-- fused_ops.py                 # FP32-opmath activations + precision-sensitive ops
|
|-- audio/                           # audio utilities (model-agnostic)
|   |-- __init__.py
|   `-- onset.py                     # start-of-clip click suppression
|
`-- models/                          # per-model self-contained packages
    |-- __init__.py
    |-- gmnet/                       # GMNet SDR-to-HDR still expansion (--model gmnet)
    |   |-- __init__.py              # lazy exports; import stays MLX-free
    |   |-- types.py                 # immutable public still-expansion request
    |   |-- resources.py             # prepared paths and policies, never live weights
    |   |-- components.py            # generator port, lease, and native provider
    |   |-- pipeline.py              # public stateless expand_gmnet recipe
    |   |-- runner.py                # generic recipe host + convenience expand method
    |   |-- output.py                # typed all-or-nothing EXR/HEIC/gain-map terminal
    |   |-- net.py                   # NHWC GMNet generator + checkpoint key map
    |   |-- resample.py              # reference-matched bicubic/bilinear resamplers
    |   |-- catalog.py               # variant facts: scale, peak, reference white, provenance
    |   |-- settings.py              # GMNetSettings record (env/TOML/CLI/--set bridge)
    |   |-- config.py                # typed GMNet TOML/CLI precedence and validation
    |   |-- expand.py                # preprocess, inference, HDR reconstruction, gain-map sidecar
    |   |-- convert.py               # GMNet key/variant/metadata conversion extension
    |   |-- converter_cli.py         # `weights convert gmnet` argument set
    |   |-- cli.py                   # complete --model gmnet inference argument set
    |   `-- weights/                 # converted weights (gitignored) + provenance docs
    `-- ltx2/                        # LTX-2.3/LTX-2.5 distilled pipeline
        |-- __init__.py
        |-- sigmas.py                # LTX-2 sigma constants + scheduler params
        |-- signals.py               # LTX-2 concrete SDR and HDR signal constants
        |-- cli.py                   # LTX-2-specific CLI flag group
        |
        |-- transformer/             # 22B DiT (velocity prediction)
        |   |-- __init__.py
        |   |-- attention.py         # MHA + 3D / 1D RoPE wiring
        |   |-- rope.py              # 3D ROPE for video, 1D for audio
        |   |-- feed_forward.py      # GELU MLP (BF16 default; optional FP16 transformer lane)
        |   |-- timestep.py          # timestep MLP + checkpoint-shaped AdaLN projections
        |   |-- preprocessing.py     # per-modality masks, RoPE, and A/V timestep preparation
        |   |-- transformer.py       # one joint A/V DiT block
        |   |-- wrappers.py          # Modality (timesteps + sigma), X0Model (velocity -> x0)
        |   `-- model.py             # 48-block LTXAVModel orchestration + cache streaming
        |
        |-- video_vae/               # metadata-selected native MLX video VAE
        |   |-- __init__.py
        |   |-- blocks.py            # Conv3d, residual, PixelNorm, BFHWC packing
        |   |-- config.py            # validated checkpoint-driven architecture
        |   |-- ops.py               # BCFHW patch packing + latent statistics
        |   |-- encoder.py           # pixels -> normalized latent
        |   |-- decoder.py           # convolutional normalized latent -> RGB
        |   |-- diffusion_decoder.py # bounded NA diffusion latent -> RGB
        |   |-- loading.py           # complete encoder/decoder checkpoint loader
        |   `-- tiling.py            # memory-bounded streaming decode + blending
        |
        |-- audio_vae/               # audio <-> latent -> mel -> waveform
        |   |-- __init__.py
        |   |-- config.py            # validated checkpoint-driven VAE architecture
        |   |-- blocks.py            # causal Conv2d, residual, norm, resampling blocks
        |   |-- encoder.py           # stereo log-mel -> normalized latent
        |   |-- decoder.py           # latent -> mel spectrogram
        |   |-- loading.py           # complete encoder/decoder family loader
        |   |-- vocoder.py           # FP32 BigVGAN + bandwidth-extension composition
        |   |-- vocoder_layers.py    # 1D convolution, SnakeBeta, AMP, and resampling layers
        |   |-- vocoder_loading.py   # complete BWE-vocoder family loader
        |   `-- vocoder_stft.py      # checkpoint-backed causal STFT and mel projection
        |
        |-- text_encoder/            # shared tokenizer + Gemma 3/4 12B + AV connector
        |   |-- __init__.py
        |   |-- _layers.py           # allocation-light Linear/Embedding shells
        |   |-- _loading.py          # centralized consumed-target preflight and binding
        |   |-- tokenizer_cache.py   # tokenizer.json -> content-addressed SentencePiece cache
        |   |-- tokenizer.py         # metadata-driven BOS, padding, mask, and decode policy
        |   |-- gemma3.py            # Gemma 3 architecture adapted from mlx-lm
        |   |-- gemma3_loading.py    # complete sharded/single-file Gemma 3 target binder
        |   |-- gemma4.py            # LTX-tuned Gemma 4 backbone and parity boundaries
        |   |-- gemma4_loading.py    # complete packaged Gemma 4 target binder
        |   |-- features.py          # per-token RMS aggregation and AV projections
        |   |-- connector.py         # independent audio/video connector blocks
        |   |-- loading.py           # monolithic/split projection and connector binder
        |   `-- encoder.py           # generation-selected, sequential text orchestration
        |
        |-- upscaler/                # public spatial/temporal latent upscalers
        |   |-- __init__.py
        |   |-- spatial.py           # normalized-latent wrapper + raw 2x spatial model
        |   `-- temporal.py          # temporal x2 component + VAE-statistics station
        |
        |-- conditioning/            # image conditioning (first-frame, keyframes)
        |   |-- __init__.py
        |   |-- item.py              # encoded-condition structural protocol
        |   |-- keyframe.py          # insert image at arbitrary frame
        |   |-- latent.py            # latent-replacement at frame 0
        |   |-- preparation.py       # raw-source -> encoded-condition station
        |   `-- tools.py             # shape-bound patch/mask/position helpers
        |
        |-- patchifier.py            # audio/video tokens, grid bounds, pixel coordinates
        |
        |-- pipelines/               # composition entry points
        |   |-- __init__.py
        |   |-- distilled.py         # public distilled recipe and station products
        |   `-- restart.py           # stage-2/direct-decode recipe shared by API and CLI
        |
        |-- components.py            # public component ports, providers, and leases
        |-- duration.py              # optional natural-duration head + VAE-grid snapping
        |-- generated_keyframes.py   # stage-1 generated-slot layout, mask, and cleanup
        |-- artifacts.py             # LTX-2 artifact schemas, names, and sidecar paths
        |-- resources.py             # immutable checkpoint/cache inventory
        |-- settings.py              # LTX-2 checkpoint and transformer policy
        |-- text_conditioning.py     # encoder-neutral station, product, provenance
        |-- denoise.py               # pure: denoise_step, denoise_loop
        |-- encode.py                # pure: encode_text, encode_image
        |-- decode.py                # audio decode + lazy owned video-frame stream
        |-- state.py                 # latent-state construction + conditioning setup
        |-- types.py                 # distilled request and generation-neutral shape contracts
        |-- runner.py                # generic recipe host over resources + stateless ports
        |
        `-- cache/                   # disposable, schema-keyed weight artifacts
            |-- __init__.py
            |-- building.py          # bounded-memory transformer conversion + sharding
            |-- family.py            # auxiliary family build/validation lifecycle
            |-- layout.py            # Conv3d NCDHW -> NDHWC + FF pretranspose baking
            |-- lora.py              # direct-strength LoRA mapping, fusion, and knockout
            |-- policy.py            # canonical layout defaults and cache identity aliases
            |-- quantization.py      # cache-backed quantized linear installation
            |-- storage.py           # safetensors read/write + sidecar metadata
            |-- schema.py            # cache identity, invalidation, and artifact paths
            |-- streaming.py         # bounded-residency transformer block rebinding
            |-- transformer.py       # prepared transformer cache build/bind lifecycle
            |-- weights.py           # dtype conversion + quantization routing
            `-- keys/                # weight key remappings kept apart from loader code
                |-- __init__.py
                |-- transformer.py   # PT -> MLX rules for the DiT stack
                |-- text_encoder.py  # Gemma 3 + AV connector remap
                |-- video_vae.py     # structure-driven lookup helpers (paired with _iter_convs)
                |-- audio_vae.py     # audio decoder + vocoder explicit keys
                `-- lora.py          # LoRA delta -> model param path map
```

## Layer Responsibilities

The package is organized as concentric layers. Higher layers compose
lower layers; lower layers never reach into higher ones.

### Layer 1 - Pure math and primitives (model-agnostic)

| Module | Owns |
|---|---|
| `kinomlx/samplers/` | Sigma schedules, Euler steps, centralized normal streams, Gaussian injection |
| `kinomlx/lora/` | Generic LoRA fusion math (`W += alpha*B@A`), safetensors LoRA parsing |
| `kinomlx/kernels/` | Custom Metal ops used by the transformer hot path |
| `kinomlx/audio/` | Audio signal-processing helpers (onset detection, ...) |
| `kinomlx/media/` | Immutable signal/delivery vocabulary and closeable frame ownership |

Pure functions, no I/O, no side effects. These would work for any
diffusion model, not just LTX-2.

### Layer 2 - Framework I/O (model-agnostic)

| Module | Owns |
|---|---|
| `kinomlx/io/` | Image input, weight I/O, and atomic file publication |
| `kinomlx/weights/` | Restricted generic checkpoint reading and verified conversion publication |
| `kinomlx/videotoolbox/` | Apple VT output subsystem (encode, mux, VSR, etc.) |
| `kinomlx/output.py` | Typed generation-to-artifact boundary and native VideoToolbox sink |
| `kinomlx/profiling/` | `os_signpost` instrumentation, xctrace analysis |
| `kinomlx/artifacts.py` | Model-neutral immutable tensor envelope and persistence port |
| `kinomlx/debug/` | Generic orchestration-owned sidecar writer and host metadata |
| `kinomlx/ui/` | CLI terminal adapters (Rich logging, machine stdout, progress) |

Touches the outside world. Still model-agnostic.

### Layer 3 - Configuration

| Module | Owns |
|---|---|
| `kinomlx/settings.py` | Infrastructure settings plus the sole environment/CLI parsing mechanism |
| `kinomlx/config/` | TOML load/merge/typed overrides/schema/dump primitives |
| `kinomlx/types.py` | Shared dataclasses (`LatentState`, `VideoLatentShape`, ...) |

The `EnvironmentSettings` mechanism in `kinomlx.settings` is the only code that
reads `os.environ`. Root `Settings` contains only model-neutral infrastructure.

### Layer 4 - Per-model packages

`kinomlx/models/<name>/` - see the two detail trees above. Each model is
self-contained and owns only the network, preprocessing, recipes, settings,
and terminals its modality needs.

### Layer 5 - Application entry

| Module | Owns |
|---|---|
| `kinomlx/cli/` | Root utility dispatch, --model inference dispatch, and CLI-to-sink configuration |

## Per-Model Conventions (`models/<name>/`)

Every model package follows the same public lifecycle even when its internal
network topology and modality differ:

| File / dir | Role |
|---|---|
| `__init__.py` | Lazily re-exports the model's runner, recipes, requests, and resources. |
| `types.py` | Immutable typed request and model-owned product contracts. |
| `resources.py` | Prepared immutable paths/capabilities/policies, never live model instances. |
| `components.py` | Injectable ports and bounded `ComponentLease` providers. |
| `pipeline.py` or `pipelines/<name>.py` | Stateless top-level recipe composition. |
| `runner.py` | Generic `run(recipe, request)` host retaining resources and stateless ports. |
| `settings.py` | Model-specific checkpoint, cache-layout, and execution policy. |
| `cli.py` | Model-specific inference arguments or complete modal parser. |
| `config_schema.py` | Runtime-light contribution to the global config registry. |
| `output.py`, `artifacts.py` | Optional model-owned terminal or sidecar contracts. |
| `convert.py` | Optional exact model checkpoint validation layered on `kinomlx.weights`. |

Diffusion-only pieces such as sigmas, transformer, VAEs, text encoders, LoRA,
and latent caches belong to LTX-2; they are not artificial requirements for a
small convolutional still model.

A model whose modality does not fit the flat generation grammar (for
example `gmnet/`, a still-expansion tool) registers as a *modal model*
in `kinomlx/cli/_registry.py` instead of contributing an argument
group: `--model <name>` routes the whole invocation to the model's own
`cli.py`, which defines a complete inference parser reusing the shared flag
conventions (`--image`, `--output`, `--output-dir`, `--config`, `--set`) and
resolves its settings record with the standard
defaults < env < TOML `[model_settings]` < CLI < `--set` precedence.
The bootstrap selection itself follows config < `--model` < `--set model=...`,
so config-only model selection reaches the same parser. Non-inference utilities
do not pretend to be model actions: `kinomlx weights convert` is generic and
dispatches `kinomlx weights convert gmnet` to the model-owned extension.

Every selectable model also contributes one complete `ModelConfigSpec` through
`config_schema.py`. The host collects those contributions in one global,
runtime-light registry. Shared infrastructure settings are one immutable table
contribution reused by every model. Dataclass coverage, parser destinations and
choices, accepted TOML tables, environment metadata, starter templates, and
full/sparse serialization all resolve through this registry; model packages do
not require edits to a central field union.

## API Surface

Two layers, both first-class.

**High-level** - used by the CLI and most user scripts:

```python
from kinomlx import LTX2Settings, Settings
from kinomlx.models.ltx2 import DistilledRequest, LTX2Runner, generate_distilled

runner = LTX2Runner(
    LTX2Settings.from_env(),
    infrastructure=Settings.from_env(),
)
with runner.run(generate_distilled, DistilledRequest(prompt="...")) as output:
    for frame in output.frames:
        consume(frame)
```

Saved-station restart is another public recipe, not a branch inside the generic
runner or the ordinary distilled recipe:

```python
from pathlib import Path

from kinomlx import DistilledRequest, DistilledRestart

request = DistilledRequest(width=768, height=448, frames=121)
selection = DistilledRestart.decode(Path("outputs/source.safetensors"))
with runner.restart(request, selection) as output:
    consume(output)
```

`DistilledRestart.stage_2(...)` selects the shared upscaler, stage-2 denoise,
decode, and terminal path from a stage-1 latent plus text-conditioning product.
The CLI resolves its run manifest into this same typed selection and invokes
the same `restart_distilled()` recipe. VSR, frame-rate conversion, and media
encoding remain terminal responsibilities downstream of either recipe.

`LTX2Runner` prepares one immutable resource inventory and retains only
stateless component, text-conditioner, reporter, and artifact ports. The
generic host contains no distilled branch: callers supply the public recipe
and its typed request. Each recipe releases live components at bounded station
boundaries. Video decoder ownership transfers to the lazy frame stream and
ends on exhaustion, early close, or output failure.

**Direct recipe** - the shortest form for a standalone generation:

```python
from kinomlx import LTX2Settings, Settings
from kinomlx.models.ltx2.pipelines.distilled import generate_distilled
from kinomlx.models.ltx2.resources import prepare_resources
from kinomlx.models.ltx2.types import DistilledRequest

resources = prepare_resources(
    LTX2Settings.from_env(),
    infrastructure=Settings.from_env(),
)
request = DistilledRequest(prompt="...")
with generate_distilled(request, resources) as output:
    for frame in output.frames:
        consume(frame)
```

`LTX2Resources` contains paths and policies, never live model instances.
`generate_distilled` leases native components at recipe-owned lexical
boundaries. Its returned frame stream is intentionally caller-owned. The
executable [`examples/ltx2_distilled.py`](../examples/ltx2_distilled.py)
expands the same recipe into every public station and makes the complete
transformer profile reuse/reload branch visible. Terminal hosts pass that
generation and an explicit `OutputColorPlan` to a `GenerationSink`; the
native `VideoToolboxGenerationSink` validates the signal/delivery pair before
opening the encoder and closes the generation on success or failure.
The terminal activates `CutDetector` only when balanced Video-input VSR or
VTFRC retains adjacent-frame state. At a detected cut, VSR drops its explicit
previous source/output pair before the post-cut frame; VTFRC emits the prior
source period as a hold and restarts its sequential session instead of
interpolating across the discontinuity. This is terminal media policy rather
than model logic because generated clips and supplied streams can both cut.

GMNet uses the same lifecycle without inheriting video abstractions:

```python
from pathlib import Path

from kinomlx import (
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    expand_gmnet,
    plan_gmnet_output,
)

request = GMNetRequest(Path("photo.jpg"))
runner = GMNetRunner()
plan = plan_gmnet_output(request, GMNetOutputConfig(directory=Path("hdr")))
with plan.reserve() as reservation:
    result = runner.run(expand_gmnet, request)
    artifacts = GMNetOutputSink(plan).write(result, reservation=reservation)
```

The reservation precedes inference and the sink publishes the selected EXR,
HEIC, and gain-map artifacts together. `GMNetRunner.expand()` is the standard
recipe convenience, while injected `GMNetComponents` make tests and alternate
hosts independent of the native loader. The small composition is
[`examples/gmnet_expand.py`](../examples/gmnet_expand.py).

Checkpoint conversion has a separate public namespace and command hierarchy:

```python
from kinomlx.weights import convert_checkpoint

receipt = convert_checkpoint("checkpoint.pth", "checkpoint.safetensors")
```

That generic API preserves tensor values/layouts and only selects or renames
keys explicitly. Models that need stronger semantics own an extension; GMNet's
`kinomlx.models.gmnet.convert.convert_checkpoint` adds exact keys, published
variant identity, provenance metadata, and a full GMNet loader verification.
The corresponding CLIs are `kinomlx weights convert` and
`kinomlx weights convert gmnet`.

TOML keeps `[settings]` model-neutral. LTX-2 checkpoint/transformer policy is
under `[model_settings]`, while LTX-2 tensor sidecar choices are under
`[model_artifacts]`. The familiar artifact CLI flags are contributed by the
model package; stale model keys in `[settings]` or `[output]` fail schema
validation rather than passing through a compatibility translation.

## Conventions

### File size

Soft ceiling: **~500 lines per file**. If a file is getting bigger,
it has more than one concern. Split by concern, not by alphabetical
quota.

### Pure functions over methods

Anywhere the math allows it. Classes are reserved for genuine state
(loaded model weights, KV caches, open file handles). Pipeline
composition reads top-to-bottom like a recipe; state flows through,
not on `self`.

### No cross-model imports

Code inside `kinomlx/models/ltx2/` never imports from
`kinomlx/models/wan/` (or any future sibling). Each model is
self-contained. Shared math is at top-level (`kinomlx/samplers/`,
`kinomlx/lora/`, etc.).

### One source of truth for configuration

Anything env-derived uses the `EnvironmentSettings` mechanism in
`kinomlx.settings`. No module calls `os.environ.get(...)` directly. Every
setting field auto-generates a matching `--kebab-case` CLI flag via the
shared argparse bridge, so the CLI override
surface stays in lockstep with the dataclass without per-field
boilerplate.

### Weight key remappings live as data

`PyTorch key -> MLX key` mappings sit in `cache/keys/*.py` (one file per
subsystem) - not inline with the loader code. The loader iterates the
rules and dispatches; the loader and the mappings are independently
readable. Video VAE / audio VAE loading is structure-driven (iterate
module tree) so those helpers stay close to the iteration logic.

### BF16 default, FP32 where precision matters

Activations and most weights are BF16 by default. Precision-sensitive sites
stay FP32: AdaLN ``scale_shift`` tables and RoPE sincos. Cast back to the
transformer activation dtype immediately after the FP32 broadcast. The cache
loader accepts FP8 E4M3 checkpoints and dequantizes them to a runnable dtype
during the one-time build. An optional FP16 transformer cache/storage lane is
implemented; the transformer consumes that dtype for compute. FP16 remains
an opt-in draft lane, not a memory-saving mode.

LTX-2.3 distilled parity details to preserve:

- The checkpoint requests `frequencies_precision=float64`; build the RoPE
  frequency grid through the double-precision path, then cast final cos/sin
  to the hidden dtype. Do not silently fall back to the fp32 grid.
- RoPE tables are stage-stable, so a per-stage self/cross precompute path is
  valid, but do not assume it is a win. On M1 Max, measured stage-2
  bench-mode measured `96.945s` with resident self + A/V cross RoPE tables vs
  `96.480s` recomputing through the normal path; keep this as an A/B knob,
  not a default, unless KinoMLX measures differently.
- Keep sigmas, timestep basis, Euler / X0 update math, and noising scalars in
  FP32, then cast the updated latent back to the payload dtype.
- `av_ca_timestep_scale_multiplier=1000` is expected for the 2.3 distilled
  checkpoint and matches upstream A/V cross-attention gate conditioning.
- Image/keyframe conditioning tokens should be cast to the current latent
  dtype before concatenation; otherwise a BF16 denoise state can silently
  promote to FP32.
- Defensive precision guards retained by KinoMLX: reshape float
  additive masks just like bool masks; fused AdaLN must fall back when `eps`,
  dtype, or broadcast pattern does not match its hardcoded BF16 kernel; if a
  fused interleaved RoPE path exists, read `x`, `cos`, and `sin` as FP32 and
  only cast the final result back to activation dtype.
- BWE vocoder is an FP32 island that returns the input dtype in upstream.
  Returning FP32 is a possible quality experiment, not strict parity.

### Memory targets

Recommended target: **64 GB unified memory** (e.g. M-series Studio or
high-end MacBook Pro). Default settings keep all transformer blocks resident
and cap the allocator cache at 1 GiB. Gemma and the transformer are loaded
sequentially and do not need to coexist in memory.

Supported minimum: **32 GB**. Use `--stream-transformer` for the validated
resident-block/compile-group preset, or configure the lower-level resident
block controls explicitly. Streaming trades step time for a bounded
transformer live set.

### Host-neutral reporting and centralized terminal output

Runtime code logs through stdlib `logging` and reports live phase progress
through `kinomlx.reporting.Reporter`; it does not import Rich or own stdout.
The CLI binds those surfaces to `RichReporter` and one shared stderr console.
Machine-readable JSON and `--print-config` use isolated message-only stdout
loggers, while `--save-config` uses the identical TOML serializer without
writing to stdout. Human timestamps and progress rows therefore cannot corrupt
the protocol.
Quiet mode disables live rows while retaining phase accounting; its logging
level filters summaries. `KINO_VERBOSE=1` also preserves native VideoToolbox
compile diagnostics that are normally suppressed around VSR session startup.
Bare `print` remains forbidden in the package.

### On-demand native profiling shim

No architecture-specific signpost library is checked into the repository.
When signpost profiling is explicitly enabled, KinoMLX compiles
`profiling/_signpost.c` with the system Clang and caches a source-and-architecture
keyed dylib under the configured KinoMLX cache. Ordinary generation performs
no native build, and a profiling-shim build failure disables signposts without
aborting generation.

### Inference is pure MLX

Model execution imports MLX and SentencePiece. A cold tokenizer-cache build
lazily imports protobuf to serialize the derived SentencePiece model; warm
loads do not. No torch, numpy, safetensors, Transformers, Tokenizers, or
Hugging Face Hub is imported at runtime; model weights are pulled with the
`hf` CLI ahead of time.

## Cross-References

- `resources.py` prepares cache artifacts and `components.py` binds or fuses
  them into bounded leases. Recipes consume only those public providers and
  station products; they never import the `cache/` subpackage.
- `cache/lora.py` mediates between the generic `kinomlx/lora/`
  fusion math and the LTX key/category rules in `cache/keys/lora.py`.
  Multiple adapters retain independent scalar and per-stage totals, and each
  adapter can exclude overlapping branch, module, projection, or control-path
  categories. The distilled recipe resolves two complete ordered profiles. It
  reuses one transformer only when those profiles are identical; otherwise it
  closes stage 1 and loads a pristine prepared-cache transformer for stage 2.
- `pipelines/distilled.py` composes `denoise.py` + `encode.py` +
  `decode.py` + `state.py`. Its final video product stays lazy through
  `output.py` and `videotoolbox/`; other pipelines follow the same
  ownership rule.
- `debug/sidecars.py` owns optional text, VAE-encoded media-conditioning, and
  generated-latent artifacts.
- `cli/output.py` resolves CLI settings into an explicit color plan and
  delegates final video, mandatory HDR EXR, optional PQ HEIC, and optional WAV
  writing to `output.py`.
