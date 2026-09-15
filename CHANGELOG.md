# Changelog

All notable user-facing changes to KinoMLX are recorded here. The format follows
Keep a Changelog, and versions follow semantic versioning.

## 0.0.2 - 2026-09-15

This release adds two still-image model families, Krea 2 and FLUX.2 Klein 9B, so KinoMLX now
covers four models behind one CLI. It also retires the generated on-disk weight caches in favor
of loading and preparing components directly from the original checkpoints, replaces the LoRA
controls with a general adapter surface that takes ordered records and many more export formats,
and adds seed lists, per-image restart, denoise-schedule control, and a wired-memory limit.
Configuration, environment variables, run records, and the LTX machine-readable result changed in
ways that require edits to existing setups; read Breaking changes first.

### Breaking changes

- Generated component weight caches are gone. `--cache-mode` and `KINO_CACHE_MODE` no longer
  exist; delete them from scripts and configuration files. Components now load from the original
  checkpoint on every run, so no cache build step remains.
- `--cache-dir` and `KINO_CACHE_DIR` are replaced by `--app-cache-dir`,
  `[settings].app_cache_dir`, and `KINO_APP_CACHE_DIR`, which now hold only converted GMNet
  weights and native profiling builds. The default moved to `~/Library/Caches/KinoMLX` (or
  `XDG_CACHE_HOME/KinoMLX`), which is also where GMNet conversions land in a non-checkout install.
- Model environment variables are now scoped by model as `KINO_<MODEL>_<FIELD>`. Rename
  `KINO_WEIGHTS_PATH` to `KINO_LTX2_WEIGHTS_PATH`, `KINO_LTX_GENERATION` to
  `KINO_LTX2_MODEL_GENERATION`, `KINO_TRANSFORMER_DTYPE` to `KINO_LTX2_TRANSFORMER_DTYPE`, and
  every other unscoped LTX variable the same way; `KINO_GMNET_WEIGHTS` becomes
  `KINO_GMNET_WEIGHTS_PATH`.
- An unrecognized `KINO_<MODEL>_*` variable is now a configuration error instead of being
  ignored, so a stale or misspelled export stops the run until it is unset.
- `--transformer-cache-quantize` is now `--transformer-quantize`, with
  `KINO_LTX2_TRANSFORMER_QUANTIZE` and `[model_settings].transformer_quantize`. The three choices
  are unchanged.
- LTX adapter configuration moved from parallel arrays to ordered records. Replace
  `[generate].lora_paths`, `lora_strengths`, `lora_stage1_strengths`, `lora_stage2_strengths`, and
  `lora_exclusions` with one `[generate].adapters` array whose records carry `path`, `strength`,
  `exclude`, `stage1_strength`, and `stage2_strength`; `DistilledRequest` takes
  `adapters=(LTX2AdapterConfig(...),)` instead of the old keyword arrays. The long option
  spellings `--lora-strength-stage-1` and `--lora-strength-stage-2` are removed in favor of
  `--adapter-stage1-strength` and `--adapter-stage2-strength`.
- Model classes are no longer re-exported from the `kinomlx` root. Change
  `from kinomlx import LTX2Runner` to `from kinomlx.models.ltx2 import LTX2Runner`, and likewise
  for GMNet; `prepare_gmnet_resources` is now `kinomlx.models.gmnet.prepare_resources`. The root
  package keeps `Settings`, the output sinks, and the memory-policy helpers.
- The `kinomlx.lora` package is now `kinomlx.adapters`; update any direct imports of its fusion
  or loading helpers.
- Saved run records use schema version 2. Records written by 0.0.1 are refused with no automatic
  conversion, so a restart needs a run produced by this version.
- The LTX `--json` record changed shape. It no longer carries a top-level `output`; it carries
  `seeds` and a `runs` array with one entry per seed, each entry holding that run's former
  payload plus its `seed` and `seed_index`. Read `runs[0].output` where you previously read
  `output`.
- LTX generated filenames changed from `kinomlx_<timestamp>.mp4` to
  `ltx2-distilled-seed<seed>_<timestamp>.mp4`. Pass `--output-prefix kinomlx` to keep the old
  stem.
- On every model the output directory now resolves `KINO_<MODEL>_OUTPUT_DIR` first, then the
  shared `KINO_OUTPUT_DIR`, then `outputs/`. GMNet generated names now also carry the
  `_YYYYMMDD_HHMMSS` timestamp that LTX already used.
- An exact LTX `--output` must end in `.mp4`; a path such as `clip.mov` is refused at
  configuration time instead of silently being written as `clip.mp4`.

### Added

- Krea 2 (`kinomlx --model krea2`) generates Raw and Turbo still images, selected with
  `--variant raw|turbo`, and writes lossless PNG or TIFF. Turbo defaults to eight guidance-free
  steps; Raw defaults to 52 steps with CFG.
- Krea 2 reference editing accepts one or two `--source-image` references in scene/subject order
  with a compatible edit adapter, controlled by `--edit-adapter`, `--edit-adapter-strength`,
  `--edit-adapter-exclude`, `--reference-fit fit|crop`, and `--match-source-aspect`.
- Krea 2 grounding and reference-attention controls: `--grounding-px`,
  `--grounding-system-prompt`, `--reference-strength`, `--scene-reference-strength`,
  `--reference-boost-region` (repeatable, percentage coordinates), and `--reference-boost-mask`.
- Krea 2 first-class adapter profiles: `--turbo-4step` selects the community four-step Turbo LoRA
  recipe and `--turbo-sda` applies the diversity adapter gated to its recorded training band,
  each with its own `-strength`, `-exclude`, `-step-strengths`, and `-sigma-window` controls. A
  profile supplies its sampling values only where the request leaves them unset, so explicit
  steps, guidance, shift, or `--sigmas` win and are announced as deviations.
- Other Krea 2 controls: `--vae-decoder upscale2x` for the optional aesthetic 2x decoder (which
  changes output size and appearance), `--transformer-dtype`, `--vae-dtype`,
  `--reference-denoise` / `--no-reference-denoise`, `--vae-tiling` with `--vae-tile-size` and
  `--vae-tile-stride`, and `--system-prompt`.
- FLUX.2 Klein 9B (`kinomlx --model flux2`) generates still images from the base, distilled, and
  KV checkpoints selected with `--variant base|distilled|kv`, writing lossless PNG or TIFF, and
  supports ordered multireference editing through repeated `--source-image` with
  `--reference-fit` and `--match-source-aspect`. It runs the checkpoint's native BF16 by default.
- Klein execution controls: `--cfg-mode sequential|batched` for base guidance branches,
  `--transformer-dtype bfloat16|float16`, `--dry-run` to describe validated local resources, and
  `--recompute-reference-kv`, which re-extracts the reference K/V every step and is what makes a
  per-step adapter schedule or sigma window usable with KV editing.
- `kinomlx config inspect --model MODEL` prints the registered configuration inventory as JSON for
  any of the four models: typed ownership, CLI and environment bindings, normalization rules,
  public API parameter roles, and each field's restart decision with its reason.
  `--field generate.seed` narrows it to one field, and nothing is read from invocation files or
  weights.
- Prompt input: `--prompt-file` loads one complete UTF-8 prompt from a file for LTX2, Krea 2, and
  Klein, preserving internal line breaks and storing the resolved text for replay;
  `--prompt-unconditional` deliberately generates with an empty user prompt while keeping normal
  text encoding, templates, adapters, and image conditioning.
- Ordered seed lists on LTX2, Krea 2, and Klein: `--seed 42 123 999` (or `--seed 42,123,999`, with
  `--seeds` as an alias) runs each seed in order and preserves duplicates, while
  `--images-per-prompt N` generates N consecutively seeded images per prompt on Krea 2 and Klein.
  All three reuse seed-independent conditioning between seeds. GMNet takes no seed.
- `--seed-mode sequential|station|resident` selects model lifetimes across seeds on Krea 2 and
  Klein: normal per-seed loading, one loaded station at a time, or transformer and VAE held
  resident for the whole request. LTX processes complete videos sequentially and takes no
  `--seed-mode`.
- A run that produces more than one output (Krea 2 or Klein images, or one LTX video per seed)
  publishes each completed output immediately, names it with its output index and seed, and
  preserves finished files when a later output fails or is interrupted.
- `--restart-index` (one-based, matching the output number in the filename) selects which image of
  a multi-image Krea 2 or Klein run to resume, narrowing the request to that image's prompt and
  seed. A run stopped with Ctrl-C is restartable for every image it published; selecting an image
  it never reached is refused by name, and a failed run stays refused.
- Cancellation is reported the same way by every model: status `canceled` with the outputs
  already published, in both the human and `--json` surfaces, and exit code 130.
- LTX2, Krea 2, and Klein take adapters with an inline strength suffix,
  `--adapter style.safetensors:0.8`, and LTX additionally accepts a two-value form for absolute
  stage-1 and stage-2 strengths, `--adapter style.safetensors:0.25:0.5` (zero skips that stage).
  Modifiers such as `--adapter-strength` and `--adapter-exclude` bind to the latest preceding
  `--adapter`, and repeating one for the same item is an error. GMNet has no adapter surface.
- Krea 2 and Klein accept a much wider set of static adapter export families, including LoKr,
  LoHa, DoRA/DoHa/DoKr, AdaLoRA, LoCon, GLoRA, IA3, OFT/BOFT, HRA, FourierFT, WaveFT, VeRA,
  RandLoRA, VBLoRA, UniLoRA, TinyLoRA, and saved full-weight or bias replacements. LTX accepts
  ordinary two-dimensional LoRA.
- `--adapter-step-strengths` sets a Krea 2 or Klein adapter's absolute strength at each denoise
  step, and `--adapter-sigma-window` activates it only inside ordered noise-level bands written as
  `low[:high][@strength]`. The two spellings are mutually exclusive. A varying adapter stays
  unfused, which limits it to LoRA, AdaLoRA, LoKr, and bounded-rank LoHa; an array that repeats
  one value for every step is a constant and fuses at that value instead.
- Adapter validation on LTX2, Krea 2, and Klein: `--check-adapter-weights` /
  `--no-check-adapter-weights` (with `[settings].check_adapter_weights` and
  `KINO_CHECK_ADAPTER_WEIGHTS`) controls the one-time scan of adapter-modified active weights after
  their first denoise step, on by default. All three also check active latents for nonfinite values
  after every denoise step and stop with the step named. GMNet has no adapters or denoising stage,
  so the flag is not offered there.
- Adapters work with LTX block streaming and with MXFP8-quantized LTX transformers, both of which
  refused them in 0.0.1. `--adapter-accumulation auto|float32|float16` selects the precision
  fusion accumulates in, with `auto` using float16 only for a streamed float16 transformer.
- LTX2, Krea 2, and Klein each report their resolved denoise schedule in the sampling section of
  the console and record the complete array in run metadata. GMNet does not denoise and has no
  schedule.
- `--sigmas` sets an explicit denoise schedule for Krea 2 and Klein, and `--sigmas-stage-1` /
  `--sigmas-stage-2` set LTX's two stage schedules independently. The array holds one more value
  than the steps it drives and therefore fixes the step count, and a schedule can be changed when
  resuming at the station that consumes it.
- `--mlx-wired-limit-gb` (with `[settings].mlx_wired_limit_gb` and `KINO_MLX_WIRED_LIMIT_GB`) caps
  how much MLX memory stays resident on all four models; unset reads the device's recommended
  working set and zero disables wiring. `kinomlx.apply_memory_policy(...)`,
  `kinomlx.MLXMemoryPolicy`, and `kinomlx.memory_managed` let standalone component callers and
  custom recipes apply the same policy that runners and public recipes apply automatically.
- Other new LTX controls: `--transformer-retain` keeps selected block tensors resident across
  streamed denoising steps (class selectors such as `small` and `video-ff`, or globs over
  block-relative module names), `--transformer-quantize-exclude-layers` leaves chosen blocks at
  the transformer dtype, and `--vae-decode-dtype float16` adds experimental FP16 compute for the
  diffusion video VAE.
- Restart bundles are themselves restartable. An LTX decode restart re-emits the final latents and
  a stage-2 restart re-emits the saved text and stage-1 latent, where 0.0.1 produced no artifacts
  from a decode restart at all; Krea 2 and Klein offer the equivalent at their own stations. GMNet
  has no restart operation.

### Changed

- Model components load from their original local checkpoints, with dtype and layout transforms
  prepared lazily and evaluated on first use, and tokenizer tables derived in memory from the
  checkpoint's own JSON assets. There is no second converted copy on disk, no cache build phase,
  and no fingerprint or model-view publication.
- LTX FP8 E4M3 sources are dequantized into the selected compute dtype during loading instead of
  during a cache build, so no expanded copy has to fit on disk alongside the compact source.
- The LTX `--stream-transformer` preset keeps 8 of 48 blocks resident as two four-block groups and
  runs them uncompiled, dispatching each group so its reads and preparation overlap the previous
  group's computation. Streamed and resident outputs now match bit for bit when their accumulation
  and execution settings match, and `--transformer-compile-group-size` still compiles groups on
  request.
- LTX prompt encoding and the untiled decode paths build one lazy graph per call and evaluate it
  once instead of evaluating between stages, so their progress bars pulse rather than counting
  stages and prompt encoding gets measurably faster at unchanged peak memory.
- LTX per-sidecar output toggles are now three-valued, so a Python caller building
  `OutputConfig(save_all_sidecars=True)` gets the run log, console log, effective config, and
  audio sidecar that the CLI already produced. `--save-hdr-heic-frames` and `--save-vae-frames`
  remain deliberately outside `--save-all-sidecars`, and the LTX help now also names the model
  artifact sidecars that save-all enables while they stay unset: latents, media conditioning, and
  text conditioning.
- A restart refusal names each field with the reason its station policy declares, instead of one
  generic sentence asserting that earlier-station values cannot be changed.
- A generated configuration template prints `Default: required.` for a field that has no default,
  such as `[restart].run`, instead of the `Default: unset.` that a reader could mistake for a
  usable value.
- LTX `output.audio_onset_trim` accepts a bare number as well as a quoted string, so
  `--set output.audio_onset_trim=120` is accepted where it previously failed with a type error;
  the saved configuration still stores the string form.

### Removed

- The generated component cache subsystem: cache building, sharding, content-addressed objects,
  published model views, source refs, and the cache-mode selector.
- On-disk tokenizer caches and their metadata sidecars.
- Root-level lazy re-exports of model classes from the `kinomlx` package.
- `--lora-strength-stage-1` and `--lora-strength-stage-2`, and the parallel-array adapter fields on
  public requests and in TOML.
- `--transformer-cache-quantize`, superseded by `--transformer-quantize`.
- Support for run records written before schema version 2.

### Fixed

- STEEL attention kernels compile again on macOS 27. Metal 4.1 rejects an unqualified member
  function returning a `thread` reference, which every 0.0.1 STEEL dispatch used, so every STEEL
  attention dispatch failed on that OS with a compile error. Kernel output is unchanged.
- An LTX VAE tile size written beside a decode mode that does not tile is now refused, naming the
  layer that holds the pair, instead of being dropped with no diagnostic.
- A configuration value containing a DEL character is escaped when written, so a file KinoMLX
  wrote can be read back instead of being refused as invalid TOML on its next use.

### Documentation

- Seven new guides: [Adapters](docs/ADAPTERS.md), [Denoise schedules](docs/SCHEDULES.md),
  [Weight loading](docs/WEIGHTS.md), [Performance](docs/PERFORMANCE.md), [Krea 2](docs/KREA2.md),
  [Klein](docs/FLUX2.md), and [Conventions](docs/CONVENTIONS.md).
- The adapter guide documents format-specific alpha and scaling rules, what makes an export
  self-contained, the formats that need a producer-side conversion, and coverage warnings. LTX
  applies strength directly and ignores saved alpha; Krea 2 and Klein apply export scaling first.
- The performance guide adds the meanings of zero for both memory controls, the measured device
  wired-memory default, an `iogpu.wired_limit_mb` table, an optional launchd boot override, and a
  measured pair of 30-second LTX-2.3 runs (17.4 minutes BF16, 15.7 minutes FP16, 41.1 GiB peak)
  with a per-phase breakdown.
- The LTX2 guide gains sections on machine-readable run results, denoise schedules, transformer
  quantization with measured memory and speed figures, selected tensor retention, and recorded
  decoder timings; "LoRA composition" is now "Adapter composition".
- The README was rewritten for four model families and now documents shared generation controls,
  prompt files, memory and performance, adapters and saved stations, seed lists and model
  lifetimes, output names and sidecars, and how to run the tests.
- The GMNet guide was rewritten around the user workflow, and its Python example now imports from
  `kinomlx.models.gmnet`.
- The LTX2, Krea 2, and Klein guides each add complete TOML invocation files, one generation
  with adapters and one source-image edit or conditioning run, ready for `kinomlx --config`.
- New runnable examples `examples/krea2_generate.py` and `examples/flux2_generate.py` join the LTX
  and GMNet examples and are type-checked, and `docs/STRUCTURE.md` was regenerated from the tree so
  every module and package under `kinomlx/` appears exactly once.
- Third-party notices now cover Krea 2, FLUX.2 Klein through Diffusers, Qwen through Transformers
  and mlx-lm, PEFT, PyWavelets coefficient data, and comfyui-krea2edit, each with the inspected
  source revision.

### Development

- The test suite runs in parallel by default (`-n auto`, one worker per physical core); pass `-n 0`
  for a single process when debugging or under `coverage run`.
- The `requires_weights` marker is now `heavy` ("loads real checkpoints or runs long GPU work"),
  deselected by default and selected with `pytest tests --run-heavy -m heavy`. A heavy run always
  uses one process, and a heavy test that reaches a parallel worker refuses to start rather than
  loading a checkpoint there. `slow` is now the sub-label for the heavy tests that take minutes, so
  `--run-heavy -m 'heavy and not slow'` runs the quick checkpoint checks.
- Cheap real-asset tests (checkpoint header reads, tokenizer comparisons, single small component
  loads) run in the default suite and resolve their inputs the way the product does: the model's
  environment settings first, then the local Hugging Face cache.
- The `dev` extra gains `pytest-xdist` (parallel execution) and `vulture` (an occasional dead-code
  sweep, not a gate).
- Ruff now also selects `TID251`, `PLC0415`, `ARG`, `FURB`, `PERF`, and `RUF`. The package uses
  Python 3.14 native deferred annotations with one import graph: no
  `from __future__ import annotations`, no `TYPE_CHECKING` branches, and no function-level imports
  outside tests.

## 0.0.1 - 2026-08-30

Initial public release: distilled LTX-2.3 and LTX-2.5 text/image-to-video generation with joint
stereo audio, conditioned HDR image and video workflows, LoRA composition, and staged restart;
plus GMNet SDR-to-HDR still expansion with transactional half-float EXR, 10-bit PQ HEIC, and
optional gain-map output. Both families ran behind one CLI with TOML invocation files,
`kinomlx config init`, the `kinomlx weights convert` checkpoint converters, and a public Python
API.
