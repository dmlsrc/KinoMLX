# KinoMLX - Project Structure

A map of every directory and module, what it owns, and why it lives
where it does. Read this when you're not sure where to put new code.

See [Project conventions](CONVENTIONS.md) for coding rules, model boundaries,
configuration ownership, precision requirements, and validation practices.

## Overview

KinoMLX is a focused multimodal inference engine on Apple Silicon via MLX. It
currently ships distilled LTX-2.3/LTX-2.5 video generation and GMNet SDR-to-HDR
still expansion. Krea 2 Raw/Turbo source validation, local component
preparation, native Qwen text conditioning, complete text-to-image generation,
and reference editing with a compatible adapter are also available. FLUX.2 Klein
supports base, distilled, and KV generation and ordered-reference image editing.
The project is designed to be **modular by purpose** and **extensible to new
models and modalities**:

- Each model is a self-contained subpackage under
  `kinomlx/models/<name>/`. No cross-model imports.
- Model-agnostic infrastructure (samplers, adapter fusion, I/O, UX,
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
|   |-- ADAPTERS.md                  # shared adapter formats, strength, and checks
|   |-- CONFIG.md                    # config files, environment, --set, shared arguments
|   |-- CONVENTIONS.md               # contributor rules for code, docs, and validation
|   |-- PERFORMANCE.md               # memory controls, precision, and comparing runs
|   |-- SCHEDULES.md                 # sigma schedules and adapter strength over a run
|   |-- STRUCTURE.md                 # this file
|   |-- USAGE.md                     # prompts, seeds, outputs, run records, restart
|   |-- WEIGHTS.md                   # model files, loading, and checkpoint conversion
|   |-- flux2/                       # Klein guides; README.md first, arguments.md catalog
|   |-- gmnet/                       # GMNet guides; README.md first, arguments.md catalog
|   |-- krea2/                       # Krea 2 guides; README.md first, arguments.md catalog
|   `-- ltx2/                        # LTX2 guides; README.md first, arguments.md catalog
|-- examples/                        # public recipe/runner compositions
|   |-- flux2_generate.py            # Klein public runner and image output
|   |-- gmnet_expand.py              # small still-expansion + transaction example
|   |-- krea2_generate.py            # runnable Raw/Turbo public API + still output
|   `-- ltx2_distilled.py            # explicit distilled ownership reference
|-- tests/                           # pytest tests (mirrors package structure)
|   |-- adapters/                    # static adapter formats, algebra, schedules, fusion
|   |-- cli/                         # root dispatch, shared arguments, config utilities
|   |-- config/                      # registry, precedence, stage policies, fixtures/
|   |-- io/                          # atomic publication, image input, safetensors, identity
|   |-- kernels/                     # shared fused-op parity
|   |-- media/                       # frame ownership and HDR transfers
|   |-- models/                      # per-model suites: flux2, gmnet, krea2, ltx2
|   |-- samplers/                    # Euler steps, noise streams, sigma schedules
|   |-- typecheck/                   # mypy-only contract modules
|   |-- ui/                          # console, logging, progress bars, reporting
|   |-- weights/                     # generic conversion and restricted Torch reading
|   `-- *.py                         # conftest, shared helpers, and root-module tests
|-- scripts/                         # small project utilities and demonstrations
|-- weights-src/                     # collected upstream source checkpoints (untracked; see its README)
|-- debug/                           # local runtime sidecar output (gitignored)
`-- kinomlx/                         # the importable Python package
```

## Package Tree (`kinomlx/`)

```text
kinomlx/
|-- __init__.py                      # version and genuinely shared public API
|-- _config_record_state.py          # supplied-argument presence for frozen records
|-- _json.py                         # recursive JSON value types
|-- _mlx_nn.py                       # typed access to the exact installed mlx.nn objects
|-- _mlx_nn.pyi                      # strict type surface for those exact objects
|-- _tensor_tree.py                  # recursive MLX Module.update tree type
|-- api_boundaries.py                # typed ownership of public API parameters
|-- artifacts.py                     # model-neutral tensor-artifact envelope and sink
|-- components.py                    # shared bounded component leases
|-- config_bindings.py               # neutral CLI contract shared by Settings and registry
|-- config_constraints.py            # pure local value constraints owned by typed fields
|-- config_record_cli.py             # argparse actions building records from item tokens
|-- config_records.py                # typed construction and presence for config records
|-- config_rules.py                  # field-owned validation of complete config records
|-- config_surfaces.py               # supported transports and reasons for unavailable ones
|-- config_values.py                 # dependency-free schema helpers for config records
|-- errors.py                        # typed operational error base
|-- huggingface.py                   # dependency-free local Hub cache/ref resolution
|-- image_restart_config.py          # shared still-image restart inputs and relationships
|-- invocation_controls.py           # control declarations shared by every host surface
|-- memory_policy.py                 # invocation-owned allocator and wired limits
|-- model_catalog.py                 # model names, help, and the default model
|-- model_registry.py                # global registry of every model's config contribution
|-- output.py                        # typed generation sink + VideoToolbox adapter
|-- prompts.py                       # shared explicit empty-user-prompt contract
|-- recipe_host.py                   # public recipe execution boundary over model ports
|-- reporter_stack.py                # the reporter stack a command builds around its work
|-- reporting.py                     # host-neutral Reporter protocol + test sinks
|-- request_rules.py                 # coupled policies shared by generation request fields
|-- restart_receipts.py              # observational receipts for the artifacts a restart reads
|-- restart_selection.py             # select one image of a multi-image run for a restart
|-- seed_execution.py                # request-scoped model ownership at image stages
|-- seeds.py                         # ordered seed selection shared by every generation family
|-- settings.py                      # infrastructure settings + env/CLI field machinery
|-- types.py                         # model-neutral shapes and compute-policy literals
|
|-- adapters/                        # shared static adapter infrastructure
|   |-- __init__.py                  # package namespace
|   |-- selection.py                 # resolved runtime selections shared by model pipelines
|   |-- source.py                    # local identity, embedded/companion config, scratch loading
|   |-- lora.py                      # low-rank pairs, application config, and LoRA math
|   |-- formats.py                   # strict tensor signatures and factor validation
|   |-- algebra.py                   # logical FP32 static adapter equations
|   |-- orthogonal.py                # PEFT OFT/BOFT and compact HRA equations
|   |-- peft.py                      # companion configuration and saved-parameter semantics
|   |-- configuration.py             # bounded scalar and pattern interpretation
|   |-- peft_formats.py              # declared PEFT field and shared-basis parsing
|   |-- peft_linear.py               # configured PEFT Linear equations in MLX
|   |-- basis.py                     # shared projection/vector-bank equations
|   |-- glora.py                     # PEFT GLoRA weight and bias paths
|   |-- fourier.py                   # FourierFT coefficient reconstruction
|   |-- permutation.py               # exact seeded Torch CPU location contract
|   |-- wavelet.py                   # WaveFT synthesis in MLX
|   |-- wavelet_filters.py           # attributed wavelet coefficient data
|   |-- semantics.py                 # static export and omitted-state boundaries
|   |-- fusion.py                    # bounded scratch preflight and lazy physical weight updates
|   |-- receipt.py                   # per-adapter tensor placement results
|   |-- scaling.py                   # intrinsic-scale determination and its console line
|   |-- config_schema.py             # adapter record contributions to the shared configuration
|   |-- huggingface.py               # offline selection of one community release file
|   |-- input_rules.py               # adapter relationships with supplied-strength ownership
|   |-- inputs.py                    # presence-preserving inputs, apart from selections
|   |-- live.py                      # scheduled adapters over activations, weights unchanged
|   |-- schedule.py                  # per-step strength arrays against one sampler recipe
|   `-- validation.py                # one first-use finite scan on a freshly adapted lease
|
|-- audio/                           # audio utilities (model-agnostic)
|   |-- __init__.py                  # audio post-processing utilities
|   |-- onset.py                     # start-of-clip click suppression
|   `-- trim_mode.py                 # sequence-start onset policy parsing and admission
|
|-- cli/                             # model-neutral utility and command dispatch
|   |-- __init__.py                  # namespace only; entry point targets main.py
|   |-- _registry.py                 # model command/config registry
|   |-- main.py                      # entry, utility routing, and model dispatch
|   |-- arguments.py                 # compact model selection and root help
|   |-- invocation.py                # shared run flags and bootstrap argument reads
|   |-- presentation.py              # shared human logging and typed error rendering
|   |-- machine_output.py            # isolated machine-readable JSON output
|   |-- config_arguments.py          # all-model parser generation from owned bindings
|   |-- config_schema.py             # one shared infrastructure-settings contribution
|   |-- config_init.py               # model-specific starter-config utility
|   |-- config_output.py             # shared full/sparse print and no-clobber save
|   |-- config_templates.py          # readable TOML renderer over the global registry
|   |-- image_schema.py              # common image output schema; tensors stay model-owned
|   |-- prompt_arguments.py          # shared prompt controls and prompt-file resolution
|   |-- seed_arguments.py            # one CLI spelling for scalar and ordered seed selections
|   `-- seed_config.py               # seed precedence and canonical serialization
|
|-- config/                          # generic TOML config primitives
|   |-- __init__.py                  # TOML composition, overrides, validation, dumping
|   |-- load.py                      # filename-aware TOML loading/composition
|   |-- merge.py                     # recursive table merge
|   |-- overrides.py                 # TOML-typed --set
|   |-- dump.py                      # deterministic round-trippable TOML
|   |-- registry.py                  # validated global model config contributions
|   |-- assembly.py                  # source preparation and precedence over a saved base
|   |-- inspection.py                # declaration inventory without resolving sources
|   |-- layer_rules.py               # selection semantics declared once for contributions
|   |-- paths.py                     # configuration addresses with explicit collection indices
|   |-- persistence.py               # saved configuration format: values plus presence
|   |-- provenance.py                # internal explanations of supplied and derived fields
|   |-- source_rules.py              # declared source decoding and reads before normalization
|   |-- stages.py                  # stage comparison policy for request projections
|   `-- transforms.py                # bounded declared transformations of one sparse layer
|
|-- debug/                           # opt-in reproducibility artifacts
|   |-- __init__.py                  # exported orchestration-owned artifact writers
|   |-- metadata.py                  # effective TOML, execution logs, run records
|   |-- sidecars.py                  # generic tensor-artifact safetensors sink
|   `-- memory.py                    # low-overhead allocator counters for durable run diagnostics
|
|-- io/                              # input / weight I/O
|   |-- __init__.py                  # I/O: image input, safetensors weight I/O
|   |-- atomic.py                    # atomic file publication primitives
|   |-- fingerprints.py              # streamed file SHA-256 receipt identity
|   |-- image.py                     # native ImageIO/CoreImage -> MLX tensor (image conditioning)
|   |-- geometry.py                  # shared source-aspect axes and aligned output sizing
|   |-- resampling.py                # native RGB resize and fitted-reference crop geometry
|   |-- reservation.py               # hidden exclusive target-reservation peers
|   |-- safetensors.py               # weight I/O, header validation, consumed-tensor fingerprints
|   |-- buffer.py                    # typed MLX construction from and exposure as Python buffers
|   |-- identity.py                  # stable source and artifact provenance identities
|   |-- output_bundle.py             # reservation and rollback for image output bundles
|   |-- output_config.py             # shared image/video path suffixes and image output selection
|   |-- output_paths.py              # generated output names, independent of model and format
|   `-- output_rules.py              # shared output-selection policies owned by fields
|
|-- kernels/                         # custom Metal kernels
|   |-- __init__.py                  # shared custom-kernel exports
|   |-- _typing.py                   # shared custom-kernel protocol
|   `-- fused_ops.py                 # reusable precision-sensitive MLX/Metal ops
|
|-- media/                           # model-neutral frame/signal boundary contracts
|   |-- __init__.py                  # public media vocabulary
|   |-- frames.py                    # closeable single-consumer frame stream
|   |-- signals.py                   # signal, color-plan, and delivery specifications
|   `-- hdr.py                       # MLX-native HDR transfers and linear-primary conversion
|
|-- modeling/                        # model-neutral graph and weight-loading helpers
|   |-- __init__.py                  # shared graph helper exports
|   |-- artifacts.py                 # local tokenizer and safetensors source inspection
|   |-- dtypes.py                    # shared floating compute dtype resolution
|   |-- evaluation.py                # typed MLX evaluation helpers (async dispatch)
|   |-- layers.py                    # empty Linear/Embedding shells for bounded loading
|   |-- qwen.py                      # shared Qwen3/Qwen3-VL language primitives
|   |-- loading.py                   # exact-target preflight, digest, and binding
|   |-- conversion.py                # lazy source conversion records, apart from storage
|   `-- memory.py                    # decode headroom from Metal's limit and reclaimable memory
|
|-- models/                          # per-model self-contained packages
|   |-- __init__.py                  # namespace for the per-model subpackages
|   |
|   |-- flux2/                       # Klein 9B resources and image recipes
|   |   |-- __init__.py              # Klein generation, conditioning stages, publication
|   |   |-- catalog.py               # base, distilled, and KV profiles
|   |   |-- arguments.py             # common image flags plus Klein controls
|   |   |-- command.py               # generation, restart, reporting, and publication
|   |   |-- config.py                # layered prompt/config and stage admission
|   |   |-- types.py                 # typed generation and restart controls
|   |   |-- config_schema.py         # model-owned configuration contribution
|   |   |-- settings.py              # local bundled and override source selection
|   |   |-- resources.py             # immutable source and tensor disposition inventory
|   |   |-- metadata.py              # strict supported graph configurations
|   |   |-- shapes.py                # full bundled source name/shape contracts
|   |   |-- source_loading.py        # exact component bindings; lazy transformer weights
|   |   |-- transformer.py           # shared double/single-stream Klein 9B graph
|   |   |-- kv.py                    # generation-owned reference K/V and masked attention
|   |   |-- sampling.py              # target-only Euler trajectories and external base CFG
|   |   |-- generation.py            # sequential component lifetimes and image result API
|   |   |-- pipelines/               # sequential complete public recipes
|   |   |   |-- __init__.py          # recipe and typed-output exports
|   |   |   |-- text_to_image.py     # prompt -> Qwen3 text -> Klein 9B -> VAE
|   |   |   `-- restart.py           # resume Klein from a saved stage or final latents
|   |   |-- runner.py                # stateless public host over resources and reporting
|   |   |-- text_conditioning.py     # complete text/negative/reference conditioning bundles
|   |   |-- restart.py               # saved stage and source validation
|   |   |-- output.py                # Klein artifacts over shared publication primitives
|   |   |-- text.py                  # Klein template, padding and THWL text positions
|   |   |-- text_encoder.py          # plain Qwen3 layers with hidden-state taps 9, 18 and 27
|   |   |-- vae.py                   # untiled spatial encoder/decoder and latent normalization
|   |   |-- image_processing.py      # output/reference geometry, native pixels and token IDs
|   |   |-- velocity.py              # velocity protocols and prepared velocity callables
|   |   |-- components.py            # public component ports and the default native provider
|   |   |-- geometry_rules.py        # Klein geometry choices needing no source-image read
|   |   |-- layout.py                # measured Klein storage layouts, apart from logical shapes
|   |   |-- output_policies.py       # Klein output availability for schemas and planning
|   |   |-- preparation.py           # Klein weight dtype and execution-layout preparation
|   |   |-- recipe_rules.py          # Klein recipe defaults and profile relationships
|   |   |-- request_constraints.py   # Klein bounds shared by fields and recipe helpers
|   |   |-- request_rules.py         # Klein request relationships and capability policy
|   |   |
|   |   `-- adapters/                # Klein aliases, module exclusions, and fresh-lease fusion
|   |       |-- __init__.py          # Klein adapter placement; parsing and fusion are shared
|   |       |-- keys.py              # BFL/Diffusers names, QKV splitting, final-AdaLN row order
|   |       |-- categories.py        # exclusions for whole stored projection modules
|   |       |-- loading.py           # structural admission, tensor receipts, and shared fusion
|   |       |-- config_schema.py     # Klein canonical adapter collection declaration
|   |       `-- inputs.py            # Klein adapter inputs and category ownership
|   |
|   |-- gmnet/                       # GMNet SDR-to-HDR still expansion (--model gmnet)
|   |   |-- __init__.py              # direct resources, recipe, runner, and output API
|   |   |-- types.py                 # immutable public still-expansion request
|   |   |-- resources.py             # source paths and policies, never live weights
|   |   |-- components.py            # generator port, lease, and native provider
|   |   |-- pipelines/               # public recipe implementations
|   |   |   |-- __init__.py          # recipe export
|   |   |   `-- sdr_to_hdr.py        # stateless expand_gmnet recipe
|   |   |-- runner.py                # generic recipe host + convenience expand method
|   |   |-- output.py                # typed all-or-nothing EXR/HEIC/gain-map terminal
|   |   |-- net.py                   # NHWC GMNet generator + checkpoint key map
|   |   |-- resample.py              # reference-matched bicubic/bilinear resamplers
|   |   |-- catalog.py               # variant facts: scale, peak, reference white, provenance
|   |   |-- settings.py              # GMNetSettings record (env/TOML/CLI/--set bridge)
|   |   |-- config.py                # typed GMNet TOML/CLI precedence and validation
|   |   |-- expand.py                # preprocess, inference, HDR reconstruction, gain-map sidecar
|   |   |-- convert.py               # GMNet key/variant/metadata conversion extension
|   |   |-- converter_cli.py         # `weights convert gmnet` argument set
|   |   |-- arguments.py             # complete --model gmnet inference parser
|   |   |-- command.py               # GMNet execution, reporting, and sidecars
|   |   |-- config_schema.py         # data-only GMNet configuration contribution
|   |   |-- output_rules.py          # output-format policy owned by GMNet fields
|   |   |
|   |   `-- weights/                 # converted weights (gitignored) + provenance docs
|   |
|   |-- krea2/                       # Krea 2 Raw/Turbo text-to-image generation
|   |   |-- __init__.py              # direct resources, recipe, runner, and stage APIs
|   |   |-- catalog.py               # Raw/Turbo source, architecture, and recipe facts
|   |   |-- arguments.py             # complete --model krea2 parser
|   |   |-- command.py               # preparation/generation execution and run sidecars
|   |   |-- config.py                # typed TOML/CLI precedence and validation
|   |   |-- config_schema.py         # Krea config contribution
|   |   |-- components.py            # transformer/decoder ports and native providers
|   |   |-- metadata.py              # strict cross-component config parsing
|   |   |-- output.py                # transactional image/tensor artifact publication
|   |   |-- runner.py                # stateless recipe host and convenience generation
|   |   |-- resources.py             # immutable source inventory
|   |   |-- sampling.py              # geometry, noise, schedule, CFG, Euler, pack/unpack
|   |   |-- settings.py              # baseline/component paths and compute dtype
|   |   |-- settings_rules.py        # pure final resource policies owned by Krea Settings fields
|   |   |-- text_conditioning.py     # sequential Qwen lifetime, products, provenance
|   |   |-- types.py                 # request plus generation/output config records
|   |   |-- weight_keys.py           # consumed source-to-target mappings
|   |   |-- source_loading.py        # source preflight and lazy binding
|   |   |-- edit_validation.py       # early admission for the one/two-reference recipe
|   |   |-- grounding.py             # image-grounded Qwen conditioning, bounded ownership
|   |   |-- image_processing.py      # native resampling and patch conditioning products
|   |   |-- output_policies.py       # Krea output availability for fields and restart
|   |   |-- preparation.py           # lazy Krea component transformations, all routes
|   |   |-- reference_regions.py     # readable geometric selections on a token grid
|   |   |-- request_constraints.py   # Krea request grammars shared by config and API
|   |   |-- request_rules.py         # coupled Krea request policies owned by fields
|   |   |-- request_sources.py       # saved-conditioning observations for a request
|   |   |-- resource_types.py        # immutable Krea 2 source, capability, and component records
|   |   |-- restart.py               # typed restart selection and prior-run artifacts
|   |   |-- sizing.py                # output sizing from the first source image's aspect
|   |   |
|   |   |-- adapters/                # Krea-specific adapter placement
|   |   |   |-- __init__.py          # Krea 2 adapter dialects and source-graph placement
|   |   |   |-- keys.py              # native/canonical/flattened aliases and exclusions
|   |   |   |-- turbo_profile.py     # four-step Turbo adapter selection and sampler defaults
|   |   |   |-- loading.py           # local files, PEFT scaling, pairing facts, shared fusion
|   |   |   |-- categories.py        # Krea 2 adapter exclusion categories
|   |   |   |-- config_schema.py     # Krea 2 canonical adapter collection declaration
|   |   |   |-- edit_profile.py      # community edit adapter selection
|   |   |   |-- inputs.py            # Krea 2 adapter inputs and category ownership
|   |   |   |-- profiles.py          # resolution order for the community adapter profiles
|   |   |   `-- sda_profile.py       # community SDA diversity-adapter selection
|   |   |
|   |   |-- pipelines/               # sequential complete public recipes
|   |   |   |-- __init__.py          # recipe and typed-output exports
|   |   |   |-- text_to_image.py     # Raw/Turbo text -> noise -> MMDiT -> VAE
|   |   |   |-- image_edit.py        # reference-conditioned editing on the Krea lifecycle
|   |   |   `-- restart.py           # resume Krea 2 from saved text conditioning or final latents
|   |   |
|   |   |-- text_encoder/            # text-only Qwen3-VL graph
|   |   |   |-- __init__.py          # Qwen graph and loading exports
|   |   |   |-- model.py             # GQA, MRoPE, SwiGLU, and twelve hidden taps
|   |   |   |-- loading.py           # exact 386-target preflight and binding
|   |   |   |-- multimodal.py        # image spans and three-axis positions
|   |   |   |-- vision.py            # adapter-independent Qwen3-VL still-image vision backbone
|   |   |   `-- vision_loading.py    # exact source binding for Qwen3-VL visual weights
|   |   |
|   |   |-- tokenizer/               # Krea-specific Qwen prompt layout
|   |   |   |-- __init__.py          # tokenizer exports
|   |   |   `-- template.py          # exact prefix, middle pad, suffix, masks, positions
|   |   |
|   |   |-- transformer/             # native Krea MMDiT graph and exact loading
|   |   |   |-- __init__.py          # native Krea 2 MMDiT graph and exact source-weight loader
|   |   |   |-- inputs.py            # compact text rows keeping suffixes and positions
|   |   |   |-- loading.py           # exact source and prepared-weight binding
|   |   |   |-- model.py             # allocation-light native MLX Krea 2 MMDiT graph
|   |   |   `-- reference.py         # compact target-to-reference attention bias
|   |   |
|   |   `-- vae/                     # native still decoder, tiling, and exact loading
|   |       |-- __init__.py          # native Krea 2 Qwen Image VAE still encoding and decoding
|   |       |-- encoder.py           # Qwen Image VAE encoding of the first still frame
|   |       |-- encoder_loading.py   # exact source binding for the first-frame encoder
|   |       |-- loading.py           # exact source binding for the Krea 2 Qwen Image decoder
|   |       |-- model.py             # allocation-light native Qwen Image still decoder
|   |       `-- tiling.py            # Qwen Image VAE tiling policy and overlap blending
|   |
|   `-- ltx2/                        # LTX-2.3/LTX-2.5 distilled pipeline
|       |-- __init__.py              # LTX-2 public API
|       |-- sigmas.py                # LTX-2 sigma constants + scheduler params
|       |-- signals.py               # LTX-2 concrete SDR and HDR signal constants
|       |-- arguments.py             # complete --model ltx2 parser
|       |-- command.py               # generation/restart execution and reporting
|       |-- config.py                # typed TOML/CLI precedence and output reservation
|       |-- config_schema.py         # LTX config contribution
|       |-- config_groups.py         # LTX semantic config-group vocabulary
|       |-- output_config.py         # LTX-2 video-delivery settings
|       |-- output.py                # VideoToolbox/HDR generation sink adapter
|       |-- restart.py               # prior-run artifacts and restart identity receipts
|       |-- generation.py            # closeable generated video/audio product
|       |-- transformer_config.py    # neutral transformer constructor record
|       |-- run_logging.py           # request, resources, and sampling summaries
|       |-- noising.py               # LTX clean-latent two-lerp Gaussian injection
|       |-- transition_noise.py      # ordered ancestral-noise stream for one stage
|       |-- patchifier.py            # audio/video tokens, grid bounds, pixel coordinates
|       |-- components.py            # public component ports, providers, and leases
|       |-- duration.py              # optional natural-duration head + VAE-grid snapping
|       |-- generated_keyframes.py   # stage-1 generated-slot layout, mask, and cleanup
|       |-- artifacts.py             # LTX-2 artifact schemas, builders, and sidecar paths
|       |-- output_policies.py       # output availability derived from the stage model
|       |-- resources.py             # immutable checkpoint inventory
|       |-- settings.py              # LTX-2 checkpoint and transformer policy
|       |-- text_conditioning.py     # encoder-neutral stage, product, provenance
|       |-- denoise.py               # pure: denoise_step, denoise_loop
|       |-- encode.py                # pure: encode_text, encode_image
|       |-- decode.py                # audio decode + lazy owned video-frame stream
|       |-- state.py                 # latent-state construction + conditioning setup
|       |-- types.py                 # request, LTX latent state, and shape contracts
|       |-- runner.py                # generic recipe host over resources + stateless ports
|       |-- compatibility.py         # central, packaging-neutral LTX-2 compatibility inspection
|       |-- hdr_profile.py           # explicit HDR producer facts, no checkpoint allowlists
|       |-- hdr_reference.py         # SDR reference-video preprocessing for the HDR recipe
|       |-- frame_position.py        # first/last/integer frame-position grammar and constraint
|       |-- image_arguments.py       # image-condition members, --image shorthand, collection
|       |-- keyframe_positions.py    # generated-keyframe positions for admission and run
|       |-- layer_rules.py           # LTX rules contributed through the shared runner
|       |-- metadata.py              # header-only component config and compatibility
|       |-- output_rules.py          # LTX HDR output restrictions for CLI and planning
|       |-- precision.py             # explicit dtype boundaries for native LTX-2 pipelines
|       |-- preparation.py           # lazy LTX transformer conversion and weight binding
|       |-- request_rules.py         # coupled LTX request policies owned by fields
|       |-- restart_rules.py         # LTX stage model and the restart record rules that read it
|       |-- settings_rules.py        # pure final resource policies owned by LTX Settings fields
|       |-- source_loading.py        # reloadable lazy transformer source preparation
|       |
|       |-- adapters/                # LTX-specific adapter placement
|       |   |-- __init__.py          # LTX adapter aliases and application to transformer weights
|       |   |-- categories.py        # public target-knockout vocabulary
|       |   |-- keys.py              # source aliases, graph targets, and knockout categories
|       |   |-- receipt.py           # adapter application receipt
|       |   |-- selection.py         # per-stage strengths and resolved LoRA profiles
|       |   |-- loading.py           # direct or PEFT-scaled application and execution layouts
|       |   |-- config_schema.py     # LTX-2 adapter collection and stage shorthand
|       |   `-- inputs.py            # LTX-2 adapter inputs with optional stage strengths
|       |
|       |-- audio_vae/               # audio <-> latent -> mel -> waveform
|       |   |-- __init__.py          # LTX audio VAE and BWE vocoder
|       |   |-- config.py            # validated checkpoint-driven VAE architecture
|       |   |-- blocks.py            # causal Conv2d, residual, norm, resampling blocks
|       |   |-- encoder.py           # stereo log-mel -> normalized latent
|       |   |-- decoder.py           # latent -> mel spectrogram
|       |   |-- loading.py           # complete encoder/decoder family loader
|       |   |-- vocoder.py           # FP32 BigVGAN + bandwidth-extension composition
|       |   |-- vocoder_layers.py    # 1D convolution, SnakeBeta, AMP, and resampling layers
|       |   |-- vocoder_loading.py   # complete BWE-vocoder family loader
|       |   `-- vocoder_stft.py      # checkpoint-backed causal STFT and mel projection
|       |
|       |-- conditioning/            # image conditioning (first-frame, keyframes)
|       |   |-- __init__.py          # first-frame replacement and keyframe insertion
|       |   |-- item.py              # encoded-condition structural protocol
|       |   |-- keyframe.py          # insert image at arbitrary frame
|       |   |-- latent.py            # latent-replacement at frame 0
|       |   |-- preparation.py       # raw-source -> encoded-condition stage
|       |   |-- tools.py             # shape-bound patch/mask/position helpers
|       |   `-- reference.py         # clean full-video tokens for IC-LoRA conditioning
|       |
|       |-- kernels/                 # LTX attention Metal kernels
|       |   |-- __init__.py          # LTX-2-specific Metal kernels
|       |   |-- dense_attention.py   # FP16 dense attention and its call gate
|       |   |-- sol_attention.py     # Sol-Attn sparse video self-attention
|       |   |
|       |   `-- metal/               # shared FP16 primitives and kernel bodies
|       |
|       |-- pipelines/               # composition entry points
|       |   |-- __init__.py          # generation pipelines for the LTX-2 model
|       |   |-- distilled.py         # public distilled recipe and stage products
|       |   `-- restart.py           # stage-2/direct-decode recipe shared by API and CLI
|       |
|       |-- text_encoder/            # tokenizer + Gemma 3/4 12B + AV connector
|       |   |-- __init__.py          # Gemma 3/4 prompt encoding and AV connectors
|       |   |-- tokenizer_derivation.py # in-memory SentencePiece derivation
|       |   |-- tokenizer.py         # metadata-driven BOS, padding, mask, and decode policy
|       |   |-- gemma3.py            # Gemma 3 architecture adapted from mlx-lm
|       |   |-- gemma3_loading.py    # complete sharded/single-file Gemma 3 target binder
|       |   |-- gemma4.py            # LTX-tuned Gemma 4 backbone and parity boundaries
|       |   |-- gemma4_loading.py    # complete packaged Gemma 4 target binder
|       |   |-- features.py          # per-token RMS aggregation and AV projections
|       |   |-- connector.py         # independent audio/video connector blocks
|       |   |-- loading.py           # monolithic/split projection and connector binder
|       |   |-- encoder.py           # AV connector model and checkpoint-derived config
|       |   `-- prompt.py            # generation-selected sequential text orchestration
|       |
|       |-- transformer/             # 22B DiT (velocity prediction)
|       |   |-- __init__.py          # LTX-2 diffusion transformer public API
|       |   |-- attention.py         # MHA + 3D / 1D RoPE wiring
|       |   |-- rope.py              # 3D RoPE for video, 1D for audio
|       |   |-- feed_forward.py      # GELU MLP in the transformer's dtype (FP16 default)
|       |   |-- timestep.py          # timestep MLP + checkpoint-shaped AdaLN projections
|       |   |-- preprocessing.py     # per-modality masks, RoPE, and A/V timestep preparation
|       |   |-- transformer.py       # one joint A/V DiT block
|       |   |-- wrappers.py          # Modality (timesteps + sigma), X0Model (velocity -> x0)
|       |   |-- model.py             # 48-block LTXAVModel orchestration + block streaming
|       |   |-- execution.py         # modality branches and the weight groups they use
|       |   `-- graph.py             # parameter-graph and checkpoint-key contracts
|       |
|       |-- upscaler/                # public spatial/temporal latent upscalers
|       |   |-- __init__.py          # spatial recipe upscaling and temporal x2 support
|       |   |-- ports.py             # neutral upscaler and latent-statistics protocols
|       |   |-- spatial.py           # normalized-latent wrapper + raw 2x spatial model
|       |   `-- temporal.py          # temporal x2 component + VAE-statistics stage
|       |
|       |-- video_vae/               # metadata-selected native MLX video VAE
|       |   |-- __init__.py          # native MLX video VAE graphs for LTX-2 checkpoints
|       |   |-- blocks.py            # Conv3d, residual, PixelNorm, BFHWC packing
|       |   |-- config.py            # validated checkpoint-driven architecture
|       |   |-- ops.py               # BCFHW patch packing + latent statistics
|       |   |-- encoder.py           # pixels -> normalized latent
|       |   |-- decoder.py           # convolutional normalized latent -> RGB
|       |   |-- diffusion_decoder.py # bounded NA diffusion latent -> RGB
|       |   |-- loading.py           # complete encoder/decoder checkpoint loader
|       |   `-- tiling.py            # memory-bounded streaming decode + blending
|       |
|       `-- weights/                 # source conversion and model binding
|           |-- __init__.py          # source mappings, policies, and weight layouts
|           |-- bindings.py          # consumed source tensors and FP8 scale companions
|           |-- dtypes.py            # transformer weight dtype names and normalization
|           |-- layout.py            # prepared Conv and transformer linear layouts
|           |-- loading.py           # bind and stream prepared transformer weights
|           |-- policy.py            # canonical transformer weight layout policy
|           |-- quantization.py      # install packed quantized transformer linear modules
|           |-- schema.py            # weight layout tags and quantization modes
|           |-- streaming.py         # block streaming over reloadable sources
|           |-- tensors.py           # checkpoint dtype handling and quantization routing
|           |-- validation.py        # exact validation of owned weight parameter graphs
|           |-- residency.py         # persistent block tensors, apart from ring residency
|           |
|           `-- keys/                # checkpoint mappings per component
|               |-- __init__.py      # LTX checkpoint key conversion and family routing
|               |-- audio_vae.py     # audio VAE and vocoder family routing
|               |-- text_encoder.py  # text-projection and AV connector checkpoint family routing
|               |-- transformer.py   # transformer checkpoint-to-MLX key conversion
|               `-- video_vae.py     # video VAE routing and structure-driven lookup
|
|-- profiling/                       # opt-in Instruments instrumentation
|   |-- __init__.py                  # opt-in Instruments profiling for orchestration boundaries
|   |-- signpost.py                  # Apple os_signpost wrapper
|   `-- _signpost.c                  # native signpost shim source
|
|-- samplers/                        # MODEL-AGNOSTIC sampling math
|   |-- __init__.py                  # model-agnostic sampling primitives
|   |-- noise.py                     # centralized MLX / Torch-MPS normal streams
|   |-- steps.py                     # Euler steps
|   |-- config.py                    # field-owned input admission for explicit denoise schedules
|   `-- sigmas.py                    # model-neutral sigma schedules and validated user schedules
|
|-- tokenizers/                      # shared native Qwen ByteLevel-BPE tables and runtime
|   |-- __init__.py                  # native tokenizer engines for model prompt policies
|   |-- byte_bpe.py                  # NFC, byte mapping, BPE, and decode
|   |-- loading.py                   # source-derived semantic tokenizer tables
|   `-- unicode_scan.py              # Qwen Unicode split scanner
|
|-- ui/                              # centralized terminal output
|   |-- __init__.py                  # Rich logging, machine output, and progress bars
|   |-- console.py                   # shared Rich stderr console
|   |-- logging.py                   # human logging + isolated machine stdout
|   `-- bars.py                      # progress bars + RichReporter adapter
|
|-- videotoolbox/                    # FIRST-CLASS Apple VideoToolbox subsystem
|   |-- __init__.py                  # VideoToolbox-backed video post-processing
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
|   |-- comparison.py                # side-by-side A/B composite
|   |-- reader.py                    # native SDR decode for model conditioning inputs
|   `-- yuv.py                       # MLX RGB to 10-bit BT.709 video-range 4:2:2 conversion
|
`-- weights/                         # generic safe checkpoint-conversion substrate
    |-- __init__.py                  # direct public converter API
    |-- cli.py                       # `weights convert` + model-specific dispatch
    |-- host.py                      # shared settings/progress/allocator converter host
    |-- output.py                    # no-clobber verified publication transaction
    |-- pickle_scan.py               # exact symbolic pickle-global scanner
    |-- torch_checkpoint.py          # bounded restricted tensor-only reader
    `-- convert.py                   # value/layout-preserving generic conversion
```

## Layer Responsibilities

The package is organized as concentric layers. Higher layers compose
lower layers; lower layers never reach into higher ones.

### Layer 1 - Pure math and primitives (model-agnostic)

| Module | Owns |
|---|---|
| `kinomlx/samplers/` | Euler steps, centralized normal-noise streams, and model-neutral sigma schedules |
| `kinomlx/adapters/` | Shared selections, low-rank parsing, static adapter algebra, bounded fusion, and tensor receipts |
| `kinomlx/kernels/` | Reusable custom Metal ops with more than one plausible model consumer |
| `kinomlx/audio/` | Audio signal-processing helpers (onset detection, ...) |
| `kinomlx/media/` | Immutable signal/delivery vocabulary and closeable frame ownership |
| `kinomlx/tokenizers/` | Native Qwen ByteLevel-BPE tables, Unicode splitting, and decode for model prompt policies |
| `kinomlx/components.py` | Model-neutral bounded `ComponentLease` ownership for heavyweight runtime components |
| `kinomlx/seeds.py` | Ordered seed selection shared by every generation family |
| `kinomlx/errors.py`, `kinomlx/reporting.py` | Typed operational failures and the host-neutral `Reporter` protocol with its test sinks |
| `kinomlx/_json.py`, `kinomlx/_mlx_nn.py`, `kinomlx/_tensor_tree.py` | Typed primitives: recursive JSON values, the exact installed `mlx.nn` objects (with `_mlx_nn.pyi`), and `Module.update` trees |

These building blocks do not discover or construct model families. Adapter
fusion updates only the explicitly supplied weights; the math and descriptors
can be reused by other diffusion models.

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
| `kinomlx/modeling/` | Model-neutral layer shells, shared Qwen primitives, dtype/evaluation helpers, and exact checkpoint binding |
| `kinomlx/huggingface.py` | Read-only discovery of already-local Hugging Face Hub snapshots |
| `kinomlx/recipe_host.py` | Common public recipe execution boundary, independent of model ports |
| `kinomlx/seed_execution.py` | Request-scoped model ownership and sequential execution at image stages |
| `kinomlx/reporter_stack.py` | The reporter stack every command and converter builds around its work |

Touches the outside world. Still model-agnostic.

### Layer 3 - Configuration

| Module | Owns |
|---|---|
| `kinomlx/settings.py` | Infrastructure settings plus the sole environment/CLI parsing mechanism |
| `kinomlx/config/` | TOML load/merge/typed overrides/schema/dump primitives |
| `kinomlx/types.py` | Model-neutral shapes and compute-policy literals |
| `kinomlx/config_*.py` | Typed record construction and presence, CLI bindings, local constraints, field rules, and transport surface policies |
| `kinomlx/model_catalog.py` | Model catalog: names, help, and the default model |
| `kinomlx/model_registry.py` | Validated global registry of per-model configuration contributions |
| `kinomlx/invocation_controls.py`, `kinomlx/api_boundaries.py` | Bootstrap and utility control declarations, plus typed ownership of public API parameters |
| `kinomlx/memory_policy.py` | Invocation-owned MLX allocator and wired-memory policy |
| `kinomlx/prompts.py`, `kinomlx/request_rules.py` | Shared generation-request contracts: the explicit empty-user-prompt rule and coupled field policies |
| `kinomlx/image_restart_config.py`, `kinomlx/restart_selection.py`, `kinomlx/restart_receipts.py` | Shared still-image restart inputs, multi-image source selection, and observational artifact receipts |

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
| `kinomlx/__init__.py` | The shared-infrastructure import surface for library callers; model APIs stay under `kinomlx.models` |

## Model package layout (`models/<name>/`)

The model packages expose a common public lifecycle even when their internal
network topologies and modalities differ:

| File / dir | Role |
|---|---|
| `__init__.py` | Directly exports the model's runner, recipes, requests, and resources. |
| `types.py` | Immutable typed request and model-owned product contracts. |
| `resources.py` | Prepared immutable paths/capabilities/policies, never live model instances. |
| `components.py` | Injectable ports and bounded `ComponentLease` providers. |
| `pipeline.py` or `pipelines/<name>.py` | Stateless top-level recipe composition. |
| `runner.py` | Generic `run(recipe, request)` host retaining resources and stateless ports. |
| `settings.py` | Model-specific checkpoint, weight-layout, and execution policy. |
| `arguments.py` | Complete model-specific parser built from the model's config schema. |
| `command.py` | Model execution, model-owned output planning, and reporting. |
| `config_schema.py` | Data-only contribution to the global config registry. |
| `output.py`, `artifacts.py` | Optional model-owned terminal or sidecar contracts. |
| `convert.py` | Optional exact model checkpoint validation layered on `kinomlx.weights`. |

Diffusion model packages additionally contain their samplers, transformers, VAEs,
text encoders, and adapter support. GMNet has a smaller convolutional still-image
implementation without those diffusion-specific components.

Every model has one entry in `kinomlx/model_catalog.py`, one complete command in the
dispatch table of `kinomlx/cli/_registry.py`, and one configuration contribution in
`kinomlx/model_registry.py`, all imported statically.
`--model <name>` routes the whole invocation to the model's own `command.py`,
whose `arguments.py` defines its complete inference parser while reusing the
shared flag conventions (`--output`, `--output-dir`, `--config`, `--set`) and
resolving its settings record with the standard defaults < env < TOML
`[model_settings]` < CLI < `--set` precedence. The root CLI does not own a
privileged generation grammar or special-case other modalities.
The bootstrap selection itself follows config < `--model` < `--set model=...`,
so config-only model selection reaches the same parser. Non-inference utilities
do not pretend to be model actions: `kinomlx weights convert` is generic and
dispatches `kinomlx weights convert gmnet` to the model-owned extension.

Every selectable model also contributes one complete `ModelConfigSpec` through
`config_schema.py`. The host collects those contributions in one validated
global registry. Shared infrastructure settings are one immutable table
contribution reused by every model. Dataclass coverage, parser destinations and
choices, accepted TOML tables, environment metadata, starter templates, and
full/sparse serialization all resolve through this registry; model packages do
not require edits to a central field union.

## API Surface

Two layers, both first-class.

**High-level** - used by the CLI and most user scripts:

```python
from kinomlx import Settings
from kinomlx.models.ltx2 import (
    LTX2Request,
    LTX2Runner,
    LTX2Settings,
    generate_ltx2,
)

runner = LTX2Runner(
    LTX2Settings.from_env(),
    infrastructure=Settings.from_env(),
)
with runner.run(generate_ltx2, LTX2Request(prompt="...")) as output:
    for frame in output.frames:
        consume(frame)
```

Saved-stage restart is another public recipe, not a branch inside the generic
runner or the ordinary distilled recipe:

```python
from pathlib import Path

from kinomlx.models.ltx2 import LTX2Request, LTX2Restart

request = LTX2Request(width=768, height=448, frames=121)
selection = LTX2Restart.decode(Path("outputs/source.safetensors"))
with runner.restart(request, selection) as output:
    consume(output)
```

`LTX2Restart.stage_2(...)` selects the shared upscaler, stage-2 denoise,
decode, and terminal path from a stage-1 latent plus text-conditioning product.
The CLI resolves its run manifest into this same typed selection and invokes
the same `restart_ltx2()` recipe. VSR, frame-rate conversion, and media
encoding remain terminal responsibilities downstream of either recipe.

`LTX2Runner` prepares one immutable resource inventory and retains only
stateless component, text-conditioner, reporter, and artifact ports. The
generic host contains no distilled branch: callers supply the public recipe
and its typed request. Each recipe releases live components at bounded stage
boundaries. Video decoder ownership transfers to the lazy frame stream and
ends on exhaustion, early close, or output failure.

**Direct recipe** - the shortest form for a standalone generation:

```python
from kinomlx import Settings
from kinomlx.models.ltx2 import LTX2Settings
from kinomlx.models.ltx2.pipelines.distilled import generate_ltx2
from kinomlx.models.ltx2.resources import prepare_resources
from kinomlx.models.ltx2.types import LTX2Request

resources = prepare_resources(
    LTX2Settings.from_env(),
    infrastructure=Settings.from_env(),
)
request = LTX2Request(prompt="...")
with generate_ltx2(request, resources) as output:
    for frame in output.frames:
        consume(frame)
```

`LTX2Resources` contains paths and policies, never live model instances.
`generate_ltx2` leases native components at recipe-owned lexical
boundaries. Its returned frame stream is intentionally caller-owned. The
executable [`examples/ltx2_distilled.py`](../examples/ltx2_distilled.py)
expands the same recipe into every public stage and makes the complete
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

from kinomlx.models.gmnet import (
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
recipe convenience, while injected `GMNetComponentsProvider` make tests and alternate
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

## Implementation notes

### Component sources and lifetime

Resource inventories describe original source files, consumed tensor keys, and
execution policies. Model providers bind lazy conversion expressions into phase-owned
components. Text encoders, transformers, and decoders are released at their respective
phase boundaries. LTX block streaming binds selected block groups through a reloadable
source provider. See [model loading](WEIGHTS.md) for the user-visible memory and timing effects.

### Checkpoint and adapter mappings

Checkpoint mappings translate source keys to MLX module weights. LTX mapping tables
are under `models/ltx2/weights/keys/`; adapter mappings are in each model's
`adapters/keys.py`. Video and audio VAE loaders also use module-tree traversal where
source-to-destination structure supplies the mapping. Adapter parsing and fusion
math are shared in `kinomlx/adapters/`, while model packages resolve their own targets.

### Precision

LTX transformers default to FP16 and Krea and Klein transformers to BF16; either
is selectable. Both formats use two bytes per weight. Precision-sensitive calculations use wider
arithmetic where required by the model. For example, LTX uses FP32 update math and
honors the checkpoint's frequency-grid precision before casting rotary tables to
the activation dtype. FP8 E4M3 source decoding and selected layout transforms are
attached lazily during weight loading.

The selected dtype affects execution and numerical rounding; it does not rewrite
original source files. See [precision choices](PERFORMANCE.md#precision) and the
model guides for supported configurations and output implications.

### Reporting and terminal output

Runtime modules use stdlib logging and `kinomlx.reporting.Reporter` for progress.
The CLI connects these to `RichReporter` and a shared stderr console. JSON and
`--print-config` use separate stdout output; `--save-config` uses the same TOML
serializer. Quiet mode suppresses live rows while retaining phase accounting.
This separation lets applications consume structured output without parsing human
progress messages.

### On-demand native profiling shim

When signpost profiling is explicitly enabled, KinoMLX compiles
`profiling/_signpost.c` with the system Clang and caches a source-and-architecture
keyed dylib under the configured application cache. Ordinary generation performs
no native build, and a profiling-shim build failure disables signposts without
aborting generation. No architecture-specific signpost library is checked in.

### MLX and native media

Model execution uses MLX and native Apple media frameworks. LTX derives its
SentencePiece tokenizer in memory using the protobuf model schema; Krea and Klein
derive ByteLevel-BPE tables from original JSON assets. Source header validation
precedes model binding. Local Hub resolution uses downloaded snapshots, and the
runtime uses its own safetensors header reader with MLX tensor loading.

## Cross-References

- `kinomlx.modeling`, `kinomlx.huggingface`, and
  `kinomlx.io.safetensors.inspect_safetensors` are shared directly by LTX-2,
  Krea 2, and Klein. Model packages do not re-export those model-neutral helpers
  through compatibility modules.
- `resources.py` inspects source artifacts and `components.py` binds or fuses
  them into bounded leases. Recipes consume only those public providers and
  stage products; model code owns their lifetimes.
- `models/ltx2/adapters/loading.py` mediates between the shared `kinomlx/adapters/`
  fusion math and the LTX key/category rules in `adapters/keys.py`.
  Multiple adapters retain independent scalar and per-stage totals, and each
  adapter can exclude overlapping branch, module, projection, or control-path
  categories. The distilled recipe resolves two complete ordered profiles. It
  reuses one transformer only when those profiles are identical; otherwise it
  closes stage 1 and loads a pristine transformer from source for stage 2.
- `kinomlx.adapters.selection` owns the resolved adapter selection record that
  LTX-2, Krea 2, and Klein share; `kinomlx.adapters.inputs` resolves the CLI and
  request controls into it.
  Krea's `adapters/keys.py` owns its graph aliases and exclusions; its loader
  delegates equations and physical-layout fusion to `kinomlx.adapters`.
  Adapter math is shared while model-specific target selection remains local.
  Fusion stays lazy until denoising. Active adapter-modified floating weights receive
  the optional first-use check after their first step.
- Klein's image API accepts shared `AdapterSelection` records. Its adapter loader
  validates source structure before text encoding and fuses into a fresh lazy
  transformer before denoising or reference K/V extraction. It reuses the static
  adapter catalog and FP32 accumulation, including LoKr and base-dependent formats.
  BFL source row transformations happen before installation in ordinary or
  pretransposed storage. Exclusions select whole stored projection modules:
  a single-stream QKV/MLP projection belongs to attention, feed-forward, and
  Q/K/V categories together. Krea and Klein share local source inspection,
  Diffusers embedded configuration, PEFT companion configuration, and scratch loading.
- `pipelines/distilled.py` composes `denoise.py` + `encode.py` +
  `decode.py` + `state.py`. Its final video product stays lazy through
  `output.py` and `videotoolbox/`; other pipelines follow the same
  ownership rule.
- `debug/sidecars.py` owns optional text, VAE-encoded media-conditioning, and
  generated-latent artifacts.
- `models/ltx2/output.py` resolves LTX-2 video settings into an explicit color plan and
  delegates final video, mandatory HDR EXR, optional PQ HEIC, and optional WAV
  writing to `output.py`.

Common image configuration lives in `cli/image_schema.py`; each model contributes only admitted
schema fields and its own features. `io/output_config.py` owns `ImageOutputConfig` for both
image models and the shared output path-suffix constraints, including the `.mp4` one the LTX
video path uses. `io/output_bundle.py` owns typed reservations, source-alias protection, and
rollback publication. Model output modules retain their tensor schemas and pixel-range handling.

The public model runners share `kinomlx/recipe_host.py` for recipe dispatch, typed-result
validation, and operational error translation without retaining failed model frames.
Klein exposes `Flux2Runner`, `Flux2Request`, `Flux2Restart`, and `Flux2GenerationOutput`;
its `pipelines/` package owns typed generation/restart recipes and `generation.py` retains the
lower-level single-image implementation. The CLI uses those same runner recipes.
