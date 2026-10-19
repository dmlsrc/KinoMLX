# Changelog

All notable user-facing changes to KinoMLX are recorded here. The format follows
Keep a Changelog, and versions follow semantic versioning.

## 0.0.3 - 2026-10-18

This release is mostly about LTX-2. Its transformer and both video VAE decoders now default to
FP16, and new Metal kernels speed up attention, the large linear products, rotary embeddings, the
vocoder, adapter fusion, and both decoders; [Performance](#performance) has the measurements.
Image conditioning becomes an ordered collection with first, last, and interior keyframes and a
strength per stage; LTX-2.5 samples both stages ancestrally, as Lightricks' ltx-pipelines 1.4.0
does; and sparse video attention and diffusion VAE sharpening are new opt-in controls.
`--dry-run` now covers all four families. KinoMLX now requires Python 3.15 and macOS 27, run
records written by 0.0.2 cannot be restarted, several settings and Python names changed, and
LTX-2 outputs at a given seed differ from 0.0.2's; read Breaking changes first.

### Breaking changes

- KinoMLX now requires Python 3.15 and macOS 27, MLX 0.32.4 or later, and protobuf 7.36.2 or
  later. MLX releases before 0.32.3 hold Python's GIL while they wait for the GPU, which can
  deadlock KinoMLX's half-float frame writer.
- Run records written by 0.0.2 cannot be restarted by this version. They record
  `restart.phase`, which is now `restart.stage`, and an LTX-2 record also carries the
  `generate.image` table that `generate.images` replaced; the refusal names the field. Klein
  conditioning artifacts saved by 0.0.2 cannot be read either, because the artifact now carries
  `schema_version` and `artifact` keys as Krea 2's and LTX-2's do. Generate again with this
  version for a restartable run.
- LTX-2.5 now samples both distilled stages with ancestral RF Euler, as Lightricks'
  ltx-pipelines 1.4.0 and its ComfyUI workflows do; 0.0.2 ran stage 2 deterministically, so an
  LTX-2.5 seed gives a different video. Stage 2 draws its renoising from its own stream, seeded
  at the run's seed plus 20,000, beside stage 1's at the seed plus 10,000. LTX-2.3 stays
  deterministic in both stages.
- LTX-2's `[generate].sampler`, which set stage 1 only, is replaced by `stage_1_sampler` and
  `stage_2_sampler`, each `auto`, `deterministic` or `ancestral`, on either generation. The
  flags are `--stage1-sampler` and `--stage2-sampler` (also `--sampler-stage-1` and
  `--sampler-stage-2`), and the environment variables `KINO_LTX2_STAGE_1_SAMPLER` and
  `KINO_LTX2_STAGE_2_SAMPLER`; `--sampler` and `KINO_LTX2_SAMPLER` are gone.
  `--stage2-sampler deterministic` gives back 0.0.2's LTX-2.5 stage 2, and a stage-2 restart may
  change `stage_2_sampler`. The run record's `sampler_override`, `sampler_policy`,
  `ancestral_noise_seed` and `ancestral_noise_state` are replaced by one `samplers` table with
  each stage's requested and effective sampler, noise seed and stream state.
- LTX-2.5 generated keyframe slots (`--generated-keyframes`) now stay in the sequence through
  stage 2, seeded with the stage-1 slots upsampled with the video, as ltx-pipelines 1.4.0 runs
  them; 0.0.2 dropped them after stage 1. A seed that generates keyframes therefore gives a
  different video, and stage 2 attends to one more latent frame of tokens per slot. The saved
  stage latents carry the slot planes as `generated_keyframes`.
- LTX-2 image conditioning is an ordered collection; Added describes what it can do. The
  `[generate.image]` table is now an `images` array of records under `[generate]`, and `--image`
  is repeatable, one condition per token, in the grammar `PATH[@POSITION][:STRENGTH]` or
  `PATH[@POSITION]:STAGE1:STAGE2`. `--image-frame-index` and `--image-strength` extend the record
  the preceding `--image` started instead of setting one run-level table. `KINO_LTX2_IMAGE`,
  `KINO_LTX2_IMAGE_FRAME_INDEX`, and `KINO_LTX2_IMAGE_STRENGTH` are replaced by one
  `KINO_LTX2_IMAGES` taking the same array of records as `KINO_LTX2_ADAPTERS`.
  `--set generate.image.strength=` becomes `--set 'generate.images[0].strength='`, and
  `DistilledRequest(image=...)` becomes `LTX2Request(images=(...,))`.
- An LTX-2 image's `frame_index` is now a string holding `first`, `last`, or a non-negative
  integer, and defaults to `"first"`, the frame `0` always named. An integer written in TOML,
  `--set`, an environment array or the API is canonicalized to its decimal string, so
  `frame_index = 120` and `frame_index = "120"` are one value and reading the field back yields
  `"120"`; code comparing `image.frame_index` to an integer needs to compare strings or resolve
  the position. The per-stage media-conditioning sidecar's `condition_{index}_frame_index` stays
  the resolved absolute integer.
- An LTX-2 image's `strength` now defaults to `1.0`, where 0.0.2's `[generate.image].strength`
  defaulted to `0.95`, so `--image` at frame 0 replaces that frame with the encoded input rather
  than with ninety-five percent of one, as `[generate.hdr_reference].strength` already did. Every
  default image-conditioned output moves; `--image-strength 0.95` restores the old value.
- An LTX-2 stage-2 restart now holds each image condition's path, frame and effective stage-1
  strength, in author order, and admits only stage-2 strength changes, the rule
  `[generate].adapters` already followed. 0.0.2 let a stage-2 restart change the image's path,
  frame or strength, which the stage-1 latent being restarted from contradicts. A decode restart
  still admits no image change.
- LTX-2 now applies the scale a PEFT export declares. When an adapter carries PEFT
  configuration, an `adapter_config.json` beside `adapter_model.safetensors` or Diffusers
  `lora_adapter_metadata` embedded in the file, each target's update is
  `strength * lora_alpha / r * (B @ A)`, with `lora_alpha / sqrt(r)` under `use_rslora` and
  per-module `alpha_pattern` and `rank_pattern` entries first. That is what PEFT merges, so
  strength 1 now reproduces the author's merge: an export with `r = 32` and `lora_alpha = 64`
  needed strength 2 for that in 0.0.2, and strength 2 now doubles it. Every other LTX-2 adapter
  keeps direct strength; saved `.alpha` tensors and loose `lora_alpha` metadata stay unused, as
  in Lightricks' loader.
- LTX-2's BF16 STEEL attention backend is removed, with its `[model_settings]` settings
  `steel_attention`, `steel_attention_d64` and `steel_attention_probe`, their flags
  (`--steel-attention`, `--steel-attention-d64`, `--steel-attention-probe` and their `--no-`
  forms), their `KINO_LTX2_STEEL_ATTENTION`, `KINO_LTX2_STEEL_ATTENTION_D64` and
  `KINO_LTX2_STEEL_ATTENTION_PROBE` environment variables, and the `steel_attention` run-record
  diagnostic. The new dense attention kernel (see Changed) runs every call STEEL ran by default,
  and faster. A configuration that still sets one of them is refused with the field named;
  effective configurations saved by 0.0.2 runs list all three, so delete those lines before
  reusing one. With `--no-dense-attention-kernel` every attention call runs on MLX SDPA, which in
  BF16 is slower than STEEL was; [Performance](#performance) compares the three.
- LTX-2's `--gemma-path` is removed, with `[model_settings].gemma_path` and
  `KINO_LTX2_GEMMA_PATH`. `--text-encoder-path` (`text_encoder_path`) now names LTX-2.3's Gemma 3
  folder as well as LTX-2.5's prepared text encoder; the two options selected the same component
  and could not be combined. A configuration that sets `gemma_path` is refused with the field
  named; rename it to `text_encoder_path`. In Python, `encode_text` and `encode_prompt` take
  `text_encoder_path=` instead of `gemma_path=`, and `LTX2Resources.gemma_path` is now
  `text_encoder_path`. An LTX-2.3 run with an explicit text encoder logs it instead of the
  `--gemma-variant` it did not use, and LTX-2.5 refuses a folder with a message asking for the
  prepared text-encoder file, as LTX-2.3 asks for a Gemma 3 folder instead of a single file.
- LTX-2's `--compile-block-groups` is now `--compile-block-group-size`, with
  `[model_settings].compile_block_group_size` and `KINO_LTX2_COMPILE_BLOCK_GROUP_SIZE`. The value
  has always been the number of blocks in each compiled group, not a number of groups, and the
  name now reads like the streamed `--transformer-compile-group-size`. A configuration that sets
  `compile_block_groups` is refused with the field named; rename it. `LTX2Settings` and the
  transformer's attribute follow.
- Krea 2's and Klein's `--seed-mode station` is now `--seed-mode stage`, and
  `seed_mode = "stage"` in configuration files and API requests. The value names the lifetime
  policy of loading one stage's components at a time across seeds, and stage is now the one word
  for a step of a recipe everywhere, including every error message that said station.
- `[restart].phase` is now `[restart].stage`, and `KINO_<MODEL>_RESTART_PHASE` is
  `KINO_<MODEL>_RESTART_STAGE`; `--restart-from` is unchanged. In Python, `Krea2Restart`,
  `Flux2Restart`, and `LTX2Restart` take `stage=` where they took `phase=`, and the
  `ImageRestartPhase` and `RestartPhase` types are gone. Saved run records write
  `restart.stage`.
- LTX-2 `--latent-stage` now names a stage, `stage-1` or `stage-2`, where it took `stage-1` or
  `final`; `KINO_LTX2_LATENT_STAGE`, `[restart].latent_stage`, and `LTX2Restart.decode()` take
  the same values. The default is the latest latent written before the restart stage, so a
  decode restart still reads the final latent. A decode restart's output bundle is suffixed
  `_stage2_decode` or `_stage1_decode` after the latent it read; the final latent's was
  `_final_decode`.
- A restart record that omits an artifact its stage reads, or carries one it does not, is
  refused with one message naming both sides, such as
  `decode restart takes final_latents, got text_conditioning`. LTX-2 refuses a latent stage its
  restart cannot read as `stage-2 restart cannot read latents written at stage-2`. Krea 2 run
  records no longer write a `restart.latent_stage` key, which was always `final` or null.
- `kinomlx config inspect` follows the stage vocabulary: `stations` became `stages` and
  `restart_stages`, `station_policies` became `stage_policies`, and `station_coverage` became
  `stage_coverage`. An artifact field's `produced_at` names the first stage, `text`, where it
  said `generate`.
- Klein's `--dry-run` now prints a human-readable resource section and emits its JSON inventory
  only under `--json` (or `KINO_JSON=1`), as the other three families do now that they have the
  flag. A script that parses `kinomlx --model flux2 --dry-run` should add `--json`.
- A prompt-less `kinomlx --model krea2` no longer inspects resources. It is now the located
  configuration error the other three families give, naming `--dry-run` as the way to inspect.
  The `"action": "inspect"` payload is gone. `--dry-run --json` prints the resource report in its
  place, with components as an array carrying `kind` rather than an object keyed by kind, and
  `runtime_tensors` and `excluded_tensors` in place of `consumed_tensors`, so a reader that
  indexed `resources.components.transformer` should iterate the array and match on `kind`.
- LTX-2's Python names now carry the family prefix, as Krea 2's, Klein's, and GMNet's do.
  `DistilledRequest`, `DistilledRestart`, `generate_distilled`, and `restart_distilled` are
  `LTX2Request`, `LTX2Restart`, `generate_ltx2`, and `restart_ltx2`; `GenerationOutput`,
  `EncodedTextConditioning`, `TextConditioningProvenance`, `HDRAuthoring`,
  `HDRReferenceConditioningConfig`, `ImageConditioningConfig`, `VideoVAEDecodeDType`, and
  `VideoVAETilingConfig` take the `LTX2` prefix; and so do the ports and providers in
  `kinomlx.models.ltx2.components`, `.conditioning`, `.denoise`, and `.upscaler`, so
  `TransformerPort` is `LTX2TransformerPort`.
- Component protocols are named for their role, and the implementation KinoMLX ships takes the
  bare name. `NativeLTX2Components`, `NativeKrea2Components`, and `NativeGMNetComponents` are now
  `LTX2Components`, `Krea2Components`, and `GMNetComponents`, and their protocols are
  `LTX2ComponentsProvider` (formerly `DistilledComponents`), `Krea2ComponentsProvider`, and
  `GMNetComponentsProvider`. Likewise `NativeTextConditioner` and `NativeKrea2TextConditioner`
  are `LTX2TextConditioner` and `Krea2TextConditioner`, with the protocols
  `LTX2TextConditionerPort` and `Krea2TextConditionerPort`. `Krea2Components`,
  `GMNetComponents`, and `Krea2TextConditioner` were protocols in 0.0.2, so an annotation that
  used them should name the `Provider` or `Port` protocol instead.
- Klein's `inspect_resources` is now `prepare_resources`, the name the other families use, and
  its modules `loading`, `stations`, `conditioning`, and `pipeline` are now `source_loading`,
  `text_conditioning`, `velocity`, and the `pipelines` package. GMNet's `pipeline` module is the
  `pipelines` package too.
- The video VAE classes drop `Native`, which distinguished nothing in an MLX-only package:
  `NativeVideoVAE` and `load_native_video_vae` are `VideoVAE` and `load_video_vae`, the
  convolutional encoder and decoder classes and their loaders follow the same pattern, except
  that `NativeResBlock3d` becomes `VAEResBlock3d`, and `TilingConfig.auto_native_conv3d` is
  `auto_conv3d`. The decoder kind is `conv` or `diffusion` throughout, so the tiling receipt's
  `decoder_kind` reads `diffusion` where it read `diffusion-na`.
- Smaller Python changes: `kinomlx.media.ExrSampleType` is now `SampleDType`, which also types a
  frame stream's sample dtype; `resolve_video_vae_decode_dtype` is now
  `resolve_video_vae_decode_precision`, which returns the dtype with its overflow fallback;
  `LTX2Settings.resolved_adapter_accumulation()` returns a dtype rather than a string; Krea 2's
  and LTX-2's settings modules no longer export `TRANSFORMER_DTYPE_CHOICES`,
  `VAE_DTYPE_CHOICES`, or `TRANSFORMER_QUANTIZE_CHOICES`, which give way to named vocabularies
  such as `kinomlx.modeling.dtypes.FloatDType`; and their restart modules no longer export
  `RESTART_PHASE_CHOICES` or `LATENT_STAGE_CHOICES`.

### Added

- LTX-2 runs can carry several image conditions at once. The headline case is a first frame plus
  a last frame, `--image start.png@first --image end.png@last`, and any number of interior
  keyframes works the same way, such as `--image middle.png@60`. A position is `first`, `last`,
  or a frame number counted from 0; `last` needs no arithmetic and works with `--auto-duration`,
  because a named position resolves once the frame count is known. Frame 0 replaces the first
  latent frame; every other position appends clean keyframe tokens the model attends to, as the
  reference implementation does, at `(H / 32) * (W / 32)` extra tokens per keyframe in stage 2
  and a quarter as many in half-size stage 1. Author order is preserved on every surface, each
  condition keeps its own strength, and two conditions on one frame are refused across
  spellings, so `last` and `120` on a 121-frame run are a duplicate. The saved configuration
  keeps the spelling you wrote, the run log adds the frame it resolved to, as in
  `frame=last (120)`, and the per-stage media-conditioning sidecar
  records the absolute index.
- Each LTX-2 image condition can hold a strength per stage. `stage1_strength` and
  `stage2_strength` are optional numbers from 0 through 1 on every `[generate].images` record;
  an omitted one inherits the common `strength`, and a supplied one is that stage's whole
  strength. Two numeric suffixes on `--image` set them, so `--image last.png@120:0.9:0.5` holds
  the last frame at 0.9 through stage 1 and 0.5 through stage 2, as two numbers already mean on
  `--adapter`. One suffix sets the common strength, and a third is refused. The member flags
  `--image-stage1-strength` and `--image-stage2-strength` extend the record the preceding
  `--image` started. The per-stage media-conditioning sidecar's `condition_{index}_strength` is
  the strength that stage applied, and the run log names both values when they differ.
- `--first-frame-mode anchor|guide` chooses how an LTX-2 frame-0 image condition is applied, with
  `first_frame_mode` under `[generate]` and `KINO_LTX2_FIRST_FRAME_MODE` as the other spellings.
  The default, `anchor`, is 0.0.2's behavior: the first latent frame is replaced and held clean,
  so at strength `1.0` the frame-0 latent is the encoded input. `guide` instead appends the image
  as clean keyframe tokens and leaves the first latent frame denoised, so frame 0 comes out close
  to the input rather than equal to it. `guide` exists for runs carrying an adapter that is
  supposed to change the image, such as a colorize, upscale, sharpen or style adapter: under
  `anchor` frame 0 is carried through untransformed while frames 1 and later are transformed,
  which a viewer sees as a jump at frame 1. The mode governs the frame-0 condition alone; every
  other position appends either way. The per-stage media-conditioning sidecar records it as
  `first_frame_mode`, and the run log names it.
- `--dry-run` now works on Krea 2, LTX-2, and GMNet as well as Klein, with
  `KINO_KREA2_DRY_RUN`, `KINO_LTX2_DRY_RUN`, `KINO_GMNET_DRY_RUN`, and an `[operation].dry_run`
  table on each. It resolves and validates every source the run would need, reports it, and
  exits without loading tensors, so a relocated or partly downloaded checkpoint fails before any
  weights load instead of partway into a run, and two runs can be compared by identity. Krea 2
  and LTX-2 report every component's resolved path, source and graph identities, and binding
  policy; GMNet has one checkpoint and reports its path, numeric contract, and memory policy. A
  configuration mistake that only the checkpoint reveals, such as a Krea 2 `--vae-decoder` that
  contradicts the selected VAE, is a configuration error naming its table, with exit code 2, as
  on Klein. A dry run cannot be combined with a prompt, saved conditioning, or a restart, and a
  restart from a saved stage resumes with the flag off.
- LTX-2 video self-attention can be made sparse per denoise step with Sol-Attn block routing, a
  training-free approximation from NVIDIA's Sana Sol-Engine. `--stage2-sparse-attention auto`
  switches it on with Sol-Engine's published stage-2 schedule (tau 1, 1.25 and 1.5 on the three
  distilled steps). Otherwise `--stage2-sparse-attention 1,1.25,1.5` gives each stage-2 step a
  tau, a single value sets every step, `2-3=1.5` sets a range and leaves the other steps dense,
  and `dense` (the default) keeps the stage exact; `--stage1-sparse-attention` does the same for
  stage 1, which has no published schedule and so no `auto`. `--stage1-dense-layers` and
  `--stage2-dense-layers` keep chosen transformer layers dense, as numbers and ranges such as
  `0,46-47` or `none`, defaulting to `0`. The TOML spellings are `stage_1_sparse_attention`,
  `stage_2_sparse_attention`, `stage_1_dense_layers` and `stage_2_dense_layers` under
  `[generate]`. A larger tau is faster and further from dense attention; the output changes, so
  it is off by default. A stage-2 restart can change the stage-2 settings; the stage-1 settings
  are fixed once stage 1 has run. The resolved schedules are logged and recorded as
  `sparse_attention` in the run metadata.
- The LTX-2.5 diffusion VAE can sharpen its decode. `--vae-sharpen AMOUNT`
  (`[generate].vae_sharpen`, `KINO_LTX2_VAE_SHARPEN`) unsharp-masks the final stage's input, so
  the decoder draws sharper edges without amplifying grain as a filter on the finished pixels
  would, and `--vae-sharpen-radius PIXELS` (`vae_sharpen_radius`, default 2) sets the blur. It is
  off by default, not part of the reference decoder, refused with the convolutional VAE, and
  costs one pass over the final stage's input. The run record's `vae_decode.context_sharpening`
  names the amount and radius.
- `--vae-cache-limit-gb` (`[generate].vae_cache_limit_gb`, `KINO_LTX2_VAE_CACHE_LIMIT_GB`) fixes
  the MLX allocator cache an LTX-2 decode runs with, instead of the one the tile planner picks;
  the tiles then shrink to fit beside it. It does not change the cache the other phases use.
- Klein's `--vae-dtype bfloat16|float32` (`KINO_FLUX2_VAE_DTYPE`) selects the decoder's dtype,
  which was fixed at BF16. FP32 is for diagnostic comparison, and a run decoded in FP32 cannot be
  restarted.
- An adapter whose intrinsic scale applies now says so in the console, as in
  `LTX-2.3 adapter adapter_model.safetensors: scale 2 from adapter_config.json (lora_alpha=64,
  r=32); effective 2 at strength 1`, and its receipt in the run record keeps `scale_source` and
  `intrinsic_scales`. The source is the PEFT configuration, `adapter_config.json` or
  `lora_adapter_metadata`, on any model, or on Krea 2 and Klein saved alpha tensors that move a
  scale away from 1.
- Klein's generation and restart entry points, `generate_flux2` and `restart_flux2` among them,
  take a `components` provider, as the other families' do, so a caller can supply the
  transformer and decoder; `Flux2Components` is the provider KinoMLX ships and
  `Flux2ComponentsProvider` its protocol.

### Changed

- LTX-2's transformer now defaults to FP16 (`transformer_dtype = "float16"`) instead of BF16, on
  LTX-2.3 and LTX-2.5. It is closer to an FP32 reference and faster, and outputs at the same seed
  differ from 0.0.2's; `--transformer-dtype bfloat16` keeps the BF16 path. FP16's narrower range
  is the cost, though both versions' largest hidden states stay well inside it. A step whose
  latents turn nonfinite stops the run as before, and on an FP16 transformer the message names
  `--transformer-dtype bfloat16`. A streamed transformer now fuses its adapters in float16 under
  the default `--adapter-accumulation auto`.
- LTX-2's video VAEs now decode in FP16 by default with either decoder, SDR or HDR, where
  `--vae-decode-dtype auto` selected BF16 for SDR and FP32 for HDR in 0.0.2, so decodes give
  different frames at the same seed. Each decoded chunk is checked before its frames are
  written. When the first chunk is not finite, the decode runs again in BF16, whose range
  matches FP32's at half the memory, and the run record's `vae_decode.overflow_fallback` names
  both dtypes. A nonfinite chunk after frames were written, or one from a requested dtype, stops
  the run with `--vae-decode-dtype bfloat16` named; 0.0.2 passed nonfinite values into
  post-processing.
- LTX-2's convolutional VAE decoder runs its 3x3x3 convolutions in FP16 as banded Winograd
  transforms whose three temporal taps multiply in one Metal Performance Primitives (MPP) product
  per transform plane, summed in FP32 and only for the frames the decoder keeps. Where MPP is
  unavailable or switched off, the decoder uses three FP16 products per plane. It decodes in
  temporal tiles of at most 128 frames, which measured faster than longer ones, and keeps its
  allocations across tiles.
- LTX-2.5's diffusion video VAE runs its FP16 blocks on Metal kernels: an exact neighborhood
  attention that scores only the keys inside each window, the Q and K projections with their
  norms and rotary embedding in one pass, MPP products with their residuals, and one pass each
  for the norms and SwiGLU's SiLU. BF16 and FP32 decodes keep the masked MLX path. Every pass but
  the attention keeps the masked path's bits; the attention sums in another order, by no more
  than the masked path differs from itself under another tile plan. The decoder frees each
  block's input once the block has consumed it, evaluates Q, K and V as soon as they are built
  and the attention before its output projection, writes the SwiGLU's token tiles into one buffer
  in place, and releases its allocator cache after stages 1 to 3 and after stage 4. The run
  record's `attention_tiling` names the `backend` that ran (`fp16-neighborhood-kernels`,
  `mlx-sdpa-exact-neighborhood` or `mixed`), counts `kernel_invocations`, and gives the first
  `fallback_reason` of a block that kept the masked path.
- LTX-2's `--vae-tiling auto` now plans the decode's tiles and its allocator cache together, from
  the memory MLX can use, where 0.0.2 decoded under the run's cache limit. An FP16 diffusion
  decode runs in one pass whenever one pass fits; otherwise it takes temporal tiles by count, the
  shortest tile length that gives each count, so the last tile is no longer a short remainder,
  beside the smallest cache that holds a tile's whole block set, or the largest whole GiB that
  fits. A plan the device cannot address, with a tile past its maximum buffer size or past the
  kernels' 32-bit launch counts, is refused before the decode starts, and `--vae-cache-limit-gb`
  is kept as given.
- LTX-2 video self-attention and video text cross-attention run through KinoMLX's FP16 dense
  attention kernel in both precisions. The 64-wide audio self-attention, audio text
  cross-attention and audio-video cross-attention take the kernel when both sides have at least
  512 tokens (clips of about 20 seconds and longer); shorter clips run those calls on MLX SDPA,
  which is faster at their sizes. BF16 inputs are converted to FP16 once, saturating at FP16's
  largest value. `--no-dense-attention-kernel`, `dense_attention_kernel = false` under
  `[model_settings]`, or `KINO_LTX2_DENSE_ATTENTION_KERNEL=false` runs every call on MLX SDPA
  instead.
- LTX-2's FP16 transformer runs its attention projections and its feed-forward up projection
  through MPP's `matmul2d` at 1,024 rows and more. The up projection keeps the checkpoint's
  (out, in) layout instead of being pretransposed, where MPP runs it faster than MLX ran the
  pretransposed copy; below 1,024 rows MLX runs it on the (out, in) weight, a little slower than
  0.0.2 on that product. The FP16 layout policy now pretransposes only the down projection;
  listing `project_in:pretranspose` in `video_ff_layout_specs`, which the audio stream mirrors
  by default, keeps 0.0.2's layout. A GPU where a small check product does not match MLX keeps
  MLX's kernels, in either precision.
- LTX-2's BF16 transformer runs its large video-token linear products through MPP on FP16
  operands: the attention and audio-video projections and the feed-forward up projection, at
  1,024 rows and more. Both operands are scaled by a power of two before the FP16 conversion and
  the FP32 sum is unscaled exactly before rounding to BF16. A resident transformer converts each
  such weight once, on its first MPP call, and keeps the converted copy in place of the BF16
  one, so memory does not grow; a streamed one, or one that compiles its block groups, converts
  per call.
- LTX-2 now rotates each attention's queries and keys by their rotary position embeddings in one
  Metal pass instead of MLX's chain of operations over FP32 intermediates. A GPU where a small
  check rotation does not match MLX keeps MLX's operations.
- LTX-2's vocoder now runs its alias-free SnakeBeta activations as one Metal pass instead of MLX's
  single-channel upsampling and low-pass convolutions around SnakeBeta. The filter sums run in a
  different order, so the waveform differs by rounding.
- Plain LoRA and AdaLoRA adapters now fuse into each weight in one GPU pass instead of a chain of
  MLX operations (widen, one FP32 `addmm` per adapter, narrow, and a transposing copy for a
  weight stored transposed). The pass stages the adapter of largest rank on the FP32 tile loop
  MLX's GEMM runs, multiplies adapters of rank 64 or less in its epilogue, and writes each
  weight once in its stored layout. It needs FP32 accumulation, ranks that are multiples of 16
  and at most eight adapters on a weight; other formats and ranks, and GPUs where a small check
  fusion does not match MLX, keep the chain. LTX, FLUX.2 Klein and Krea 2 share the fuser. With
  one operation per weight, MLX's lazy weight reads overlap the first forward again.
- `[model_settings].model_generation`, `gemma_variant`, `transformer_quantize`, and
  `adapter_accumulation` are now named vocabularies, so an unrecognized value is refused with the
  members listed, as in `expected one of 'auto', 'float32', 'float16', got 'half'`. Accepted
  values are unchanged.
- An adapter whose PEFT family is not one this version implements is now refused when the
  checkpoint is parsed, naming the family, rather than passing through as an unrecognized kind
  and failing later against whichever fusion path received it.
- A PEFT export without its configuration now loads with a warning, at strength without
  `alpha/r`, where Krea 2 and Klein refused it. It is recognized by the standard
  `adapter_model.safetensors` name or by PEFT's `base_model.model.` tensor prefix, and the
  warning also lands in the receipt's `warnings`. Formats that cannot be read without their
  configuration still fail.
- Klein and LTX-2 adapter log lines now name the adapter by the path as selected, as Krea 2's
  already did, so a cached adapter reads `adapter_model.safetensors` rather than its blob digest.
- The Krea 2 console resource section now names the denoiser the way the generation header and
  the report already did, so `Denoise mode` reads `compact_unmasked_pretransposed` rather than
  `optimized` for the same run.

### Performance

Measured on an M1 Max with 64 GB under macOS 27, each against the path 0.0.2 ran unless the row
names another. The entries above say what changed.

LTX-2 transformer and attention:

| Change | Workload | Result |
| --- | --- | --- |
| FP16 transformer by default | 768x448x121 denoising steps | 6.0% faster in stage 1, 5.6% in stage 2 |
| Dense attention kernel | One 32-head video self-attention call, 30,576 tokens | 1.67 s, against 1.99 s for MLX's FP16 attention and 2.13 s for STEEL in BF16 |
| Dense attention kernel | 768x448x121 transformer forward | 2.9% faster in FP16, 4.0% in BF16 |
| Dense attention kernel, self-attention alone | 768x448x721 FP16 transformer forward | 8.3% faster |
| Dense attention kernel, 64-wide calls | One audio-to-video call, 768x448x721, BF16 | 23.9 ms, against 28.2 ms for STEEL |
| MLX SDPA after STEEL's removal | One 32-head video self-attention call, 5,376 tokens, BF16 | 85.7 ms, against 67.6 ms for STEEL and 52.9 ms for the dense kernel |
| MPP attention projections, FP16 | Joint video and audio transformer forward | 3.3% faster at 768x448x121, 1.9% at 768x448x721 |
| MPP feed-forward up projection, FP16 | The up projection at 1,024 rows and more | 7 to 8.5% faster than MLX on the pretransposed copy |
| MPP feed-forward up projection, FP16 | 768x448x121 transformer forward | 1.7% faster |
| MLX feed-forward up projection, FP16 | The up projection below 1,024 rows | 2 to 3% slower |
| MPP linear products, BF16 | Transformer forward | 11.3% faster at 768x448x121, 11.8% at 384x224x121 |
| MPP linear products, BF16 | 768x448x121 generation, two adapters | 128.5 s, against 140.8 s |
| One-pass rotary embeddings | BF16 transformer forward | 2.9% faster at 768x448x121, 3% at 384x224x121 |
| One-pass rotary embeddings | One block's rotations, 30,576 tokens | 8.6 ms, against 86 ms |
| Sparse attention, tau 1 | One video self-attention call, 30,576 tokens | About 430 ms, against 1.67 s for the dense kernel |

Video decoding:

| Change | Workload | Result |
| --- | --- | --- |
| FP16 convolutional decoder | 768x448x721 decode, 20 GiB cache | 25.7 s, against 54.6 s in BF16 |
| FP16 diffusion decoder kernels | 768x448x121 decode, 16 GiB cache | 12.8 s, against 54.0 s on the masked FP16 path and 63.5 s in BF16 |
| FP16 diffusion decoder kernels | 768x448x121 decode, MLX peak | About 8.6 GiB |
| Diffusion decode planner | 1024x576x721 decode, idle machine | Three 264-frame tiles beside a 26 GiB cache, 137.1 s |
| `--vae-sharpen` | 1024x576x121 diffusion decode | 22.1 s, against 22.0 s stock |

Audio, adapters, checks, and startup:

| Change | Workload | Result |
| --- | --- | --- |
| One-pass vocoder activations | 30-second audio of a 768x448x721 generation | 2.25 s, against 14.4 s |
| One-pass vocoder activations | 5-second audio | 0.40 s, against 2.09 s |
| One-pass adapter fusion | LTX-2.3 dev checkpoint, two adapters of rank 16 and 384 | About 7.5 TFLOP/s |
| One-pass adapter fusion | First step of each stage, 768x448x121, same adapters | 8.9 s against 12.1 s in stage 1, 23.2 s against 25.8 s in stage 2 |
| One-pass adapter fusion | 768x448x121 generation, same adapters | 123.7 s, against 130.9 s |
| `--dry-run` | Validating every source a run needs | About a second, without loading weights |
| Static imports of all four model families | `kinomlx --help` | About 0.2 s slower |

How the new paths' output compares:

| Change | Output |
| --- | --- |
| FP16 transformer | Forwards 3 to 16 times closer to an FP32 reference than BF16's; the largest hidden state sits 2.8 times (LTX-2.3, about 23,600) and 4.3 times (LTX-2.5) below FP16's largest finite value |
| Dense attention kernel | FP16 video calls bit-identical to MLX's; BF16 differs from STEEL by rounding |
| MPP linear products | FP16 bit-identical to `mx.addmm`; BF16 bit-identical to MLX's BF16 `addmm` unless a value lies about 2^28 below the largest of its tensor |
| MPP linear products, BF16 generation | Latents identical to MLX's BF16 linears |
| One-pass rotary embeddings | Bit-identical |
| One-pass vocoder activations | In the two clips above, 99.4% and 99.6% of BF16 samples unchanged, the rest one BF16 step away (77 and 79 dB SNR) |
| One-pass adapter fusion | Fused weights bit-identical to the chain's; latents identical |
| FP16 convolutional decoder | About 0.73 levels from BF16 on an 8-bit scale (first tile, mean) |
| FP16 diffusion decoder kernels | 82.1 dB PSNR against the masked path, no further apart than the masked path under another tile plan |

### Removed

- LTX-2's BF16 STEEL attention backend and its three settings.
- LTX-2's `--gemma-path` and `--sampler`, replaced by `--text-encoder-path` and the per-stage
  samplers.
- LTX-2's `[generate.image]` table and the `KINO_LTX2_IMAGE`, `KINO_LTX2_IMAGE_FRAME_INDEX`, and
  `KINO_LTX2_IMAGE_STRENGTH` variables, replaced by `[generate].images` and `KINO_LTX2_IMAGES`.
- Krea 2's prompt-less resource inspection and its `"action": "inspect"` payload.

### Fixed

- An LTX-2 adapter that targets the transformer's adaLN modulation tables now has those targets
  applied. The tables are bare parameters with no `.weight`, so 0.0.2 recorded such a target as
  missing and fused only the ordinary ones, dropping the adapter's conditioning half without a
  word. An audio exclusion now also reaches the audio-side cross-attention table,
  `scale_shift_table_a2v_ca_audio`, which was classed as video. The receipt counts these
  targets. Diffusers rejects such a checkpoint and ComfyUI skips these keys, so their output for
  it differs from KinoMLX's.
- An ancestral stage 1, LTX-2.5's default, accepts a `--sigmas-stage-1` schedule of more than
  eight steps. 0.0.2 sized its transition-noise stream for the pinned eight-step schedule, so a
  longer one failed on its eighth step with an internal counter instead of naming the schedule.
- On Krea 2 and Klein, a PEFT adapter selected through a Hugging Face cache snapshot now gets its
  `adapter_config.json` scale. The cache links `adapter_model.safetensors` to a digest-named
  blob, and 0.0.2 resolved that link before looking for the companion file, so it applied the
  adapter as if it had no configuration, without a word. The lookup now follows the path as
  selected and each link it passes through.
- The config-template text for Klein's `--width` and `--height` now says what they accept: any
  size of at least 64 when source-aspect matching applies, which rounds it up to a multiple of
  16, and otherwise a multiple of 16.

### Documentation

- Each model's guide is now a folder, `docs/ltx2/`, `docs/krea2/`, `docs/flux2/`, and
  `docs/gmnet/`, with an overview `README.md` and one page per task: model files, generation,
  editing or image conditioning, adapters, restart, performance, and the Python API. The former
  `docs/LTX2.md`, `docs/KREA2.md`, `docs/FLUX2.md`, and `docs/GMNET.md` are gone.
- Every argument of every model is listed in that model's `arguments.md`, with its default, its
  values, and what each does. The arguments all models share are listed in the new
  `docs/CONFIG.md`, which also covers config files, environment variables, and `--set`.
- The new `docs/USAGE.md` collects prompts, seed lists and seed modes, output names, sidecars,
  run records, stopping a run, and restart, which the README used to hold. The README is now a
  short landing page.
- The guides give their measured times, memory peaks, and quality differences in tables. The
  LTX guide's run-time table covers four LTX-2.5 runs on an M1 Max, from 768x448 for 5 seconds to
  1024x576 for 30 seconds, and the guide lists macOS 27's known frame-rate conversion problems
  with `--target-fps`, the input size limits of each `--vsr-spatial-mode`, and the restart output
  names.
- The performance guide explains the High Power energy mode, under which a sustained GPU workload
  keeps full speed where Automatic throttles it after a few minutes.
- A cached LTX-2.3 download needs no `--weights-path`; the 0.0.2 guide said to set it.

### Development

- The `dev` extra requires mypy 2.4.0, Ruff 0.17.1, tokenizers 0.23.3, and NumPy 2.5.4, and
  hatchling 1.32.4 for offline release builds, which run with `--no-build-isolation`. Ruff and
  mypy target Python 3.15.
- The CLI dispatcher and the configuration registry import every model's command and config
  schema statically; nothing in the package imports a module by name, and the import graph has
  no cycles. Catalog data moved to `kinomlx.model_catalog`, `validate_model_parser` to
  `kinomlx.cli.parser_contract`, and saved-configuration persistence (`configuration_payload`,
  `read_configuration`, `RunRecord`, and the restart loaders) takes the model's
  `ModelConfigSpec` instead of its name. Every invocation now loads all four model families,
  including `kinomlx --help` and the `config` and `weights` utilities; Performance has the cost.
- Ruff bans name-based imports inside the package through `kinomlx/.ruff.toml`:
  `importlib.import_module`, `importlib.__import__`, the `importlib.util` file loaders, and
  `runpy`. Tests may still import modules by name. The wheel excludes dotfiles, as the sdist
  already did.

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
  [Weight loading](docs/WEIGHTS.md), [Performance](docs/PERFORMANCE.md), [Krea 2](docs/krea2/README.md),
  [Klein](docs/flux2/README.md), and [Conventions](docs/CONVENTIONS.md).
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
