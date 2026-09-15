# Using adapters

Adapters are additional learned weights that change a model's style, subject,
motion, or generation recipe. LoRA is one adapter format; LoKr, LoHa, and other
formats use different ways to represent the learned change.

Download an adapter compatible with your model, then add its local safetensors
file to a generation command. KinoMLX does not bundle or download adapters.

```bash
kinomlx --model krea2 \
  --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors:0.8
```

Replace `style.safetensors` with your file. If the author specifies trigger words,
include them in the prompt; KinoMLX does not insert them automatically. The
`--lora` spelling remains an alias for `--adapter`.

- [Choose a compatible adapter](#choose-a-compatible-adapter)
- [Set strengths and combine adapters](#inline-strengths)
- [Understand strength, alpha, and scaling](#strength-alpha-and-effective-scale)
- [Exclude parts of an adapter](#exclude-parts-of-an-adapter)
- [Keep required export files](#keep-the-export-self-contained)
- [Understand performance and output changes](#fusion-and-accounting)
- [Troubleshoot loading and checks](#first-use-weight-checks)

## Choose a compatible adapter

The adapter must match the model architecture as well as a supported export
format. A recognizable filename or a successfully parsed file does not establish
that it was trained for your selected model.

| Model | Adapter support | Model-specific instructions |
| --- | --- | --- |
| LTX-2.3 / LTX-2.5 | Ordinary two-dimensional LoRA weight adapters for the corresponding generation; stage-specific strengths and exclusions | [LTX adapter composition](LTX2.md#adapter-composition) |
| Krea 2 | Static LoRA, LoKr, LoHa, and the export families below; denoiser only, including text-fusion layers | [Krea adapters](KREA2.md#adapters) |
| FLUX.2 Klein 9B | The shared static export families, with Klein-compatible targets; choose adapters intended for your variant and recipe | [Klein generation and adapters](FLUX2.md#generate-and-edit) |
| GMNet | No adapter support | [GMNet guide](GMNET.md) |

Krea adapters trained on Raw can be intended for Turbo inference; follow the
[Krea pairing guidance](KREA2.md#adapters). Acceleration adapters also require
appropriate steps and guidance: adding a file alone does not select its sampling
recipe. LTX's distilled adapters and Klein base Turbo LoRAs have examples in their
model guides.

## Inline strengths

Append a strength to each path. The default is `1.0`; lowering it generally reduces
that adapter's influence. Strength is a multiplier, not a percentage of the final
image, and different adapters can respond differently to the same value.

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors:0.8 --adapter detail.safetensors:0.3
```

Repeat `--adapter` to stack files. Selecting the same file twice applies it twice;
KinoMLX does not deduplicate paths. Keep the order fixed when comparing results,
since floating-point accumulation can depend on order.

For LTX-2, a two-value suffix sets absolute stage-1 and stage-2 strengths:

```bash
kinomlx --model ltx2 --prompt "A slow camera move around a ceramic vase" \
  --adapter style.safetensors:0.25:0.5 \
  --adapter detail.safetensors:0:0.8
```

Here the style adapter affects both stages, while the detail adapter affects only
stage 2. One suffix applies to both stages. Krea and Klein accept one strength
because their generation recipes have one denoising stage.

Strengths must be finite. Zero disables an adapter's contribution in the selected
stage, and negative strengths subtract its learned change. Large magnitudes can
produce poor results or overflow FP16; start with the author's recommended range.
Quote the entire argument if the path contains spaces. Numeric `.safetensors` suffixes decode
without filesystem probes; use a structured path for a literal numeric suffix.

All adapter models use ordered adapter records. A modifier applies to the latest preceding CLI item,
with no broadcasting:

```bash
kinomlx --model krea2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors --adapter-strength 0.8 \
  --adapter detail.safetensors --adapter-strength 0.3
```

A modifier requires a preceding local `--adapter`. Supplying the same member twice is an error,
including equal values, aliases, and shorthand followed by a separate strength. Controls on one
item do not affect its siblings. An omitted strength retains the default `1.0` for that item.

Their TOML and API surfaces use `generate.adapters` and `adapters`, respectively. Every record has
`path`, `strength` and `exclude`. Krea and Klein add `step_strengths` and `sigma_window`; LTX adds
`stage1_strength` and `stage2_strength`. Unsupported model-specific fields are errors. Structured
paths remain literal.
See [Krea records](KREA2.md#adapters) and [Klein records](FLUX2.md#adapter-records) for examples and
model-specific category vocabularies. Higher-precedence collections replace the lower user
collection; `adapters=[]` clears user adapters while preserving independently enabled profiles.

LTX adds `--adapter-stage1-strength` and `--adapter-stage2-strength`. Each overrides that item's
common strength for one stage; a single inline common strength can be combined with a separate
stage override. Two-number shorthand explicitly supplies both stages. Its
[model guide](LTX2.md#adapter-composition) describes stage-specific restart locks. `--lora*` aliases
are command-line spellings only. Former public parallel-array fields are not accepted.
Variables in the selected model's configuration namespace must also match the current schema;
unknown names fail even when empty or overridden by the CLI. Use `KINO_KREA2_ADAPTERS`,
`KINO_FLUX2_ADAPTERS`, or `KINO_LTX2_ADAPTERS` with the same array-of-records syntax shown above.

## Varying strength over a run

An adapter can apply a different strength at each denoise step, either by step
index or by noise level, which switches it from fused to an unfused path applied
to activations. That changes which adapter formats are usable, how the run
compares numerically to a fused one, and how it composes with fused adapters.

See [Denoise schedules](SCHEDULES.md#adapter-strength-over-a-run) for the
spellings, the format support table, and the interactions worth knowing.

## Strength, alpha, and effective scale

The strength you select controls the adapter's contribution at inference. Saved
`alpha` describes how an export scales its learned factors; it is not a second
user strength. KinoMLX handles it differently for LTX and for Krea/Klein, so do
not manually multiply your requested strength by alpha without checking the format.

### LTX: direct strength, without alpha scaling

LTX applies ordinary LoRA factors as:

```text
W_run = W_base + stage_strength * (B @ A)
```

Saved alpha is deliberately ignored. There is no additional `alpha/r` or
`alpha/sqrt(r)` multiplier, including when a supported LoRA has a PEFT companion
configuration. LTX uses the mapped A/B factors directly; it does not carry a saved
export scalar into this update either. A stage override is the total strength for
that stage, not a multiplier on the common strength.

For example, rank 16, saved alpha 8, and strength `0.5` still give an LTX update of
`0.5 * (B @ A)`. This preserves the direct-strength behavior used by its community
checkpoints.

### Krea and Klein: export scaling, then user strength

For ordinary additive adapters, the effective update is:

```text
W_run = W_base + user_strength * export_scalar * intrinsic_scale * raw_delta
```

`raw_delta` is `B @ A` for ordinary LoRA; other formats reconstruct it differently.
A saved `scalar`, where supplied by the export, is multiplied once and otherwise
defaults to `1`. The intrinsic scale follows the format:

| Export | Intrinsic scale before user strength and saved scalar |
| --- | --- |
| Ordinary LoRA with saved alpha | `alpha/r`, where `r` is the factor rank |
| Configured rank-stabilized LoRA (`use_rslora`) | `alpha/sqrt(r)` |
| Rank-scaled LoHa or factored LoKr | `alpha/r`, using the format's rank denominator |
| Dense/dense LyCORIS or AI Toolkit LoKr without PEFT configuration | `1`; alpha is ignored because there is no rank divisor |
| Configured PEFT LoKr, including dense/dense factors | `alpha/r`, using the configured rank |
| Self-contained rank-scaled LoRA, LoHa, or LoKr with no alpha | `1` |
| Rank-scaled LoRA, LoHa, or LoKr with saved alpha `0` | `1`; zero means unit intrinsic scaling, not a disabled adapter |

For configured LoRA, LoHa, and LoKr, an explicit alpha tensor takes precedence.
Otherwise KinoMLX uses a matching `alpha_pattern`, then `lora_alpha` (or `alpha`),
with a fallback of `8` if those settings are absent. Rank-stabilized LoRA still
uses the square-root rank rule when alpha comes from a tensor. Keep the matching
[export configuration](#keep-the-export-self-contained): removing it can change
the effective scale even when the weight tensors are unchanged.

For example, ordinary LoRA with rank 16, alpha 8, scalar 1, and user strength `0.5`
contributes `0.25 * (B @ A)`. With rank-stabilized scaling it contributes
`1.0 * (B @ A)`. These are coefficients on the learned update, not percentages of
the generated image.

A very large finite alpha in dense/dense LyCORIS or AI Toolkit LoKr does not make
that update enormous: the format uses unit intrinsic scaling. This is a format
rule, not a general clamp on large alpha values. The same alpha in a rank-scaled
export can produce a very large update.

**Set user strength to `0` to disable a contribution.** The one exception is a
noise-level window whose bands carry their own `@strength`: those bands own the
magnitude, so the adapter still applies. Editing saved alpha to zero does not
disable a contribution either: the rank-scaled families above treat it as unit
scaling, and it
does not suppress DoRA magnitude changes or saved bias updates. Specialized
formats such as AdaLoRA and OFT have their own parameter meanings; OFT's alpha,
for example, controls a rotation constraint rather than the LoRA multiplier.
For magnitude and rotation formats, user strength blends the completed adapted
result with the original weight. Saved full weights use replacement semantics;
see [fusion and accounting](#fusion-and-accounting).

## Exclude parts of an adapter

Use `--adapter-exclude` when you want to keep selected model components unchanged.
For example, an LTX visual adapter can leave audio and cross-modal targets alone:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" --generate-audio \
  --adapter style.safetensors:0.8 --adapter-exclude audio,cross
```

Each exclusion modifier supplies one comma-separated list for the latest preceding adapter.
Omit it for an item with no exclusions; `exclude=[]` explicitly clears a structured record's list.
Category names differ by model; consult
[LTX](LTX2.md#adapter-composition), [Krea](KREA2.md#adapters), or
[Klein](FLUX2.md#adapter-records) before reusing an exclusion across models.
Exclusions select model weights, not spatial regions in the output.

## Keep the export self-contained

A standalone community LoRA or LoKr file does not automatically need an
`adapter_config.json`. Supported self-contained exports carry the needed factors,
alpha values, and format information in their tensors or safetensors metadata.
Do not create a companion file just because the download did not include one.

The separate JSON belongs to the standard PEFT export layout:

```text
adapter-folder/
  adapter_model.safetensors
  adapter_config.json
```

It supplies settings that weight shapes alone may not reveal, such as adapter
family, alpha/scaling rules, saved biases, and format-specific options. Keep both
files when the author distributes this layout. KinoMLX requires the companion
for the exact filename `adapter_model.safetensors`, with one exception: Krea and
Klein can use a valid embedded Diffusers `lora_adapter_metadata` configuration
when no companion is present. LTX's ordinary-LoRA loader requires the companion
for that filename and does not use this embedded-configuration exception.

For other filenames, KinoMLX does not apply an unrelated neighboring
`adapter_config.json`. Krea and Klein still read supported embedded configuration.
If both embedded and companion configurations are used, they must agree.
Do not rename a PEFT export to bypass a missing-config error: that can lose its
scaling semantics. Obtain the matching configuration or a self-contained re-export.
If an export contains several named adapters, export the intended adapter separately.

If loading reports missing parameters, obtain a complete export from its author. These
formats have specific requirements:

- VeRA, PVeRA, and RandLoRA need their saved projection bases. UniLoRA needs its
  saved indices and scales; SUPERTUNING needs saved precomputed indices.
- TinyLoRA needs saved projection tensors. Multiple vector groups require a
  producer-side conversion because the export omits the module-to-group map.
- ADAMSS, FROD, OSF, and PSOFT exports omit cluster assignments or frozen
  initialization bases. Convert in the producer environment to a complete
  static LoRA or dense delta against the original base checkpoint.
- PiSSA, CorDA, OLoRA, LoftQ, LoRA-GA, and KaSA can change the frozen base during
  setup. Export a delta that includes that change against the original
  checkpoint; ordinary saved A/B factors alone do not describe the full update.
- Parameterized PEANUT activations require omitted activation state. Export a
  complete dense delta rather than substituting freshly initialized parameters.
- WaveFT exports using the inconsistent PEFT `coif5` filter definition need a
  corrected export. That definition has 48 filter taps but declares a 28-sample
  reduction, so substituting another wavelet would change the intended adapter.

Reconstruction does not substitute a different random generator or guess
trainer-specific module order. FourierFT and WaveFT's declared location seeds
are reproducible without Torch. Projection bases omitted by other families are
not recreated from a seed alone.

Prompt/embedding tuners, input/task-dependent routing (LILY, POLY, X-LoRA,
SHADOW, Arrow), token-conditional aLoRA, PVeRA sampling during inference, and
timestep-dependent TLoRA are not supported as static adapters in KinoMLX.
Spatial convolution adapters cannot be fitted to denoiser Linear weights.

## Supported static export formats

Krea and Klein accept the following static export families when their saved
parameters describe a complete update to compatible denoiser weights. LTX accepts
only the ordinary LoRA format described above. This table is useful when choosing
an export from a training tool or diagnosing a format error; most users can start
with the author's ready-made safetensors file and its companion configuration.

Different formats can interpret rank, alpha, bias, or normalization differently.
Those distinctions affect the result even when filenames all say "LoRA".

| Export family | Compatibility details |
| --- | --- |
| LoRA / rsLoRA / exported DyLoRA | A/B or down/up factors, per-target rank and alpha, rank-stabilized scaling, and optional B bias |
| AdaLoRA | A/B/E, retained or zero rank, original-rank normalization from configuration or legacy `ranknum` |
| DoRA / DoHa / DoKr | Magnitude-normalized LoRA, LoHa, or LoKr; PEFT and LyCORIS magnitude axes are distinguished |
| LoCon | Pointwise or Tucker factors that reconstruct a denoiser Linear weight |
| LoHa / Hada | Hadamard product of two low-rank products, including Tucker factors |
| LoKr | Dense, factored, mixed, or Tucker Kronecker factors; PEFT configured alpha/r differs from all-dense LyCORIS scaling |
| GLoRA | LyCORIS matrix-product form and the distinct PEFT A/B/C/D/E paths, including weight-dependent bias |
| IA3 | LyCORIS delta-from-identity or PEFT identity-initialized vectors; output scaling also transforms bias |
| OFT / BOFT | LyCORIS output rotations, PEFT input rotations, current packed OFT and older full-block OFT plus output scale |
| HRA | PEFT ordered Householder reflections or configured Gram-Schmidt reflection |
| FourierFT | Spectrum, declared scale, and exact seeded Torch CPU frequency locations, reconstructed with MLX FFT |
| WaveFT | Spectrum with direct placement or one-level wavelet synthesis; `db1`-`db10`, `sym2`-`sym10`, `coif1`-`coif4`, and `haar` as `db1` |
| BEFT / LN_TUNING | Bias addition or saved full norm parameters |
| HiRA / DeLoRA | Base-weight Hadamard low-rank product, or normalized factors with saved gain and weight norms |
| C3A / MiSS | Circulant blocks or repeated/mini/BAT block updates |
| ROAD | Output rotations and scaling, including transformed bias |
| GraLoRA | Grouped factors and optional general low-rank branch |
| PEANUT | Saved weight-space encoder/decoder factors with stateless activation functions |
| DEFT / PaRa | Saved projection, optional injection matrix, and ReLU or QR projection |
| SHIRA / SUPERTUNING | Saved sparse indices and values, plus SUPERTUNING's optional low-rank branch |
| VeRA / PVeRA | Saved shared projection bases and per-target vectors; deterministic PVeRA inference only |
| RandLoRA | Saved shared random bases and per-target scale vectors |
| VBLoRA | Saved vector bank with complete logits or unambiguous compressed top-k indices/probabilities |
| UniLoRA | Saved shared vector, indices, and scales |
| TinyLoRA | Saved A/B/projection tensors and one unambiguous shared-vector group |
| Block-diagonal LoRA / QALoRA | Configured block-diagonal factors or the declared grouped-input pooling map |
| VeLoRA / Monteclora / MiCA inference | Complete saved A/B factors; declared training-only state is accounted for but not applied |
| Full / norm / replacement | Direct weight/bias deltas, `set_weight`, configured full modules, and trained base biases |

Exported target names must identify actual weights in the selected model.
Official dotted names, supported Diffusers/PEFT wrappers, and recognized flattened
names are accepted. A legacy `lora_unet_` prefix alone does not indicate the model
architecture. Ambiguous names or conflicting updates produce an error rather than
silently applying a different interpretation.

## Fusion and accounting

### Why loading an adapter does not change the checkpoint

KinoMLX combines adapters with model weights in memory for the current generation.
Original checkpoint files stay unchanged, so removing `--adapter` on the next run
returns to the original model. LTX can use a different combination in each stage.

Adapter preparation includes file validation and building the calculation, but much of the GPU
work runs when denoising first uses the weights. Consequently, the adapter preparation time
alone understates the full cost. LTX block streaming can repeat fusion each time a block is
reloaded; retained tensors avoid that repeat work at the cost of additional live memory. See
[LTX streaming](LTX2.md#block-streaming-on-32-gb-systems).

### Why precision and format affect the result

Ordinary adapter deltas are computed against the same original weight, accumulated
in a stable order in FP32, then cast to the selected transformer dtype. User
strength multiplies the complete unit-strength delta. For magnitude and rotation
formats, fractional strength blends the completed adapted result with the original
weight; it does not simply scale an internal normalization or rotation parameter.
Saved full weights and trained biases replace their corresponding parameters, so
they should not be interpreted as another additive low-rank delta.

LTX's `--adapter-accumulation auto` instead accumulates ordinary LoRA terms in FP16
on a streamed FP16 transformer, where fusion repeats. Its matrix products still
sum in FP32. `--adapter-accumulation float32` selects FP32 fusion explicitly when
comparing resident and streamed results. FP16 accumulation can be faster but
changes rounding and numerical range; see [LTX precision details](LTX2.md#component-loading).

Some rotation and projection formats need small inverses or QR decompositions on
the CPU because MLX does not provide those operations on the GPU. This can add
preparation time without moving ordinary LoRA fusion or the transformer to the
CPU. Supported spectral adapters reproduce their declared frequency locations;
missing random projection bases are not guessed from a seed.

### Understand coverage warnings

Coverage reports how much of the adapter could apply to the selected model.
Shared projection bases and training-only tensors are recorded separately so they
do not inflate this percentage. Low coverage warns rather than automatically
rejecting the file; intentionally partial adapters can be valid. Unmapped updates
still warn even when aggregate coverage is high.

If an adapter appears ineffective, check its model/variant, strength, exclusions,
trigger words, and coverage warnings. Inspect the saved run record for applied,
excluded, unmapped, shared, and training-only tensors. An error about unknown
parameters on a recognized target requires a compatible export, not a larger strength.

## First-use weight checks

Klein, Krea, and LTX check adapter-modified active floating-point weights after
first use. This catches nonfinite fused weights before continuing generation.
Stacked updates to one target share a check; disabled and excluded updates do not
schedule a scan. A new transformer or adapter stage profile can require another
check. The console reports the duration and tensor count within denoising time.
Packed LTX weights rely on per-step latent checks instead of this weight scan.

The check is enabled by default. These controls are equivalent:

- CLI: `--check-adapter-weights` / `--no-check-adapter-weights`.
- TOML: `[settings].check_adapter_weights = true` or `false`.
- Environment: `KINO_CHECK_ADAPTER_WEIGHTS=true` or `false`.
- Python: `Settings(check_adapter_weights=False)` when constructing the runner or resources.

Disabling it removes the final fused-weight scan, which can save time for a
previously checked model/dtype/adapter combination. It does not disable per-step
latent checks or format, shape, strength, and target validation. Ordinary LoRA and
LoKr factors are not separately certified finite before fusion, so detection can
happen when weights or latents are evaluated. Format-specific constraints such as
DoRA normalization remain mandatory.

For a nonfinite result, try BF16, reduce unusually large strengths, and verify the
adapter's intended base model. Passing a finite-value check establishes numerical
finiteness, not visual quality or that the adapter matches your intended subject.
