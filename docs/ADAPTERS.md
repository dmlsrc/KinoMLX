# Using adapters

Adapters are small sets of learned weights that change a model's style, subject, motion, or
sampling recipe. LoRA is one adapter format; LoKr, LoHa, and the others store their change
differently. KinoMLX does not bundle or download adapters: download one made for your model, then
add its local file.

```bash
kinomlx --model krea2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors:0.8
```

Add any trigger words the author lists to your prompt; KinoMLX does not insert them. `--lora` is
the same option as `--adapter`.

- [Choose a compatible adapter](#choose-a-compatible-adapter)
- [Set strengths and stack adapters](#set-strengths-and-stack-adapters)
- [Strength, alpha, and effective scale](#strength-alpha-and-effective-scale)
- [Exclude parts of an adapter](#exclude-parts-of-an-adapter)
- [Keep the export self-contained](#keep-the-export-self-contained)
- [Supported static export formats](#supported-static-export-formats)
- [How adapters are applied](#how-adapters-are-applied)
- [First-use weight checks](#first-use-weight-checks)

## Choose a compatible adapter

An adapter must be made for your model's architecture as well as be in a supported format. A
familiar filename, or a file that parses, does not show it was trained for the model you selected.

| Model | Supports | Guide |
| --- | --- | --- |
| LTX-2.3 / LTX-2.5 | Ordinary LoRA, with a strength per stage | [LTX2 adapters](ltx2/adapters.md) |
| Krea 2 | The static formats below, on the denoiser | [Krea 2 adapters](krea2/adapters.md) |
| FLUX.2 Klein 9B | The static formats below, on the transformer | [Klein adapters](flux2/adapters.md) |
| GMNet | No adapters | |

Acceleration adapters also need their own steps and guidance: adding the file does not select
its recipe. LTX2's distilled LoRAs and Klein's Turbo LoRAs have worked examples in their guides.

For Krea 2, Krea's guidance is to train adapters on Raw and run them on Turbo.

## Set strengths and stack adapters

Append a strength to the path; it defaults to 1. Strength multiplies the adapter's learned change.
It is not a percentage of the image, and different adapters respond differently to one value.

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors:0.8 --adapter detail.safetensors:0.3
```

Repeat `--adapter` to stack adapters. The same file twice is applied twice; KinoMLX does not
remove duplicates. Keep the order fixed when comparing results, because floating-point
accumulation can depend on it.

Separate flags set the same members on the latest `--adapter`, and never on its siblings:

```bash
kinomlx --model krea2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors --adapter-strength 0.8 \
  --adapter detail.safetensors --adapter-strength 0.3
```

A flag before any `--adapter`, or a member set twice, is an error, even with an equal value or
through an alias or the shorthand.

LTX2 also takes two numbers, `PATH:STAGE1:STAGE2`, and its own per-stage flags; see
[LTX2 strength per stage](ltx2/adapters.md#strength-per-stage).

Strengths must be finite. Zero turns an adapter off, and a negative strength subtracts its change.
Large values can ruin the result or overflow FP16, so start from the author's recommendation.
Quote a path that contains spaces. A numeric suffix after `.safetensors` is read as a strength
without checking the file; use a config file record for a path that really ends in one.

In TOML and Python, adapters are an ordered `adapters` list of records with `path`, `strength`,
and `exclude`; `exclude = []` clears a record's exclusions. Krea 2 and Klein add `step_strengths`
and `sigma_window`, and LTX2 adds `stage1_strength` and `stage2_strength`. A member another model
uses, such as `stage1_strength` in a Krea 2 record, is an error. The `--lora` spellings exist only
on the command line.

A list from a later source replaces the earlier list. `adapters=[]` clears yours but keeps
profiles switched on separately, such as Krea 2's `--turbo-4step`. The `KINO_<MODEL>_ADAPTERS`
variables take the same list.

An adapter's strength can also change from step to step; see
[Denoise schedules](SCHEDULES.md#adapter-strength-over-a-run).

## Strength, alpha, and effective scale

Your strength controls the adapter's contribution. A saved `alpha` describes how an export scales
its own factors; it is not a second strength. LTX2 and the image models treat it differently, so
do not multiply your strength by alpha yourself.

### LTX2: direct strength, except for PEFT exports

LTX2 applies ordinary LoRA factors at your strength, ignoring saved alpha and any saved export
scalar:

```text
W_run = W_base + stage_strength * (B @ A)
```

A rank-16 LoRA with alpha 8 at strength 0.5 therefore adds `0.5 * (B @ A)`. That matches
Lightricks' loader and the community LoRAs made for it. A stage strength replaces the common
strength for its stage; it does not multiply it.

A PEFT export is the exception. When it carries PEFT configuration, in an
[`adapter_config.json`](#keep-the-export-self-contained) or as Diffusers metadata inside the file,
LTX applies the scale it declares, as PEFT does when merging:

```text
W_run = W_base + stage_strength * (lora_alpha / r) * (B @ A)
```

Rank-stabilized LoRA (`use_rslora`) uses `lora_alpha / sqrt(r)`, and `alpha_pattern` and
`rank_pattern` entries override per module. Strength 1 then reproduces the author's merge: `r = 32`
with `lora_alpha = 64` adds `2 * (B @ A)`.

### Krea and Klein: export scaling, then user strength

```text
W_run = W_base + user_strength * export_scalar * intrinsic_scale * raw_delta
```

`raw_delta` is `B @ A` for ordinary LoRA; other formats rebuild it their own way. A saved
`scalar`, where the export has one, multiplies once; otherwise it is 1. The intrinsic scale
depends on the format:

| Export | Intrinsic scale |
| --- | --- |
| Ordinary LoRA with saved alpha | `alpha/r`, where `r` is the rank |
| Configured rank-stabilized LoRA (`use_rslora`) | `alpha/sqrt(r)` |
| Rank-scaled LoHa or factored LoKr | `alpha/r`, with the format's rank |
| Dense LyCORIS or AI Toolkit LoKr without PEFT configuration | `1`: no rank divisor, so alpha is ignored |
| Configured PEFT LoKr, dense factors included | `alpha/r`, with the configured rank |
| Self-contained rank-scaled LoRA, LoHa, or LoKr with no alpha | `1` |
| Rank-scaled LoRA, LoHa, or LoKr with saved alpha `0` | `1`: zero means unit scaling, not off |

For configured LoRA, LoHa, and LoKr, an alpha tensor wins; otherwise KinoMLX uses a matching
`alpha_pattern`, then `lora_alpha` or `alpha`, then 8. Rank-stabilized LoRA keeps its square-root
rule even with an alpha tensor. Keep the export's configuration with it: removing it can change
the scale even though the weights are the same.

For example, a rank-16 LoRA with alpha 8 at strength 0.5 contributes `0.25 * (B @ A)`, or
`1.0 * (B @ A)` with rank-stabilized scaling. A huge alpha in a dense LyCORIS or AI Toolkit LoKr
changes nothing, because that format ignores alpha; the same alpha in a rank-scaled export can
make a very large update.

**To turn an adapter off, set its strength to 0.** Editing saved alpha to 0 does not, as the
table shows, and it would not stop DoRA magnitude changes or saved biases either. The one case
where strength 0 still applies is a noise-level window whose bands carry their own `@strength`.

Some formats give alpha other meanings: in OFT it bounds a rotation, for example. For magnitude
and rotation formats, strength blends the finished adapted weight with the original. Saved full
weights replace the originals; see [how adapters are applied](#how-adapters-are-applied).

### See which scale applied

When an intrinsic scale applies, the console names its source and its effect at your strength.
That happens for PEFT configuration on any model, and for a Krea 2 or Klein saved alpha that moves
the scale away from 1:

```text
LTX-2.3 adapter adapter_model.safetensors: scale 2 from adapter_config.json (lora_alpha=64, r=32); effective 2 at strength 1
```

The run record keeps the same facts as `scale_source` and `intrinsic_scales` in the adapter's
receipt. No line means your strength applies to the raw update as it is.

## Exclude parts of an adapter

`--adapter-exclude` keeps parts of the model unchanged by the latest adapter. For example, an LTX2
visual adapter can leave the audio and the audio-video attention alone:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" --generate-audio \
  --adapter style.safetensors:0.8 --adapter-exclude audio,cross
```

Give the categories as one comma-separated list. Each model has its own categories, so check
[LTX2](ltx2/adapters.md#exclude-parts-of-the-model), [Krea
2](krea2/adapters.md#exclude-parts-of-the-model), or
[Klein](flux2/adapters.md#exclude-parts-of-the-model) before reusing a list across models.
Exclusions select model weights, not regions of the image.

## Keep the export self-contained

A standalone community LoRA or LoKr file needs no `adapter_config.json`: its tensors and metadata
carry the factors, alpha, and format. Do not create a configuration file because a download came
without one.

The separate JSON belongs to PEFT's export layout:

```text
adapter-folder/
  adapter_model.safetensors
  adapter_config.json
```

It holds what weight shapes cannot show: the adapter family, scaling rules, saved biases, and
format options. Keep both files when the author ships them. KinoMLX reads the configuration next
to a file named exactly `adapter_model.safetensors`, along the path you gave and every link it
follows, so a file selected from the Hugging Face cache still finds it. Embedded Diffusers
`lora_adapter_metadata` works too; if both are present, they must agree.

An `adapter_config.json` next to a differently named file is ignored.

A file that looks like a PEFT export, because it is named `adapter_model.safetensors` or its keys
start with PEFT's `base_model.model.`, still loads without its configuration. KinoMLX warns that
its scale is unknown and applies your strength without `alpha/r`; get the matching configuration
or a self-contained re-export. Formats whose factors cannot be read without their configuration
fail.

Do not rename a PEFT export or move it away from its configuration, and export a multi-adapter
file one adapter at a time.

If loading reports missing parameters, get a complete export from the author. These formats have
specific needs:

- VeRA, PVeRA, and RandLoRA need their saved projection bases; UniLoRA its saved indices and
  scales; SUPERTUNING its saved precomputed indices.
- TinyLoRA needs its saved projection tensors. Several vector groups need converting where the
  adapter was made, because the export omits which module uses which group.
- ADAMSS, FROD, OSF, and PSOFT exports omit cluster assignments or frozen bases. Convert them where
  they were made, into a complete LoRA or dense delta against the original checkpoint.
- PiSSA, CorDA, OLoRA, LoftQ, LoRA-GA, and KaSA can change the base weights during setup. Export a
  delta that includes that change; the saved A/B factors alone do not describe the update.
- Parameterized PEANUT activations need state the export omits. Export a dense delta instead.
- WaveFT exports using PEFT's inconsistent `coif5` filter need a corrected export: that filter has
  48 taps but declares a 28-sample reduction.

KinoMLX does not invent missing data: it does not substitute another random generator or guess
module order. FourierFT and WaveFT locations are rebuilt from their declared seeds, but other
omitted projection bases are not recreated from a seed.

Not supported as static adapters: prompt and embedding tuners, input-dependent routing (LILY,
POLY, X-LoRA, SHADOW, Arrow), token-conditional aLoRA, PVeRA sampling during inference, and
timestep-dependent TLoRA. Spatial convolution adapters do not fit the denoisers' linear weights.

## Supported static export formats

Krea 2 and Klein accept these families when the saved parameters describe a complete update to
compatible weights; LTX2 accepts ordinary LoRA only. Most users can start from the author's
ready-made file; this table helps when choosing a training tool's export or reading a format
error. Formats that share the name "LoRA" can still read rank, alpha, bias, or normalization
differently, and that changes the result.

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

Target names must name real weights in the selected model. Official dotted names, the supported
Diffusers and PEFT wrappers, and known flattened names are accepted. A legacy `lora_unet_` prefix
says nothing about the architecture. Ambiguous names or conflicting updates are errors, never a
silent reinterpretation.

## How adapters are applied

KinoMLX merges adapters into the model's weights in memory, for the current run only. The
checkpoint files never change, so leaving out `--adapter` next time gives the original model. LTX2
can merge a different set for each stage.

Most of the merging runs on the GPU when denoising first uses the weights, so the first step is
slower and the adapter preparation time alone understates the cost. LTX2 streaming repeats the
merge whenever a block reloads; see [LTX2 performance](ltx2/performance.md#run-on-32-gb).

Updates are computed against the original weight, summed in a fixed order in FP32, and cast to the
transformer's dtype once, so the result does not depend on how the work was scheduled. LTX2's
`--adapter-accumulation auto` sums in FP16 on a streamed FP16 transformer instead, where the
merge repeats; that changes rounding, see [fusion precision](ltx2/adapters.md#fusion-precision).

For magnitude and rotation formats, a fractional strength blends the finished adapted weight with
the original, rather than scaling an internal parameter. Saved full weights and trained biases
replace their parameters instead of adding to them.

A few rotation and projection formats need small matrix inverses or QR decompositions, which run
on the CPU. That adds preparation time only; ordinary LoRA merging and the transformer stay on the
GPU.

### Coverage warnings

Coverage is the share of the adapter that found a weight in the selected model. Shared bases and
training-only tensors are counted separately, so they do not inflate it. Low coverage warns rather
than refusing the file, because a deliberately partial adapter can be valid; unmapped updates warn
even when coverage is high.

If an adapter seems to do nothing, check its model and variant, strength, exclusions, trigger
words, and coverage. The run record lists which tensors were applied, excluded, unmapped, shared,
or training-only. An unknown parameter on a recognized target needs a compatible export, not a
higher strength.

## First-use weight checks

LTX2, Krea 2, and Klein check every adapter-changed weight the run uses for nonfinite values after
its first use, catching a broken merge before the run continues. Stacked updates to one weight share
a check, excluded or disabled updates skip it, and a new transformer or LTX2 stage profile is
checked again. The console reports the check's time and tensor count within denoising time. Packed
LTX2 weights rely on the per-step latent checks instead.

The check is on by default. These are equivalent ways to turn it off:

- CLI: `--no-check-adapter-weights` (and `--check-adapter-weights` to turn it back on).
- TOML: `[settings].check_adapter_weights = false`.
- Environment: `KINO_CHECK_ADAPTER_WEIGHTS=false`.
- Python: `Settings(check_adapter_weights=False)` for the runner or resources.

Turning it off saves its time on a model, dtype, and adapter combination you have already
checked. The per-step latent checks and the format, shape, strength, and target validation stay,
as do format rules such as DoRA normalization.

For a nonfinite result, try BF16, lower unusually large strengths, and check that the adapter was
made for this model. Passing the check shows the numbers are finite, not that the image is good.
