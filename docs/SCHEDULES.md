# Denoise schedules and adapter strength over a run

Two things vary as a model denoises: the **noise level** at each step, and the
**adapter strength** applied at each step. They are documented together because
they are coupled - an adapter can select its steps by noise level, which only
means something once the schedule is known.

Everything here is optional. A run that sets none of it behaves exactly as it did
before these controls existed.

- [The schedule](#the-schedule)
  - [What every run reports](#what-every-run-reports)
  - [Setting the schedule](#setting-the-schedule)
  - [LTX-2 has two of them](#ltx-2-has-two-of-them)
- [Adapter strength over a run](#adapter-strength-over-a-run)
  - [One strength per step](#one-strength-per-step)
  - [Strength by noise level](#strength-by-noise-level)
  - [Which spelling to use](#which-spelling-to-use)
- [How a varying adapter is applied](#how-a-varying-adapter-is-applied)
  - [Which formats can vary](#which-formats-can-vary)
  - [What this changes numerically](#what-this-changes-numerically)
  - [Composition with fused adapters](#composition-with-fused-adapters)
- [First-class profiles](#first-class-profiles)
- [A worked example](#a-worked-example)
- [Things that will bite](#things-that-will-bite)

## The schedule

A schedule is a list of noise levels from high to low. It holds **one more value
than the steps it drives**: step `i` is evaluated at `sigmas[i]`, and the last
value is the level the final step arrives at, never a step of its own. Eight
steps therefore need nine values.

Each model produces its own by default:

| Model | Default schedule |
| --- | --- |
| Krea 2 | A uniform 1-to-0 ramp, shifted by `mu`. Turbo pins `mu=1.15`; Raw derives it from the token grid. |
| Klein | An empirical shift over the target token grid, so the same step count gives different levels at different resolutions. |
| LTX-2 | Two literal pinned schedules, one per denoise stage. |

### What every run reports

The resolved schedule appears in the sampling section of the console:

```text
Sampling:
  Steps: 8 | Guidance: 0 | Mu: 1.15 | Noise backend: mlx
  Sigmas: 1, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0
```

A long schedule keeps its head in full and reports its length, because the early
high-noise steps are the ones worth reading. The run metadata always records the
complete array: Krea 2 under `denoise.sigmas`, Klein per image under `schedule`,
LTX-2 as `stage_1_sigmas` and `stage_2_sigmas`.

### Setting the schedule

Supply the levels to replace the model's derivation:

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise" \
  --sigmas "1,0.9567,0.9045,0.8403,0.7595,0.6547,0.5128,0.3109,0"
```

The array **fixes the step count**, because the count is just its length minus
one. Nine values run eight steps whatever `--steps` resolved to, and the sampling
section reports the step count actually used so the override stays visible. For
Krea 2, `mu` only shapes a generated schedule, so it is reported as `unused
(explicit sigmas)`.

Levels must be finite, lie between 0 and 1, and decrease strictly. A final value
above zero is allowed and simply stops the denoise early, leaving residual noise
in the latent - a deliberate choice, not an error.

TOML uses `sigmas`:

```toml
[generate]
sigmas = [1.0, 0.85, 0.6, 0.3, 0.0]
```

### LTX-2 has two of them

The distilled pipeline denoises in two stages with independent schedules. Both are
pinned literals, and both are reported and recorded:

| Stage | Steps | Pinned levels |
| --- | --- | --- |
| 1 | 8 | `1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0` |
| 2 | 3 | `0.909375, 0.725, 0.421875, 0` |

Stage 2 runs after the spatial upscaler, and its first three levels mirror the
tail of stage 1. Each stage has its own flag:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" \
  --sigmas-stage-1 "1,0.99,0.98,0.96,0.94,0.88,0.7,0.4,0" \
  --sigmas-stage-2 "0.9,0.72,0.42,0"
```

Each array sets its own stage's step count, so the nine values above keep stage 1
at eight steps while respacing them. Overriding one stage leaves the other on its
pinned values, so either flag can be used alone:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" \
  --sigmas-stage-2 "0.9,0.72,0.42,0"
```

Stage 2 legitimately starts **below** 1.0, with a pinned first level of
`0.909375`, because it continues from stage 1's tail rather than from pure noise.
Writing a replacement that starts at 1.0 would treat an already-denoised latent as
pure noise.

## Adapter strength over a run

An adapter normally applies one constant strength for the whole run. Some
adapters should not: a turbo diversity adapter decides composition in the first
high-noise steps, and the remaining steps are detail refinement it only disturbs.
There are two ways to say when it applies.

### One strength per step

Give one value per step, in order:

```bash
kinomlx --model krea2 --prompt "A still life with fruit" --steps 8 \
  --adapter sda.safetensors \
  --adapter-step-strengths "1,1,0,0,0,0,0,0"
```

Full strength for the first two steps, off for the rest. The array must hold
exactly one value per resolved step, which is checked before any weights load.
An array that repeats one value for every step is a constant, so it fuses at
that value rather than taking the unfused path, and the run is identical to
giving that strength on its own.
Each modifier applies to the latest preceding CLI adapter. Structured `step_strengths` belongs
inside that adapter's record; an omitted member leaves that item unscheduled. Empty arrays are
invalid and arrays do not broadcast:

```toml
[generate]
adapters = [
  {path="sda.safetensors", step_strengths=[1, 1, 0, 0, 0, 0, 0, 0]},
  {path="style.safetensors"},
]
```

These record rules apply to both [Krea](KREA2.md#adapters) and
[Klein](FLUX2.md#adapter-records). Removing an inherited schedule requires replacing the item
without its `step_strengths` member.

An array replaces that adapter's own `--adapter-strength`, so supplying both for
one adapter in the same config layer is an error.

### Strength by noise level

Select noise levels instead of step indices:

```bash
kinomlx --model krea2 --prompt "A still life with fruit" \
  --adapter sda.safetensors --adapter-sigma-window "0.93"
```

`0.93` means "at every level from 0.93 up to 1". A band is written
`low[:high][@strength]`, and several comma-separated bands express a taper:

```bash
--adapter-sigma-window "0.93@1,0.6:0.93@0.4,0:0.6@0.1"
```

Bands are ordered and **the first band holding a level wins**, which makes
overlap deterministic and lets a coarse band sit last as a fallback. A level no
band holds contributes nothing. A band that omits `@strength` uses the adapter's
own `--adapter-strength`, so a window pairs with a constant strength naturally -
the window says *when*, the strength says *how much*:

```bash
--adapter sda.safetensors --adapter-strength 0.75 --adapter-sigma-window "0.93"
```

When a band declares its own magnitude the scalar is a fallback that band never
reads, so `--adapter-strength 0` beside a window whose bands carry `@strength`
still places the adapter and the bands decide every step.

**Zero means the step applies nothing.** A zero strength - from a `@0` band, from
a level no band holds, or from an all-zero step array - skips that adapter's term
outright for that step, producing exactly what the unadapted model produces. A
`@0` band is therefore a useful carve-out in the middle of a taper.

A window whose bands hold *no* level of the resolved schedule is refused rather
than silently doing nothing, because that means the bands were written for a
different step count or a different model. A band that holds levels and declares
zero is a deliberate choice and stands.

A window is resolved against the schedule the sampler actually walks, including
one you supplied and one a resolution-dependent shift produced. It needs no step
count, so nothing has to be edited when the step count changes.

A window and a step array are exclusive: they are two spellings of the same
thing.

### Which spelling to use

They are not interchangeable, and neither is a default for the other.

A step array is literal and matches what adapter model cards and community
tooling describe ("active for the first 2 of 8 steps"). It breaks when the step
count changes, which is why a mismatch is an error rather than a silent reshape.

A window is portable across step counts and is the honest spelling when an
adapter was trained at a known noise level. The catch is that the two genuinely
disagree on some schedules. The Krea community SDA adapter records a training
floor of 0.93; on the eight-step Turbo schedule that selects steps 0 and 1,
matching the gate its model card prescribes. On a four-step schedule no level
falls in its training band at all, so the window selects one step while the
community recipe uses two:

| Schedule | Levels | `--adapter-sigma-window 0.93` | `--adapter-step-strengths "1,1,..."` |
| --- | --- | --- | --- |
| 8-step Turbo | 1, **0.9567**, 0.9045, ... | steps 0 and 1 | steps 0 and 1 |
| 4-step Turbo | 1, 0.9045, 0.7595, ... | step 0 only | steps 0 and 1 |

Pick the one that matches what you are reproducing.

## How a varying adapter is applied

A constant-strength adapter is **fused**: its delta is written into the weight
once, and every step reads the modified weight. That cannot express a strength
that changes, because each change point would mean rewriting every adapted
weight - and for any format whose delta is a function of the base weight, the
original is gone and the change is not even invertible.

So an adapter with a step array or a window stays **unfused**. Its factors are
retained in the transformer's compute dtype and applied to activations, leaving
the base weight untouched. This is cheap only when the delta never has to be
built densely, which is what limits which formats can vary.

For a rank-32 adapter covering 264 Krea 2 targets, the retained factors are about
224 MiB, against 23.9 GiB of adapter-touched base weights. One or two varying
adapters alongside any number of fused ones is the expected case.

### Which formats can vary

| Can vary | Why |
| --- | --- |
| LoRA, AdaLoRA | A low-rank product applies as two skinny products. |
| LoKr | A Kronecker product applies by reshaping the input axis; the expansion is never built. |
| LoHa | The Hadamard product of two rank-r factors expands into one pair of rank r squared, so applied cost and retained size both grow with the square of the rank. Past the rank where that pair outgrows the dense delta, rank 40 on a 3072-wide target, it is refused by name instead. |

Formats whose delta is a function of the live base weight, DoRA and HiRA and DEFT
and MiSS among them, have no activation form and are refused by name. Fuse them at
one constant strength.

### What this changes numerically

A varying run is **not** numerically identical to a fused run at the same
constant strength. Fusion rounds the combined weight once and then multiplies;
the unfused path multiplies and adds the delta to the output. Both are correct,
they round differently, and there is no way to make them agree.

Runs without a step array or a window still fuse exactly as before, so existing
parity comparisons keep their meaning. Compare a varying run against another
varying run.

### Composition with fused adapters

Fused adapters compose into the weight at load; varying ones add at forward time.
So **a varying adapter always applies after anything fused into the same
weight**, whatever order the adapters were listed in.

For LoRA-family deltas this is invisible, because addition commutes. It becomes
visible with a base-dependent fused adapter: fuse a DoRA and vary a LoRA, and
DoRA's normalization is computed against base+DoRA with the varying term absent,
which then rides on top. That is the only tractable option - the alternative
recomputes DoRA's column norms every step, which is the dense cost being avoided.

## First-class profiles

Krea 2 wraps two community adapters as switches, so the controls above do not
have to be assembled by hand. `--turbo-4step` supplies the distilled four-step
recipe as defaults, and `--turbo-sda` applies the diversity adapter gated to its
own recorded training band. Both accept `--turbo-<name>-step-strengths` and
`--turbo-<name>-sigma-window`, and using them together prints a warning because
the pairing is unvalidated. A profile's step array carries its own magnitude, so
`--turbo-<name>-strength` beside it is an error, exactly as it is for a user
adapter. See [Krea 2 profiles](KREA2.md#combining-the-two-profiles)
for the combinations worth exploring.

## A worked example

The Krea community SDA diversity adapter is trained at a single high-noise node
and must be switched off after the first two of eight steps. Its repository ships
two key formats; the ComfyUI-format file is the native trainer output and maps
completely, while the diffusers-format conversion mis-splits two module names.

```bash
kinomlx --model krea2 \
  --prompt "A still life with fruit on a table" \
  --steps 8 \
  --adapter krea2_turbo_sda_v1.0_comfy.safetensors \
  --adapter-step-strengths "1,1,0,0,0,0,0,0"
```

The run reports what it resolved:

```text
Sampling:
  Steps: 8 | Guidance: 0 | Mu: 1.15 | Noise backend: mlx
  Sigmas: 1, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0
Adapter result 1/1: krea2_turbo_sda_v1.0_comfy.safetensors
  Model: Krea 2 Turbo | Types: lora | Strength: 1 | Targets: 264/264 | Step Strengths: 1, 1, 0, 0, 0, 0, 0, 0
```

Reading the two lines together shows the gate covering the level the adapter was
trained at, `0.956724` at step 1.

## Things that will bite

**Klein KV editing reuses its reference K/V.** It is extracted once at the first
step and reused, so a strength that changes after that extraction would be read
back from a cache built at the first step's strength. That combination is refused.
Fuse the adapter at one constant strength, or enable `recompute_reference_kv`
(`--recompute-reference-kv`), which extracts the reference K/V again at every step
for one extra reference forward per step.

**`--turbo-4step` brings its own schedule.** It rewrites the step count, so a
step array written for eight steps becomes wrong. An explicit `--sigmas` is an
advanced override that replaces the profile's schedule rather than a conflict
with it: the array decides the levels and the step count, and a step array
written for the resulting step count still applies per step.

**Changing `--steps` invalidates a step array.** The mismatch is an error naming
both counts, raised before any weights load. Use a window if you expect to sweep
step counts.

**A window needs the schedule to exist.** It is resolved at the start of
denoising, not at configuration time, so a malformed band is caught early but an
empty selection is only discovered once the schedule is known.

**Alternative spacings are not implemented.** Karras, beta, and similar spacings
have a place to plug in, but no such scheduler ships today. `--sigmas` is the way
to run an arbitrary schedule now: compute the levels however you like and pass
them.
