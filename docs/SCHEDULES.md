# Denoise schedules and adapter strength over a run

Two things can vary as a model denoises: the **noise level** of each step, and the **adapter
strength** applied at each step. They are documented together because an adapter can select its
steps by noise level, which only means something once the schedule is known.

Everything here is optional. Without it, each model uses its own schedule and every adapter keeps
one strength for the whole run.

- [The schedule](#the-schedule)
  - [What every run reports](#what-every-run-reports)
  - [Setting the schedule](#setting-the-schedule)
  - [LTX2 has two of them](#ltx2-has-two-of-them)
- [Adapter strength over a run](#adapter-strength-over-a-run)
  - [One strength per step](#one-strength-per-step)
  - [Strength by noise level](#strength-by-noise-level)
  - [Which spelling to use](#which-spelling-to-use)
- [How a varying adapter is applied](#how-a-varying-adapter-is-applied)
- [Krea 2 profiles](#krea-2-profiles)
- [A worked example](#a-worked-example)
- [Things that will bite](#things-that-will-bite)

## The schedule

A schedule is a list of noise levels from high to low. It holds **one more value than the steps
it drives**: step `i` runs at `sigmas[i]`, and the last value is where the final step arrives, not
a step of its own. Eight steps need nine values.

| Model | Default schedule |
| --- | --- |
| Krea 2 | A uniform 1-to-0 ramp, shifted by `mu`. Turbo uses `mu` 1.15; Raw derives it from the image size. |
| Klein | A shift that depends on the output size, so one step count gives different levels at different sizes. |
| LTX2 | Two fixed schedules, one per denoising stage. |

### What every run reports

The console's sampling section shows the resolved schedule:

```text
Sampling:
  Steps: 8 | Guidance: 0 | Mu: 1.15 | Noise backend: mlx
  Sigmas: 1, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0
```

A long schedule shows its first values and its length, since the early high-noise steps matter
most. The run record always keeps the whole list: Krea 2 under `denoise.sigmas`, Klein per image
under `schedule`, and LTX2 as `stage_1_sigmas` and `stage_2_sigmas`.

### Setting the schedule

`--sigmas` replaces the model's own levels:

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise" \
  --sigmas "1,0.9567,0.9045,0.8403,0.7595,0.6547,0.5128,0.3109,0"
```

The list **fixes the step count**, its length minus one. Nine values run eight steps whatever
`--steps` says, and the sampling section shows the count actually used. Krea 2's `mu` shapes only a
generated schedule, so it is reported as `unused (explicit sigmas)`.

Levels must be finite, between 0 and 1, and strictly decreasing. A last value above zero is
allowed: it stops early and leaves some noise in the latent, deliberately.

In TOML, use `sigmas`:

```toml
[generate]
sigmas = [1.0, 0.85, 0.6, 0.3, 0.0]
```

### LTX2 has two of them

The distilled pipeline denoises in two stages, each with a fixed schedule:

| Stage | Steps | Levels |
| --- | --- | --- |
| 1 | 8 | `1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0` |
| 2 | 3 | `0.909375, 0.725, 0.421875, 0` |

Stage 2 runs after the spatial upscaler, and its levels repeat the tail of stage 1. Each stage has
its own option, and either can be used alone; the other keeps its fixed levels:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise" \
  --sigmas-stage-1 "1,0.99,0.98,0.96,0.94,0.88,0.7,0.4,0" \
  --sigmas-stage-2 "0.9,0.72,0.42,0"
```

Each list sets its own stage's step count, so the nine values above keep stage 1 at eight steps
while respacing them.

Stage 2 rightly starts **below** 1.0, at `0.909375`, because it continues from stage 1 rather
than from pure noise. A replacement starting at 1.0 would treat an almost finished latent as pure
noise.

## Adapter strength over a run

An adapter normally keeps one strength for the whole run. Some should not: a Turbo diversity
adapter decides composition in the first high-noise steps and only disturbs the detail steps that
follow. Krea 2 and Klein offer two ways to say when an adapter applies.

### One strength per step

Give one value per step, in order:

```bash
kinomlx --model krea2 --prompt "A still life with fruit" --steps 8 \
  --adapter sda.safetensors \
  --adapter-step-strengths "1,1,0,0,0,0,0,0"
```

This is full strength for two steps and off for the rest. The list must have exactly one value
per step, which is checked before any weights load, and an empty list is invalid. A list
repeating one value is a constant: the adapter is merged at that value, exactly as with a plain
strength.

The option applies to the latest `--adapter` only and never carries over to the others; an adapter
without one keeps its constant strength. In TOML, `step_strengths` goes in that adapter's record.
To remove an inherited list, replace the record without it:

```toml
[generate]
adapters = [
  {path = "sda.safetensors", step_strengths = [1, 1, 0, 0, 0, 0, 0, 0]},
  {path = "style.safetensors"},
]
```

A step list replaces that adapter's `--adapter-strength`, so giving both in one config file or
command line is an error.

### Strength by noise level

Select noise levels instead of step numbers:

```bash
kinomlx --model krea2 --prompt "A still life with fruit" \
  --adapter sda.safetensors --adapter-sigma-window "0.93"
```

`0.93` means every level from 0.93 up to 1. A band is written `low[:high][@strength]`, and several
comma-separated bands make a taper:

```bash
--adapter-sigma-window "0.93@1,0.6:0.93@0.4,0:0.6@0.1"
```

**The first band holding a level wins**, so overlaps are predictable and a broad band can go last
as a fallback. A level no band holds gets nothing. A band without `@strength` uses the adapter's
`--adapter-strength`: the window says *when*, the strength *how much*.

```bash
--adapter sda.safetensors --adapter-strength 0.75 --adapter-sigma-window "0.93"
```

When every band carries its own `@strength`, the plain strength is never read, so
`--adapter-strength 0` beside such a window still applies the adapter.

**Zero means that step applies nothing.** A `@0` band, a level no band holds, or an all-zero step
list skips the adapter for that step, giving exactly what the model gives without it. A `@0` band
is a handy gap in the middle of a taper.

A window that holds *no* level of the schedule is refused rather than silently doing nothing,
since it was clearly written for another step count or model. A band that holds levels and
declares zero is a deliberate choice and stands.

A window is matched against the schedule the run actually uses, including one you supplied or
one a size-dependent shift produced. It needs no step count, so it survives step-count changes.

A window and a step list are two ways to say the same thing, so an adapter takes one or the
other.

### Which spelling to use

Neither is a default for the other. A step list is literal and matches what model cards say
("active for the first 2 of 8 steps"); it breaks when the step count changes, which is why a
mismatch is an error. A window carries across step counts and is the honest spelling when an
adapter was trained at a known noise level.

The two genuinely disagree on some schedules. The Krea community SDA adapter was trained in the
band 0.93 to 0.97, and its window `0.93` covers 0.93 up to 1. On the eight-step Turbo schedule
that selects steps 0 and 1, as its model card prescribes. On a four-step schedule only the first
level, 1.0, is in the window and none is in the training band, so the window selects one step
while the community recipe uses two:

| Schedule | Levels | `--adapter-sigma-window 0.93` | `--adapter-step-strengths "1,1,..."` |
| --- | --- | --- | --- |
| 8-step Turbo | 1, **0.9567**, 0.9045, ... | steps 0 and 1 | steps 0 and 1 |
| 4-step Turbo | 1, 0.9045, 0.7595, ... | step 0 only | steps 0 and 1 |

Pick the one that matches what you are reproducing.

## How a varying adapter is applied

A constant adapter is merged into the weights once, and every step reads the merged weight. A
varying strength cannot work that way, so an adapter with a step list or window stays separate: its
factors are applied to the activations at each step, and the base weight is untouched.

That is cheap only when the update never has to be built in full, which limits the formats:

| Can vary | Why |
| --- | --- |
| LoRA, AdaLoRA | A low-rank product applies as two thin products. |
| LoKr | A Kronecker product applies by reshaping the input; the full update is never built. |
| LoHa | Its product expands into one pair of rank r squared, so its cost grows with the square of the rank. Above about rank 40 on a 3072-wide target it is refused by name. |

Formats whose update depends on the base weight, including DoRA, HiRA, DEFT, and MiSS, cannot
vary and are refused by name; merge them at one strength. Memory stays small:

| Varying adapter | Factors kept in memory |
| --- | ---: |
| Rank 32 on 264 Krea 2 targets | About 224 MiB |

A varying run is **not** numerically identical to a merged run at the same strength: merging
rounds the combined weight once, while the separate path adds the update to the output. Both are
correct, and they round differently. Compare a varying run with another varying run.

A varying adapter always applies after anything merged into the same weight, whatever the listed
order. For LoRA-like updates that changes nothing, because addition commutes. With a merged DoRA,
its normalization is computed without the varying adapter, which then adds on top.

## Krea 2 profiles

Krea 2 wraps two community adapters as switches, so you need not assemble these controls by hand.
`--turbo-4step` supplies the four-step recipe, and `--turbo-sda` applies the diversity adapter
within its own training band.

Both profiles take step lists and windows: `--turbo-4step-step-strengths`,
`--turbo-4step-sigma-window`, `--turbo-sda-step-strengths`, and `--turbo-sda-sigma-window`. A
profile's step list carries its own magnitudes, so `--turbo-4step-strength` or
`--turbo-sda-strength` beside one is an error. See [Krea 2
profiles](krea2/adapters.md#both-profiles-together) for combinations worth trying.

## A worked example

The Krea community SDA diversity adapter is trained at one high noise level and should switch off
after the first two of eight steps. Its repository ships two formats; the ComfyUI file is the
trainer's own output and maps completely, while the Diffusers conversion splits two module names
wrongly.

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

Read together, the two lines show the gate covering the level the adapter was trained at,
`0.956724` at step 1.

## Things that will bite

**Klein KV editing reuses its reference keys and values.** They are extracted on the first step,
so a strength that changes later would be read back at the first step's strength; that combination
is refused. Merge the adapter at one strength, or add `--recompute-reference-kv`, which extracts
them again every step for one extra reference pass per step.

**`--turbo-4step` brings its own schedule.** It sets the step count, so a step list written for
eight steps becomes wrong. An explicit `--sigmas` replaces the profile's schedule rather than
conflicting with it, and a step list written for the resulting count still applies.

**Changing `--steps` breaks a step list.** The mismatch is an error naming both counts, raised
before any weights load. Use a window if you sweep step counts.

**A window needs the schedule to exist.** It is resolved when denoising starts, so a malformed band
is caught early but an empty selection only once the schedule is known.

**Other spacings are not built in.** There is no Karras, beta, or similar spacing yet. To run any
schedule now, compute the levels however you like and pass them with `--sigmas`.
