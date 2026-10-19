# Adapters with Krea 2

Krea 2 takes LoRA, LoKr, LoHa, and the other [static
formats](../ADAPTERS.md#supported-static-export-formats) for its denoiser. Two community adapters
have dedicated switches. The [shared adapter guide](../ADAPTERS.md) covers formats and scaling in
depth.

- [Add adapters](#add-adapters)
- [Exclude parts of the model](#exclude-parts-of-the-model)
- [Four-step Turbo](#four-step-turbo)
- [Seed diversity (SDA)](#seed-diversity-sda)
- [Both profiles together](#both-profiles-together)
- [Alpha and unexpectedly strong adapters](#alpha-and-unexpectedly-strong-adapters)
- [Check what an adapter changed](#check-what-an-adapter-changed)
- [Config files and Python](#config-files-and-python)

## Add adapters

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise, cinematic light" \
  --adapter /path/to/style.safetensors --adapter-strength 0.8 \
  --adapter /path/to/detail.safetensors --adapter-strength 0.3
```

Each `--adapter` starts a new adapter, and the flags after it change only that one. Strength
defaults to 1, and zero and negative values are allowed. `--adapter style.safetensors:0.8` is the
same as a separate `--adapter-strength 0.8`. `--lora` is the same option as `--adapter`.

[Krea's guidance](https://github.com/krea-ai/krea-2#finetuning-krea-2) is to train adapters on
Raw and run them on Turbo, so that pairing is expected. When an adapter declares the model it
was trained on, the log and run record show the pairing; a filename alone does not establish it.

Adapters change the denoiser, including its text-fusion blocks, but not the Qwen text encoder or
the decoder. Add any trigger words the author lists to your prompt. `--adapter-step-strengths`
and `--adapter-sigma-window` vary an adapter's strength over the steps; see
[Denoise schedules](../SCHEDULES.md#adapter-strength-over-a-run).

## Exclude parts of the model

`--adapter-exclude` leaves parts of the denoiser unchanged by the latest adapter:

| Category | Leaves unchanged |
| --- | --- |
| `transformer` | The 28 main blocks |
| `text_fusion` | The blocks that turn the text encoder's output into the denoiser's text input |
| `input`, `output` | The input and output projections |
| `attn` | Attention |
| `ff` | Feed-forward layers |
| `gate` | Attention `to_gate` projections and feed-forward gates |
| `norm`, `bias`, `modulation` | Norms, biases, and modulation layers |
| `to_q`, `to_k`, `to_v`, `to_out`, `to_gate` | Single attention projections |

Separate categories with commas. For example, `--adapter-exclude text_fusion` keeps an adapter
from changing how the denoiser reads the prompt.

## Four-step Turbo

The community four-step adapter makes Turbo usable at four steps instead of eight. Download it
once (see [optional files](downloads.md#optional-files)), then switch it on:

```bash
kinomlx --model krea2 --turbo-4step --prompt "A painted landscape."
```

The profile supplies four steps, guidance 0, `mu` 1.15, and strength 1 as defaults. An explicit
`--steps`, `--guidance`, `--mu`, or `--sigmas` still wins, and KinoMLX notes that you left the
distilled recipe. The ordinary eight-step Turbo recipe is unchanged.

Its strength and exclusions are separate from your other adapters:

```bash
kinomlx --model krea2 --turbo-4step --turbo-4step-strength 0.75 --turbo-4step-exclude output \
  --adapter style.safetensors:0.8 --prompt "A painted landscape."
```

`--turbo-4step-exclude` takes the categories below, or `none`. Excluding parts of a distillation
adapter deliberately changes the recipe it was trained for.

Raw takes the profile with a warning: also supply a standard eight-step Turbo LoRA with
`--adapter`, because KinoMLX cannot tell whether another adapter provides it. The profile also
works with editing, where the edit adapter keeps its own strength and exclusions.

The run records the adapter's repository, revision, file, and placements. A denoise restart
refuses a different release; a decode restart does not need the adapter.

## Seed diversity (SDA)

Turbo's distillation makes seeds look alike: one prompt tends toward one composition. The
community SDA adapter restores variety. Download it once, then switch it on:

```bash
kinomlx --model krea2 --turbo-sda --prompt "A still life with fruit on a table."
```

It was trained in a narrow band of high noise levels, 0.93 to 0.97, and blurs detail if left on
for the whole run. By default it therefore applies at noise levels from 0.93, the floor it
records, up to 1. On the eight-step schedule that is the first two steps, as its model card
prescribes:

```text
step 0 @ 1.0000   active
step 1 @ 0.9567   active   <- the level it was trained at
step 2 @ 0.9045   off
```

The window follows the noise level, not the step number, so it keeps working when the step count
changes. Override it either way:

```bash
--turbo-sda-sigma-window "0.93:0.97"      # only the training band itself: step 1
--turbo-sda-step-strengths "1,1,0,0"      # name the steps instead
--turbo-sda-strength 1.5                  # stronger inside the window
```

A step list carries its own magnitudes, so `--turbo-sda-strength` is refused beside it; write
the magnitude into the list. Empty lists and empty windows are invalid. The four-step profile has
the same two controls.

## Both profiles together

The two profiles can run together, but no one has validated the pair, and KinoMLX says so on the
console. The four-step schedule never reaches the SDA's training band:

```text
4-step levels   1.0000, 0.9045, 0.7595, 0.5128
SDA band        0.93 to 0.97          <- nothing lands inside
```

Starting points to explore, not recommendations:

| Combination | Effect |
| --- | --- |
| `--turbo-sda` alone | The adapter as trained: eight steps, its own band |
| `--turbo-sda --turbo-4step` | Only step 0 falls inside the default window, since step 1 is below 0.93 |
| add `--turbo-sda-step-strengths "1,1,0,0"` | Both early steps, around the trained level; reported to work in the adapter's discussion thread |
| raise it to `"1.5,1.5,0,0"` | More strength instead of a wider band |
| add `--sigmas "1,0.9567,0.7595,0.5128,0"` | Moves step 1 onto the trained level: more diversity, less sharpness |

On a small local test, the last row gave the most seed-to-seed variety and the least fine
detail, and most of that detail loss came from moving the schedule, not from the adapter. How
much variety there is to gain depends on the prompt: a prompt that pins subject, wardrobe, and
place leaves little to vary. Test on your own prompts.

## Alpha and unexpectedly strong adapters

Your strength multiplies the update the export defines, and saved alpha is not always another
multiplier: some formats ignore it, and a saved alpha of 0 does not disable an update. Start from
the author's recommended strength rather than correcting for alpha yourself, and set strength 0 to
turn an adapter off. See [strength, alpha, and
scale](../ADAPTERS.md#strength-alpha-and-effective-scale) for each format's rule.

A standard PEFT export needs its `adapter_config.json` next to `adapter_model.safetensors`; Krea
also reads configuration embedded in the file. If loading reports missing bases, indices, or
mappings, get a complete export from the author: a higher strength cannot replace missing
parameters. See [keep the export self-contained](../ADAPTERS.md#keep-the-export-self-contained).

## Check what an adapter changed

Coverage below 50% warns, 0% is reported explicitly, and unmapped updates warn even when coverage
is high. Before raising the strength of a weak adapter, check its model pairing and exclusions.
The run record's `generation.adapters` lists which tensors were applied, excluded, unmapped,
shared, or training-only.

Adapters are merged into the weights in memory when the denoiser first uses them, so the first
step includes that work; the checkpoint files never change. Adapter-changed weights are then
checked for nonfinite values; see [first-use checks](../ADAPTERS.md#first-use-weight-checks).

## Config files and Python

TOML and `KINO_KREA2_ADAPTERS` take an ordered `adapters` list. Each record has `path`,
`strength`, `exclude`, `step_strengths`, and `sigma_window`:

```toml
[generate]
adapters = [
  {path = "style.safetensors", strength = 0.8, exclude = ["text_fusion"]},
  {path = "detail.safetensors"},
]
```

A step list replaces the strength and window of its record, so giving both is refused.
`--set 'generate.adapters[1].strength=0.25'` edits the second record, and
`--set 'generate.adapters[1]={path="detail.safetensors"}'` replaces it whole, dropping its other
members.

`generate.adapters=[]` clears your adapters but keeps the four-step, SDA, and edit profiles. The
profiles' TOML fields are `turbo_4step`, `turbo_4step_strength`, `turbo_4step_exclusions`, and the
matching `turbo_sda` fields.

Python requests take `Krea2Request(adapters=(Krea2AdapterConfig(...), ...))` from
`kinomlx.models.krea2`.
