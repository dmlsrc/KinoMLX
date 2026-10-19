# Running a model

Every model runs through one command, `kinomlx --model MODEL`. This page covers the controls
the models share: prompts, seeds, output files, run records, and stopping or restarting a run.
Each [model guide](../README.md#documentation) covers the rest, and
[Configuration](CONFIG.md) covers config files and environment variables.

- [Pick a model](#pick-a-model)
- [Check the model files first](#check-the-model-files-first)
- [Prompts](#prompts)
- [Seeds](#seeds)
- [Output files](#output-files)
- [Run records and machine-readable output](#run-records-and-machine-readable-output)
- [Stopping a run](#stopping-a-run)
- [Restarting from saved results](#restarting-from-saved-results)

## Pick a model

| `--model` | Makes | Guide |
| --- | --- | --- |
| `ltx2` | Video, with optional stereo audio, from text and images | [LTX2](ltx2/README.md) |
| `krea2` | Still images from text; edits of one or two reference images | [Krea 2](krea2/README.md) |
| `flux2` | Still images from text; edits of ordered reference images | [FLUX.2 Klein](flux2/README.md) |
| `gmnet` | An HDR still from an SDR photograph | [GMNet](gmnet/README.md) |

`kinomlx --model MODEL --help` lists that model's options. Without `--model`, and without a config
file that selects a model, KinoMLX runs LTX2.

## Check the model files first

`--dry-run` finds every model file a run would read, checks that they fit together, and exits
without loading any weights. Run it once after downloading a model:

```bash
kinomlx --model krea2 --dry-run
kinomlx --model ltx2 --model-generation 2.5 --dry-run --json
```

Add `--json` for a machine-readable inventory.

A dry run cannot be combined with a prompt, saved conditioning, a restart, or a GMNet `--image`.

## Prompts

`--prompt` takes the text directly. `--prompt-file` reads one complete prompt from a UTF-8 file
instead, keeping its line breaks; use it for long or multi-line prompts:

```bash
kinomlx --model ltx2 --prompt-file scene.txt
```

A missing or empty file, invalid UTF-8, and NUL characters are errors. Saved configs and run
records store the text itself, so a rerun does not need the file. Text given to `--prompt` is
never read as a path.

Krea 2 and Klein accept `--prompt` more than once to run a batch of prompts in order. LTX2 takes
one prompt per run.

### Generating without a prompt

`--prompt-unconditional` generates from a deliberately empty prompt with LTX2, Krea 2, or Klein.
Text encoding still runs on the empty prompt, and adapters and image conditions still apply. It
does not zero the text embeddings or switch guidance off. Giving neither a prompt nor this option
keeps each model's usual behavior for a missing prompt.

```bash
kinomlx --model flux2 --prompt-unconditional --seed 42
```

It conflicts with a nonempty prompt, including one inherited from a config file. Pass
`--no-prompt-unconditional` when a new prompt should override a config that sets it. In TOML it
is `[generate].prompt_unconditional = true`; requests take `prompt_unconditional=True`.

Krea 2's `--system-prompt` still applies. Saved text conditioning reused with this option must
itself hold an empty prompt.

## Seeds

`--seed` takes one seed or an ordered list. Spaces and commas both separate a list, and
`--seeds` is the same option:

```bash
kinomlx --model ltx2 --prompt-file scene.txt --seed 42 123 999
kinomlx --model krea2 --prompt-file scene.txt --seed 42,123,999
```

Order and duplicates are kept. Every prompt runs every seed, and results are ordered by prompt,
then seed. Seeds are unsigned 64-bit integers; an invalid list fails before anything loads.

For Krea 2 and Klein, `--seed 42 --images-per-prompt 4` means seeds 42, 43, 44, and 45 for each
prompt. An explicit seed list cannot be combined with more than one image per prompt.

In TOML, write `seed = 42` or `seeds = [42, 123, 999]`, never both in one file; a later source
replaces the earlier choice. Requests take `seeds=(42, 123, 999)` or a scalar `seed=`.

### How long models stay loaded across seeds

The image models choose how they reuse loaded models between seeds with `--seed-mode`:

| `--seed-mode` | What happens | Choose it when |
| --- | --- | --- |
| `sequential` (default) | Each seed runs through every stage, loading and releasing models as usual. | Memory is tight, or one image matters more than throughput. |
| `stage` | The transformer denoises every seed, then the decoder decodes every seed. | Several seeds should share one transformer load. |
| `resident` | Each seed runs through every stage while the transformer and decoder stay loaded; both are released at the end. | Memory holds both models together. |

Every mode runs one seed per model call, so none of them batches seeds into one tensor. `stage`
keeps each seed's latents until decoding, and `resident` needs room for the transformer and the
decoder at once. KinoMLX never switches to a different mode on its own.

The text encoder runs once per request and is released before the denoiser loads, whatever the
mode. Reference-image encodings are reused across seeds the same way, and Krea 2 editing also
reuses its source-image grounding. Each seed gets its own noise.

Models are released when the request completes, is canceled, or fails, in every mode.

### Seeds with LTX2

LTX2 makes one complete video per seed, one after another, and `--seed-mode` does not apply to
it. Each video gets its own sidecars and run record. With `--json` the command still prints one
record, whose `runs` array has an entry per seed. A restart accepts one seed.

## Output files

### Default names

Without `--output`, KinoMLX writes a timestamped file:

| Model | Default name |
| --- | --- |
| LTX2 | `ltx2-distilled-seed<seed>_<timestamp>.mp4` |
| Krea 2 | `krea2-<variant>-seed<seed>_<timestamp>.png` |
| Klein | `flux2-<variant>-seed<seed>_<timestamp>.png` |
| GMNet | `<input-stem>_<timestamp>.exr` and `<input-stem>_<timestamp>.heic` |

The timestamp is `YYYYMMDD_HHMMSS`. A name already taken gets `_2`, `_3`, and so on, so a
generated name never replaces a file. `--output-prefix` replaces the part before the timestamp.

The directory is `--output-dir`, or `[output].directory` in a config file. Without either, it is
`KINO_<MODEL>_OUTPUT_DIR` (for example `KINO_LTX2_OUTPUT_DIR`), then the shared `KINO_OUTPUT_DIR`,
then `outputs/`.

When one run makes several images, each name carries a three-digit output number and its seed,
for example `krea2-turbo-seed42_20261018_101500-002-seed43.png`. Repeated seeds and
`--images-per-prompt` variations therefore never collide.

### Exact paths and replacing files

`--output PATH` writes to exactly that path, with no timestamp. For several images, the path
becomes the stem the numbered names start from. An existing exact target is refused.

Krea 2, Klein, and GMNet accept `--force` to replace an existing exact target, and only with
`--output`. LTX2 has no replacement option: choose a new path.

### Sidecars

Sidecars are optional files written next to the output, sharing its name. Each one is off
unless you select it:

| Option | Writes | Use it to |
| --- | --- | --- |
| `--save-run-log` | `<stem>_run.json` | Read timings and settings; a restart reads it with the tensor sidecars. |
| `--save-console-log` | `<stem>_console.log` | Keep the human-readable log. |
| `--save-effective-config` | `<stem>_config.toml` | Rerun the same invocation with `--config`. |
| `--save-all-sidecars` | The model's reproducibility set | Make the run reproducible and restartable. |

`--save-all-sidecars` also selects the model's tensor sidecars, such as saved latents, which a
restart reads back. Each individual option follows it unless you set it, so
`--save-all-sidecars --no-save-console-log` saves everything but the console log.

LTX2's two frame sequences, `--save-vae-frames` and `--save-hdr-heic-frames`, are too large for
save-all and stay off unless set themselves. The model guides list their own sidecars.

A sidecar that fails to write does not fail the run. The failure is reported, and the run
record lists it under `sidecar_errors`.

## Run records and machine-readable output

The run record, `<stem>_run.json`, records the resolved settings, model files, phase timings,
memory samples, adapter results, and every file the run wrote. It is also the input to a
restart.

`--json` prints one machine-readable result on standard output instead of the live display. Its
fields are `status`, `model`, `action`, `outputs`, `elapsed_seconds`, `timings`, `diagnostics`,
`warnings`, and `sidecar_errors`, plus model details under `generation`. Read output paths from
`outputs`. Adapter results are under `generation.adapter_receipts` for LTX2,
`generation.adapters` for Krea 2, and `generation.images[].adapters` for Klein.

The run record's phase timings nest: a phase's time includes the phases inside it, so do not add
them up.

`--quiet` turns off the live display without changing what is written. `--verbose` adds
diagnostic log lines.

## Stopping a run

Ctrl-C stops a run after keeping everything it finished. Completed images, videos, and sidecars
stay at their final paths, and so do the console log and the latest run record.

The run record is marked `aborted`, or `failed` for a run that failed. The command reports
`canceled` with the outputs it published and exits with status 130.

Each output is published as soon as it is finished. A multi-image run that replaces existing
files is therefore not all-or-nothing: if it stops partway, the finished images are already
replaced and the rest keep their old contents. Write to a new name when the old set must survive
until a run completes.

## Restarting from saved results

A restart repeats only the later part of a run, reading the earlier results from its sidecars.
Save them on the original run with `--save-all-sidecars`, then pass its run record:

```bash
kinomlx --model krea2 --restart outputs/previous_run.json --restart-from decode
```

| Model | Restart points | Guide |
| --- | --- | --- |
| LTX2 | Decode either stage's latent; rerun stage 2 | [LTX2 restart](ltx2/restart.md) |
| Krea 2 | `decode`; `denoise` from saved text conditioning | [Krea 2 restart](krea2/restart.md) |
| Klein | `decode`; `denoise` from saved conditioning | [Klein restart](flux2/restart.md) |
| GMNet | None; GMNet has no denoising stages | |

Settings already baked into the saved results cannot change at a restart; each guide lists what
can. Krea 2 and Klein resume one image: select it from a multi-image run with `--restart-index`,
counting from 1 as the output names do.

A Krea 2 or Klein run you stopped with Ctrl-C can restart from each image it had published, just
like a completed run. Asking for an image it never reached is refused by name, for example
`image 2 of 3 was not published before cancellation`. A run that failed cannot be restarted.

Run records written before the current record format, run schema version 2, are refused rather
than converted.
