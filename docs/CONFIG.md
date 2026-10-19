# Configuration files and environment variables

Most options can be set five ways, and a later source overrides an earlier one:

```text
built-in default < environment variable < TOML config file < command-line flag < --set
```

Config files, environment variables, `--set`, and Python requests share one set of field names,
so they describe a run the same way; flags usually spell the same names with dashes. This page
shows how to write and inspect config files, how environment variables are named, and the
arguments every model shares.

- [Start from a template](#start-from-a-template)
- [Run from a config file](#run-from-a-config-file)
- [See or save what a run will use](#see-or-save-what-a-run-will-use)
- [Change one value with --set](#change-one-value-with---set)
- [Environment variables](#environment-variables)
- [Look up one field](#look-up-one-field)
- [Names that differ from the flag](#names-that-differ-from-the-flag)
- [Shared arguments](#shared-arguments)

## Start from a template

`kinomlx config init` writes an annotated config file for one model. It lists every field with
its accepted values, its default, and its environment variable, so it is the quickest way to find
the TOML name of a flag:

```bash
kinomlx config init --model krea2 --output krea2.toml
```

It refuses to replace an existing file. LTX2's prompt is written as an editable multi-line
string.

## Run from a config file

```bash
kinomlx --config run.toml
kinomlx --config run.toml --seed 7
```

Flags on the command line still override the file. A config file can select its model with a
top-level `model = "krea2"`, so `--model` is optional. Settings are grouped into tables:

| Table | Holds |
| --- | --- |
| `[generate]` | The request: prompt, size, seeds, adapters, image conditions. GMNet uses `[expand]`. |
| `[output]` | Output paths, names, and sidecars, and LTX2's upscaling, frame rate, cut detection, and encoding. |
| `[model_settings]` | Model files, variant, precision, and runtime choices. |
| `[model_artifacts]` | LTX2 tensor sidecars. |
| `[settings]` | Infrastructure shared by every model, such as memory limits. |
| `[restart]` | The run record and stage to restart from. |
| `[operation]` | `dry_run`. |

Unknown fields and invalid values are refused before any model loads, with the closest valid
names suggested, so a misspelled setting never falls back to a default silently.

## See or save what a run will use

`--print-config` resolves defaults, environment variables, the config file, flags, and `--set`
into one TOML document, prints it, and exits without loading anything. `--save-config PATH`
writes the same document to a new file:

```bash
kinomlx --config run.toml --set generate.seed=7 --print-config
kinomlx --config run.toml --save-config selected.toml
kinomlx --config run.toml --save-config overrides.toml --only-non-defaults
```

Loading the saved file reproduces the invocation. Built-in defaults appear as comments for
reference and are not applied on reload; `--only-non-defaults` leaves them out. Values that came
from environment variables stay in the file, so it remains reproducible on another shell.

The document records what you asked for, not what the run computed. A duration stays a duration,
for example, rather than becoming a frame count.

## Change one value with --set

`--set KEY=VALUE` changes one field after everything else has been applied. Values use TOML
syntax: `0.35` is a number, `true` a boolean, `"text"` a string, and `[1, 2]` an array. Repeated
`--set` flags apply in order.

```bash
kinomlx --config run.toml --set generate.seed=7
kinomlx --config run.toml --set 'generate.adapters[1].strength=0.25'
kinomlx --config run.toml --set 'generate.adapters[1]={path="detail.safetensors"}'
kinomlx --config run.toml --set 'generate.adapters=[]'
```

An index edits one item of a list and keeps its siblings. Replacing a whole item drops the
members it leaves out, and `[]` clears the list.

A list from a later source replaces the earlier list rather than adding to it. `--set
model="krea2"` also selects the model.

## Environment variables

Model options are named `KINO_<MODEL>_<FIELD>`, from the model name and the field's config
name, which can differ from the flag:

| Flag | Environment variable |
| --- | --- |
| `--model ltx2 --vae-decode-dtype float16` | `KINO_LTX2_VAE_DECODE_DTYPE=float16` |
| `--model flux2 --seed 42` | `KINO_FLUX2_SEED=42` |
| `--model krea2 --output image.png` | `KINO_KREA2_OUTPUT_PATH=image.png` |

Shared infrastructure options use names without a model, such as `KINO_MLX_CACHE_LIMIT_GB`.
Record lists such as adapters take a TOML array, for example
`KINO_KREA2_ADAPTERS='[{path="style.safetensors", strength=0.8}]'`.

A variable in the selected model's namespace must name a current field: an unknown
`KINO_<MODEL>_` variable is an error even when it is empty or a flag overrides it. The template
from `kinomlx config init` lists every variable a model reads.

The model and the config-file controls (`--config`, `--set`, `--print-config`, `--save-config`,
and `--only-non-defaults`) have no environment variables.

## Look up one field

`kinomlx config inspect` prints a model's configuration inventory as JSON, without reading config
files or weights. Select one field by its dotted name:

```bash
kinomlx config inspect --model ltx2 --field generate.seed
```

Each entry lists the field's flags, environment variable, type, default, and how its value is
normalized. It also says where the Python API takes it and whether a restart may change it, with
the reason. GMNet, which has no restart, reports `no_restart` and says why.

## Names that differ from the flag

These shared arguments have config and environment names you cannot read off the flag. Each
model's arguments page lists its own.

| Flag | Config field | Environment variable |
| --- | --- | --- |
| `--output` | `output.path` | `KINO_<MODEL>_OUTPUT_PATH` |
| `--output-dir` | `output.directory` | `KINO_<MODEL>_OUTPUT_DIR`, then `KINO_OUTPUT_DIR` |
| `--output-prefix` | `output.prefix` | `KINO_<MODEL>_OUTPUT_PREFIX` |
| `--restart` | `restart.run` | `KINO_<MODEL>_RESTART_RUN` |
| `--restart-from` | `restart.stage` | `KINO_<MODEL>_RESTART_STAGE` |
| `--restart-latents` | `restart.latents` | `KINO_<MODEL>_RESTART_LATENTS` |
| `--restart-index` | `restart.index` | `KINO_<MODEL>_RESTART_INDEX` |
| `--dry-run` | `operation.dry_run` | `KINO_<MODEL>_DRY_RUN` |
| `--json` | `settings.json_output` | `KINO_JSON` |
| `--hf-home` | `settings.hf_home` | `HF_HOME` |
| `--app-cache-dir` | `settings.app_cache_dir` | `KINO_APP_CACHE_DIR`, then `XDG_CACHE_HOME/KinoMLX` |

## Shared arguments

These arguments mean the same thing for every model that takes them, and each is listed here
once. Each model's arguments page lists the rest, including `--dry-run`, whose checks depend on
the model.

The converters under `kinomlx weights` take the memory, logging, profiling, and location
arguments, but not the config-file ones or `--check-adapter-weights`.

### Config files and inspection

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--config</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Load a TOML config file. Flags and <code>--set</code> still override it.</td></tr>
<tr><td><code>--set</code></td><td><em>none</em></td><td><code>KEY=VALUE</code></td><td>Change one field last, in TOML syntax. Repeat it to change several, in order.</td></tr>
<tr><td><code>--print-config</code></td><td><em>off</em></td><td><em>flag</em></td><td>Print the resolved invocation as TOML and exit without loading a model.</td></tr>
<tr><td><code>--save-config</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Save the resolved invocation to a new TOML file and exit; an existing path is refused.</td></tr>
<tr><td><code>--only-non-defaults</code></td><td><em>off</em></td><td><em>flag</em></td><td>With <code>--print-config</code> or <code>--save-config</code>, leave out the commented built-in defaults.</td></tr>
<tr><td rowspan="4" valign="top"><code>--model</code></td><td rowspan="4" valign="top"><code>ltx2</code></td><td><code>ltx2</code></td><td>LTX2 video.</td></tr>
<tr><td><code>krea2</code></td><td>Krea 2 stills.</td></tr>
<tr><td><code>flux2</code></td><td>FLUX.2 Klein stills.</td></tr>
<tr><td><code>gmnet</code></td><td>GMNet HDR expansion. A config file can select the model instead, with <code>model = "..."</code>.</td></tr>
</tbody>
</table>

### The config command

`kinomlx config init` and `kinomlx config inspect` take these arguments:

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--model</code></td><td><code>ltx2</code></td><td><code>ltx2</code>, <code>krea2</code>, <code>flux2</code>, <code>gmnet</code></td><td>The model to describe.</td></tr>
<tr><td><code>--output</code></td><td><code>kino-config.toml</code></td><td><code>PATH</code></td><td><code>config init</code> only: where to write the template. An existing file is refused.</td></tr>
<tr><td><code>--field</code></td><td><em>every field</em></td><td><code>DOTTED.NAME</code></td><td><code>config inspect</code> only: print one field, such as <code>generate.seed</code>.</td></tr>
</tbody>
</table>

### Memory

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--mlx-cache-limit-gb</code></td><td><code>1</code></td><td><code>GiB</code></td><td>How much freed memory MLX keeps for reuse. <code>0</code> keeps none; it does not limit the memory a model uses. LTX2's decoder chooses its own unless <code>--vae-cache-limit-gb</code> is set. See <a href="PERFORMANCE.md#allocator-cache">allocator cache</a>.</td></tr>
<tr><td><code>--mlx-wired-limit-gb</code></td><td><em>device recommendation</em></td><td><code>GiB</code></td><td>How much MLX memory stays resident instead of being compressed or paged out. <code>0</code> disables wiring, and a value above the device limit is refused. See <a href="PERFORMANCE.md#memory-residency">memory residency</a>.</td></tr>
</tbody>
</table>

### Checks, logging, and profiling

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--check-adapter-weights</code>, <code>--no-check-adapter-weights</code></td><td><em>on</em></td><td><em>flag</em></td><td>LTX2, Krea 2, and Klein: check adapter-modified weights for nonfinite values after their first use. Turning it off keeps the per-step latent checks. See <a href="ADAPTERS.md#first-use-weight-checks">first-use checks</a>.</td></tr>
<tr><td><code>--json</code>, <code>--no-json</code></td><td><em>off</em></td><td><em>flag</em></td><td>Print one machine-readable result on standard output instead of the live display. See <a href="USAGE.md#run-records-and-machine-readable-output">run records</a>.</td></tr>
<tr><td><code>--quiet</code>, <code>--no-quiet</code></td><td><em>off</em></td><td><em>flag</em></td><td>Turn off the live display. Output files are unchanged.</td></tr>
<tr><td><code>--verbose</code>, <code>--no-verbose</code></td><td><em>off</em></td><td><em>flag</em></td><td>Add diagnostic log lines.</td></tr>
<tr><td><code>--profile-signposts</code>, <code>--no-profile-signposts</code></td><td><em>off</em></td><td><em>flag</em></td><td>Emit signposts that Instruments shows on its timeline, marking each phase of the run.</td></tr>
<tr><td><code>--profile-signpost-log</code></td><td><em>none</em></td><td><code>PATH</code></td><td>With <code>--profile-signposts</code>, also save the signposts to this log.</td></tr>
</tbody>
</table>

### Locations

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--hf-home</code></td><td><code>~/.cache/huggingface</code></td><td><code>PATH</code></td><td>The Hugging Face state folder. Downloaded models are found in its <code>hub</code> folder.</td></tr>
<tr><td><code>--app-cache-dir</code></td><td><code>~/Library/Caches/KinoMLX</code></td><td><code>PATH</code></td><td>Where converted GMNet weights and the profiling helper are stored outside a source checkout.</td></tr>
</tbody>
</table>
