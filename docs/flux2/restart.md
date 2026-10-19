# Restart from saved stages

A Klein run can restart at two points:

| `--restart-from` | Reuses | Repeats |
| --- | --- | --- |
| `decode` (default) | The final latent | Decoding only |
| `denoise` | The saved conditioning: prompt, negative prompt, and references | Noise, denoising, and decoding |

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window." --save-all-sidecars
kinomlx --model flux2 --restart previous_run.json --restart-from denoise --seed 43
kinomlx --model flux2 --restart previous_run.json --restart-from decode
kinomlx --model flux2 --restart previous_run.json --restart-latents replacement.safetensors
```

- [What to save](#what-to-save)
- [What a restart can change](#what-a-restart-can-change)
- [Replace saved files](#replace-saved-files)
- [Reuse saved conditioning without a run record](#reuse-saved-conditioning-without-a-run-record)
- [Multi-image and interrupted runs](#multi-image-and-interrupted-runs)
- [Output names](#output-names)

## What to save

`--save-all-sidecars` saves the run record and both restart inputs, or select them one at a time:

| Option | Saves |
| --- | --- |
| `--save-final-latents` | The final latent and its grid, for a decode restart |
| `--save-text-conditioning` | All 512 text states, the negative states for guided base, and any encoded references, for a denoise restart |
| `--save-initial-noise` | The starting noise and its schedule |
| `--save-run-log` | The run record you pass to `--restart` |

Each option also has a negated form, such as `--no-save-initial-noise`.

A run decoded with `--vae-dtype float32` cannot be restarted: its record must name the BF16
decoder.

## What a restart can change

| Restart point | Can change | Fixed |
| --- | --- | --- |
| Decode | The output, and the latent file with `--restart-latents` | Everything that made the latent, and both precisions |
| Denoise | Seed, steps, guidance scale, schedule, noise backend, `--cfg-mode`, adapters, and KV recomputation | Prompt, size, references and their fit, variant, model files, and both precisions |

The guidance scale can change at a denoise restart, but guidance cannot be switched on or off:
the saved conditioning either has negative states or does not.

A restart reopens the model files the run recorded, which must still be present and unchanged; a
decode restart needs only the VAE. To change a precision or a model file, generate again.

## Replace saved files

`--restart-latents` decodes another final latent, which must be a Klein final latent with a valid
grid. `--text-conditioning` with `--restart-from denoise` replaces the saved conditioning, which
must match the saved prompts.

A decode restart saves only what it produces, so `--save-all-sidecars` there skips the noise and
conditioning.

## Reuse saved conditioning without a run record

`--text-conditioning` replays saved conditioning directly, with its own prompt and references:

```bash
kinomlx --model flux2 --text-conditioning previous_text.safetensors --seed 43
```

An explicit `--prompt` that conflicts with the saved one is refused, as are new references and
their fit or aspect options. Reference KV is never saved, so a replay extracts it afresh on its
first step.

## Multi-image and interrupted runs

A restart resumes one image. Pick it from a multi-image run with `--restart-index`, counting from
1 as the output names do; without an index such a run is refused, with the valid range.

A run you stopped with Ctrl-C restarts from the images it had published; see
[restarting from saved results](../USAGE.md#restarting-from-saved-results).

## Output names

A restart writes next to the source run, named after it with a `_final_decode` or `_denoise`
suffix and a new timestamp. It never inherits `--force` or the source's exact path; `--output`,
`--output-dir`, and `--output-prefix` still apply.

From Python, `Flux2Runner.restart()` does the same; see [the Python API](api.md#restart).
