# Restart from saved stages

Krea 2 has one denoising stage, so a run can restart at two points:

| `--restart-from` | Reuses | Repeats |
| --- | --- | --- |
| `decode` (default) | The final latent | Decoding only |
| `denoise` | The encoded prompt | Noise, denoising, and decoding |

A decode restart tries another decoder without denoising again. A denoise restart tries a new
seed, steps, or adapters without encoding the prompt again.

- [What to save](#what-to-save)
- [Decode again](#decode-again)
- [Denoise again](#denoise-again)
- [Reuse an encoded prompt without a run record](#reuse-an-encoded-prompt-without-a-run-record)
- [Multi-image and interrupted runs](#multi-image-and-interrupted-runs)
- [Substituted files](#substituted-files)

## What to save

`--save-all-sidecars` saves the run record and both restart inputs. To save only what you need:

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise, cinematic light" \
  --save-text-conditioning --save-final-latents --save-run-log
```

The run record, `<stem>_run.json`, is what you pass to `--restart`. The model files the run used
must still be available.

## Decode again

```bash
kinomlx --model krea2 --restart /path/to/source_run.json --vae-decoder upscale2x
```

This decodes the saved final latent without loading the text encoder or the transformer. The
original settings apply unless you override them, so omitting `--vae-decoder` reuses the original
decoder. `--vae-decoder`, `--vae-path`, `--vae-dtype`, and the tiling options can all change here.
A new `--vae-path` is detected as usual unless you also name a decoder, and a path that
contradicts the named decoder is refused.

The image is written next to the source run with a `_final_decode` suffix and a new timestamp,
unless `--output`, `--output-dir`, or `--output-prefix` says otherwise.

A decode restart saves the latent it read again, so its own output can be restarted in turn. It
makes no new noise or text conditioning: `--save-all-sidecars` there means the run log, console
log, effective config, and final latents, and an explicit `--save-initial-noise` or
`--save-text-conditioning` is refused.

## Denoise again

```bash
kinomlx --model krea2 --restart /path/to/source_run.json --restart-from denoise --seed 99
```

This skips the tokenizer and text encoder. These can change: the seed, noise backend, size, steps,
guidance, `mu`, schedule, adapters and profiles, transformer precision and denoising mode, decoder,
and output.

These are fixed: the variant, the prompts and the system and negative prompts, editing inputs, a
seed list, `--images-per-prompt`, `--seed-mode`, `--prompt-unconditional`, and the tokenizer, text
encoder, and transformer files. The saved conditioning already embodies them.

The image is written next to the source run with a `_denoise` suffix. The restart saves new noise
and final latents, and the text conditioning it read, again, so it can be restarted in turn.

An edit cannot restart at `denoise`: its saved data lacks the encoded references. Decode restarts
of edits work as usual.

## Reuse an encoded prompt without a run record

`--text-conditioning` replays a saved encoded prompt directly, with no run record. The file
carries its prompts, so `--prompt` can be omitted:

```bash
kinomlx --model krea2 --text-conditioning /path/to/source_text.safetensors --seed 99
```

Each generated image has its own conditioning file; give it a seed list to make new variations of
that prompt.

A file holding several prompt rows makes one image per row when `--prompt` is omitted, and a seed
list multiplies them. A replay must make exactly one image per row, so one `--prompt` with
`--images-per-prompt` above 1 against such a file is refused; use the per-image file instead.

Direct replay requires the same tokenizer, text encoder, dtype, and layers as the file recorded,
and the same `--system-prompt`. Raw or Turbo does not matter, because the variant does not affect
text encoding. A file without a negative branch cannot drive a guided run.

## Multi-image and interrupted runs

A run that made several images restarts one of them. Pick it with `--restart-index`, counting
from 1 as the output names do; the restart takes that image's prompt and seed:

```bash
kinomlx --model krea2 --prompt "A lighthouse at sunset" --seed 42 99 --save-all-sidecars
kinomlx --model krea2 --restart previous_run.json --restart-from decode --restart-index 2
```

A multi-image run without an index is refused, with the valid range. A run you stopped with
Ctrl-C restarts from the images it had published; see
[restarting from saved results](../USAGE.md#restarting-from-saved-results).

## Substituted files

`--restart-latents` decodes another final-latent file with the source run's settings, and
`--text-conditioning` supplies another encoded prompt to a denoise restart:

```bash
kinomlx --model krea2 --restart /path/to/source_run.json \
  --restart-latents /path/to/edited_final_latents.safetensors --output /path/to/decoded.png
```

KinoMLX records each file's SHA-256 and warns when it differs from the original run's, but a
different hash alone does not stop the restart. The tensors must still match: one finite BF16,
FP16, or FP32 latent of the recorded shape, or conditioning with the same batch, token count,
layers, width, and positions. An arbitrary embedding file is not interchangeable with Krea text.

A substitute also works after the original sidecar is deleted, because the run record keeps the
prompts.

A restart's own run record can seed a further restart. When an input is not among that run's
outputs, KinoMLX uses the input it recorded consuming, without needing the earlier run records;
`--restart-latents` still takes precedence over both.
