# Restart from saved stages

A restart repeats only the end of a run from its saved results. Decoding again takes seconds to
minutes; rerunning stage 2 skips text encoding and stage 1. Both avoid regenerating when you only
want to change the decoder, the post-processing, or stage 2.

- [What to save](#what-to-save)
- [Three kinds of restart](#three-kinds-of-restart)
- [What a restart can change](#what-a-restart-can-change)
- [Substitute a saved file](#substitute-a-saved-file)
- [Output names](#output-names)

## What to save

Add `--save-all-sidecars` to the original run. It saves the run record and everything a restart
reads back:

| Restart | Needs |
| --- | --- |
| Decode the final latent | `<stem>_run.json` and `<stem>.safetensors` |
| Decode the stage-1 latent | `<stem>_run.json` and `<stem>_stage1.safetensors` |
| Rerun stage 2 | `<stem>_run.json`, `<stem>_stage1.safetensors`, and `<stem>_text.safetensors` |

The model files the run used must still be available.

## Three kinds of restart

```bash
# Decode the final latent again, with another decoder and post-processing.
kinomlx --restart outputs/source_run.json \
  --video-vae diffusion --seed 43 \
  --vsr-spatial-mode balanced --target-fps 60 \
  --output outputs/redecoded.mp4

# Decode the stage-1 latent directly, skipping the upscaler and stage 2.
kinomlx --restart outputs/source_run.json \
  --latent-stage stage-1 --vsr-spatial-mode fast \
  --output outputs/stage1-preview.mp4

# Run stage 2 again from the saved stage-1 latent and prompt.
kinomlx --restart outputs/source_run.json --restart-from stage-2 \
  --output outputs/stage2-rerun.mp4
```

The default is `--restart-from decode` with the final latent. The original settings apply unless
you override them, as if the run record were a config file underneath your flags.

The stage-1 latent decodes at half the width and height. `--vsr-spatial-mode fast` upscales it
2x back to the delivery size, which makes it a quick preview of what stage 2 will refine.

## What a restart can change

Anything that already shaped the saved result is fixed; anything after it can change:

| Restart point | Can change | Fixed |
| --- | --- | --- |
| Decode | The decoder and its file, precision, tiling, and sharpening; seed and noise backend; the HDR profile; dropping audio; upscaling, frame rate, and encoding; output | Everything that shaped the latent: prompt, size, length, adapters, images, sampling, and the transformer |
| Stage 2 | All of the above, plus the stage-2 sampler, schedule, sparse attention, and dense layers; adapter and image strengths for stage 2; the HDR reference; the transformer's precision, streaming, quantization, fusion precision, and tuning; the locations of the transformer, text encoder, Gemma, and upscaler files | Prompt, size, length, frame rate, generated keyframes, prompt padding, the frame-0 mode, the model generation, and every stage-1 setting |

At a decode restart, a new seed or noise backend rerolls the diffusion decoder's texture; the
convolutional decoder reads no seed. At a stage-2 restart they reroll stage 2's noise. Adapters
and images keep their stage-1 strengths, so only their stage-2 strengths can change.

`--no-generate-audio` drops the audio of a run saved with it, but audio cannot be added to a
latent saved without it.

Changing a fixed setting is refused with its name. `kinomlx config inspect --model ltx2` lists
the decision for every field. A restart accepts one seed.

## Substitute a saved file

`--restart-latents` decodes a different latent file, and `--text-conditioning` supplies different
prompt tensors for a stage-2 restart, for example from another run. KinoMLX records the file's
hash and warns when it differs from the original, but a hash difference alone does not stop it.
The tensors must still load, be finite, and match the expected shape and dtype.

Stage-1 latents saved before KinoMLX recorded noise positions restart only with the default
`--noise-backend mlx`; regenerate stage 1 to use `torch-mps`.

A stage-2 restart of a run with [generated keyframes](generation.md#generated-keyframes) needs
stage-1 latents that carry them, which current versions save.

## Output names

Without `--output`, a restart writes next to the source video, named after it with a suffix and
a new timestamp: `<source>_stage2_decode_...` for a final decode, `<source>_stage1_decode_...`
for a stage-1 decode, and `<source>_stage2_...` for a stage-2 rerun. `--output-dir`,
`--output-prefix`, and `--output` override that as usual.

From Python, `LTX2Runner.restart()` does the same; see [the Python API](api.md#restart-from-python).
