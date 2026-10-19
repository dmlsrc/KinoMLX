# FLUX.2 Klein 9B: image generation and editing

Generate lossless PNG or TIFF images from text with Black Forest Labs' FLUX.2 Klein 9B, or edit
from ordered reference images. Prompts, references, output paths, sidecars, and restarts use the
same arguments as [Krea 2](../krea2/README.md).

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window." --seed 42
```

## Choose a variant

| `--variant` | What it is | Default recipe | Choose it for |
| --- | --- | --- | --- |
| `distilled` (default) | The four-step distilled model | 4 steps, no external guidance | Fast text-to-image and editing |
| `kv` | A separate four-step checkpoint built to cache reference keys and values | 4 steps, no external guidance | Faster editing with references |
| `base` | The undistilled model | 50 steps, guidance 4 | Guidance, negative prompts, and Turbo LoRAs |

A Turbo LoRA can run base in a few steps; see [Turbo LoRAs](adapters.md#turbo-loras-for-base).
"Turbo LoRA" names an adapter, not a variant: the separately released four-step model is
`distilled`.

## Quick start

```bash
# Edit with the KV variant, keeping the reference's aspect ratio
kinomlx --model flux2 --variant kv --source-image reference.jpg \
  --prompt-file edit.txt --match-source-aspect --height 1024 --save-all-sidecars

# Base with guidance and a negative prompt
kinomlx --model flux2 --variant base --prompt "A painted landscape." \
  --negative-prompt "blurry" --guidance 4
```

## Guides

| Page | Read it to |
| --- | --- |
| [Model files](downloads.md) | Download a variant and point KinoMLX at it. |
| [Generating images](generation.md) | Set steps, guidance, schedules, and prompts. |
| [Image editing](editing.md) | Edit from ordered references and control their fit. |
| [Adapters](adapters.md) | Stack adapters, use Turbo LoRAs on base, and exclude layers. |
| [Restart from saved stages](restart.md) | Decode again or denoise again without repeating earlier work. |
| [Performance and memory](performance.md) | Choose precision, plan reference memory, and use the KV cache. |
| [Python API](api.md) | Generate from Python and use the text and VAE components. |
| [Arguments](arguments.md) | Look up any argument. |

Shared behavior is in [Running a model](../USAGE.md): prompts, seed lists, seed modes, output
names, and stopping a run.

## Limits

- Several requested images run one after another, one per model call; there is no simultaneous
  batch.
- No VAE tiling, no restart from the middle of denoising, and none of Krea 2's grounding or region
  controls.
- Not supported: Klein 4B, FLUX.2-dev, inpainting masks, the FP8 text encoder, prompt upsampling,
  built-in moderation and watermarking, and reference KV kept across generations.
- Options that do not apply fail instead of being ignored. Interfaces are pre-alpha.

## Licenses and credits

The 9B weights are under Black Forest Labs' [FLUX Non-Commercial
License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md). Its section
2(e) requires you to filter or review outputs for unlawful or infringing content before
distributing, displaying, transmitting, or disseminating them, and to disclose AI generation or
modification where the law requires it. KinoMLX includes no content filter, so these obligations are
yours. Read the license attached to the checkpoint for its complete terms.

KinoMLX's Klein implementation follows the Apache-2.0
[Diffusers](https://github.com/huggingface/diffusers) FLUX.2 modules and component packaging, and
Diffusers and PEFT adapter conventions. See [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md)
for license texts. KinoMLX's own code is [MIT-licensed](../../LICENSE).
