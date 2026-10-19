# Krea 2: image generation and editing

Generate lossless PNG or TIFF images from text with Krea 2, or edit one or two reference images
with a compatible edit adapter. Both variants run from local model files.

```bash
kinomlx --model krea2 \
  --prompt "A ceramic cup on a walnut table, soft window light" \
  --width 1024 --height 1024 --seed 42
```

## Choose a variant

| | Turbo (default) | Raw |
| --- | --- | --- |
| What it is | The distilled model | The base model |
| Default recipe | 8 steps, no guidance | 52 steps, guidance 3.5, negative prompt |
| Speed | Fast | About 6.5 times the steps, each running twice for guidance |
| Adapters | Where Krea recommends running them | Where Krea recommends training them |

Select Raw with `--variant raw`. Krea's own guidance is to train adapters on Raw and run them on
Turbo, so a Raw-trained adapter on Turbo is the intended pairing.

## Quick start

```bash
# Raw, with a negative prompt
kinomlx --model krea2 --variant raw \
  --prompt "A quiet coastal sunrise, cinematic light" --negative-prompt "low contrast"

# Edit a photograph, keeping its aspect ratio
kinomlx --model krea2 --source-image source.png --match-source-aspect \
  --prompt "Change the jacket to blue, preserving the person and background." \
  --output edited.png

# Four steps with the community four-step adapter
kinomlx --model krea2 --turbo-4step --prompt "A painted landscape."
```

## Guides

| Page | Read it to |
| --- | --- |
| [Model files](downloads.md) | Download Raw, Turbo, and the optional decoder and adapters. |
| [Generating images](generation.md) | Set steps, guidance, schedules, prompts, and the 2x decoder. |
| [Image editing](editing.md) | Edit one or two reference images, and focus the edit on regions. |
| [Adapters](adapters.md) | Stack adapters, use the four-step and diversity profiles, and exclude layers. |
| [Restart from saved stages](restart.md) | Decode again or denoise again without repeating earlier work. |
| [Performance and precision](performance.md) | Choose denoising mode, dtypes, and decoder tiling. |
| [Python API](api.md) | Generate from Python, encode text separately, and decode yourself. |
| [Arguments](arguments.md) | Look up any argument. |

Shared behavior is in [Running a model](../USAGE.md): prompts, seed lists, seed modes, output
names, and stopping a run.

## Limits

- Adapters change the denoiser only, in the [supported formats](../ADAPTERS.md#supported-static-export-formats).
  Dynamic, timestep-dependent, and prompt-embedding adapters are not supported.
- Editing takes one or two references, one prompt, and a compatible edit adapter. There is no
  separate masked-inpainting workflow, no input of preencoded source latents, and no ComfyUI-style
  partial denoise.
- Interfaces are pre-alpha and can change between commits.

## Licenses and credits

KinoMLX does not redistribute the weights. [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)
and [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw) are governed by the terms on their model
cards, and each community adapter and the 2x decoder by its own.

KinoMLX's Krea 2 implementation follows these Apache-2.0 sources:

- The [Krea 2 reference implementation](https://github.com/krea-ai/krea-2).
- [Diffusers](https://github.com/huggingface/diffusers) and
  [Transformers](https://github.com/huggingface/transformers).
- The edit geometry of [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit).

See [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) for license texts. KinoMLX's own code
is [MIT-licensed](../../LICENSE).
