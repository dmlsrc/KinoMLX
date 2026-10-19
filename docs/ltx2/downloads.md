# LTX2 model files

KinoMLX reads LTX checkpoints that are already on your Mac; it never downloads during a run.
This page lists what to download for each version, how KinoMLX finds the files, and how to
replace one component. Install KinoMLX first; see [Install](../../README.md#install).

- [Get access](#get-access)
- [LTX-2.3 files](#ltx-23-files)
- [LTX-2.5 files](#ltx-25-files)
- [Download into the Hugging Face cache](#download-into-the-hugging-face-cache)
- [Download into your own folders](#download-into-your-own-folders)
- [Replace one component](#replace-one-component)
- [FP8 transformer files](#fp8-transformer-files)
- [Check the files](#check-the-files)

## Get access

The LTX releases are gated on Hugging Face. Sign in, open each repository you need, read its
model card and license, and accept or request access. Signing in to the `hf` CLI does not grant
access by itself.

| Repository | Needed for |
| --- | --- |
| [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | LTX-2.3 |
| [Lightricks/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized) | LTX-2.3's default text encoder |
| [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | LTX-2.5 |
| [Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8) | Optional FP8 transformer files |
| [Lightricks/LTX-2.3-22b-IC-LoRA-HDR](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR) | Optional [SDR-to-HDR conversion](hdr.md) |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | Optional `--gemma-variant plain` |

## LTX-2.3 files

| File | Requirement |
| --- | --- |
| `ltx-2.3-22b-distilled-1.1.safetensors` | Required: the checkpoint, including its VAEs and audio |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Required: the latent upscaler between the two stages |
| The paired Gemma 3 repository, complete | Required: the text encoder |
| `ltx-2.3-22b-dev.safetensors` | Optional: the full dev transformer |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | Optional: the distilled LoRA that pairs with the dev transformer |

The dev transformer can run KinoMLX's distilled recipe, but it was not distilled for it.
Lightricks' recipe pairs it with its distilled LoRA; see
[the dev-transformer recipe](adapters.md#run-a-dev-transformer).

## LTX-2.5 files

LTX-2.5 is a pack of separate components. A run reads only the ones it uses:

| Component | Path in the repository | Requirement |
| --- | --- | --- |
| Distilled transformer and connectors | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | Required |
| Gemma 4 text encoder and projection | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | Required |
| Convolutional video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | Required by the default decoder |
| Spatial x2 latent upscaler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | Required |
| Audio VAE and vocoder | `vae/ltx-2.5-audio-vae-bf16.safetensors` | With `--generate-audio` |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | With `--auto-duration` |
| Diffusion video VAE | `vae/ltx-2.5-video-vae-bf16.safetensors` | With `--video-vae diffusion` |
| Dev transformer | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | Optional |
| Distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | Optional, for the dev transformer |
| Temporal x2 latent upscaler | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | Optional, for the Python component API |

The pack's INT8 and NVFP4 files are other formats, not substitutes for these; skip them when
downloading.

## Download into the Hugging Face cache

With access approved, the `hf` CLI keeps the files in its cache, where KinoMLX finds them
without paths:

```bash
hf auth login

# LTX-2.3: checkpoint, upscaler, and text encoder
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-1.1.safetensors \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors
hf download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized

# Optional: the dev transformer and its distilled LoRA
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-dev.safetensors \
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors

# Optional: Google's Gemma 3, for --gemma-variant plain
hf download google/gemma-3-12b-it

# LTX-2.5: the pack without its INT8 and NVFP4 files
hf download Lightricks/LTX-2.5 --exclude "*int8*" --exclude "*nvfp4*"
```

KinoMLX then finds the cached files with no paths: the LTX-2.3 checkpoint, its upscaler, and
the paired Gemma by default, and the LTX-2.5 pack with `--model-generation 2.5`. Paths are only
needed for files kept elsewhere.

## Download into your own folders

The `hf` CLI is optional. You can download through each model page's Files and versions tab
into any folder, or have `hf` download into one with `hf download --local-dir`:

```bash
hf download Lightricks/LTX-2.5 --local-dir /models/ltx25 \
  --exclude "*int8*" --exclude "*nvfp4*"
```

For LTX-2.3, pass the checkpoint, and the Gemma folder with `--text-encoder-path`. The upscaler
is found next to the checkpoint, so `--spatial-upscaler-path` is needed only when it is
elsewhere. The Gemma folder must contain `config.json`, the tokenizer files, and every model file:

```bash
kinomlx --model ltx2 \
  --weights-path /models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upscaler-path /models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-encoder-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

For LTX-2.5, pass the folder once. It may keep the repository's subfolders or hold the files
directly, under their original names:

```bash
kinomlx --model ltx2 --model-generation 2.5 --weights-path /models/ltx25 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

KinoMLX then finds the audio VAE, duration head, and diffusion decoder in that folder when a run
needs them.

## Replace one component

`--weights-path` is the baseline: the LTX-2.3 checkpoint or the LTX-2.5 pack. A component path
replaces only that component, and everything else still comes from the baseline. This run keeps
the LTX-2.5 pack but swaps in the dev transformer:

```bash
kinomlx --model ltx2 --model-generation 2.5 --weights-path /models/ltx25 \
  --transformer-path /models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

| Option | Replaces |
| --- | --- |
| `--transformer-path` | The transformer and its audio/video connectors |
| `--text-encoder-path` | The text encoder: LTX-2.3's Gemma 3 folder, or LTX-2.5's prepared file with its tokenizer data and projection |
| `--video-vae-path` | The video decoder; it overrides `--video-vae` |
| `--audio-vae-path` | The audio VAE and vocoder |
| `--spatial-upscaler-path` | The spatial x2 latent upscaler |
| `--temporal-latent-upscaler-path` | The temporal x2 latent upscaler |
| `--duration-head-path` | The automatic-duration head |

An explicit path always wins. For LTX-2.5 without `--weights-path`, missing components come from the
pack next to an explicit `--transformer-path`, then from a cached pack. Some parts travel with
others and have no path of their own: the connectors with the transformer, the tokenizer and
projection with the text encoder, and the vocoder with the audio VAE.

Before loading any weights, KinoMLX checks every component against the transformer: text width,
latent channels, connector shapes, upscaler kind, and generation. A mismatched file fails with
its name, instead of producing a broken video.

## FP8 transformer files

KinoMLX reads transformers stored in E4M3 FP8, such as the
[LTX-2.3 FP8 release](https://huggingface.co/Lightricks/LTX-2.3-fp8). FP8 only saves disk space
here: the weights are converted to the transformer dtype (FP16 by default) as they are used.
The files are optional:

```bash
hf download Lightricks/LTX-2.3-fp8 \
  ltx-2.3-22b-dev-fp8.safetensors \
  ltx-2.3-22b-distilled-fp8.safetensors \
  --local-dir /models/ltx23-fp8
```

Use one as the baseline like any LTX-2.3 checkpoint:

```bash
kinomlx --model ltx2 \
  --weights-path /models/ltx23-fp8/ltx-2.3-22b-distilled-fp8.safetensors \
  --text-encoder-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upscaler-path /models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --frames 121
```

E4M3 with scalar weight scales is supported. E5M2 is refused, and the LTX-2.5 INT8 and NVFP4
files are not FP8 sources.

## Check the files

`--dry-run` finds every file a run would read, checks them against each other, and exits without
loading weights:

```bash
kinomlx --model ltx2 --dry-run
kinomlx --model ltx2 --model-generation 2.5 --dry-run --json
```

The JSON form adds each component's graph identity and dtype.
