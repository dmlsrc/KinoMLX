# LTX2: video generation

Generate video, with optional stereo audio, from a text prompt, images, or both, using Lightricks'
LTX-2.3 or LTX-2.5. Output is an HEVC MP4. KinoMLX runs the distilled two-stage recipe: eight
steps at half resolution, a 2x latent upscale, then three steps at full resolution.

```bash
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 --generate-audio
```

## Choose a version

LTX-2.3 is the default; select LTX-2.5 with `--model-generation 2.5`. Both make video from text
and images, with or without audio, and both take adapters and restart from saved stages. They
differ in these features:

| Feature | LTX-2.3 | LTX-2.5 |
| --- | --- | --- |
| [Automatic length](generation.md#length-and-frame-rate) (`--auto-duration`) | No | Yes |
| [Generated keyframes](generation.md#generated-keyframes) (`--generated-keyframes`) | No | Yes |
| [HDR from an SDR video](hdr.md#ltx-23-convert-sdr-video-to-hdr) | Yes, with the HDR IC-LoRA | No |
| [HDR from an HDR still](hdr.md#ltx-25-animate-an-hdr-still) | No | Yes |
| [Diffusion video decoder](decode.md#choose-a-decoder) (`--video-vae diffusion`) | Yes, using LTX-2.5's decoder | Yes |
| Default sampler | Deterministic in both stages | Ancestral in both stages |

Text-only HDR is refused by both versions.

## Hardware

LTX2 runs on Apple Silicon with 32 GB of unified memory or more. Below 64 GB, add
`--stream-transformer` and start with a modest size such as 768x448; see
[performance](performance.md#run-on-32-gb). The text encoder and the transformer load one after
the other, never together.

## Guides

| Page | Read it to |
| --- | --- |
| [Model files](downloads.md) | Download the checkpoints and point KinoMLX at them. |
| [Generating video](generation.md) | Set size, length, sampling, and audio, and use config files. |
| [Image conditioning](images.md) | Pin the first, last, or any frame to an image. |
| [Adapters](adapters.md) | Add LoRAs, set per-stage strengths, and leave parts of the model alone. |
| [Video decoding](decode.md) | Choose a decoder, its precision, and its memory plan. |
| [Upscaling, frame rate, and encoding](output.md) | Upscale, interpolate, encode, and name the output. |
| [Restart from saved stages](restart.md) | Decode again or rerun stage 2 without starting over. |
| [HDR video](hdr.md) | Make HDR video from an SDR video or an HDR still. |
| [Performance and memory](performance.md) | Fit a run in memory, speed it up, and estimate run times. |
| [Python API](api.md) | Generate from Python and consume frames directly. |
| [Arguments](arguments.md) | Look up any argument. |

Shared behavior is in [Running a model](../USAGE.md): prompts, seed lists, output names, run
records, and stopping a run.

## Limits

- Distilled two-stage generation only. Dev-model guidance (CFG/STG), DFR, and general
  video-to-video are not implemented; the one video-to-video workflow is LTX-2.3's SDR-to-HDR
  conversion.
- Width and height are multiples of 64. Frame counts are `8*k+1`, and `--duration` rounds up to
  the next one.
- Apple Silicon Macs only. Interfaces and defaults can change between commits.

## Licenses and credits

KinoMLX does not redistribute model weights; download them under their own licenses:

- **LTX-2.3** and its **HDR IC-LoRA**, by
  [Lightricks](https://huggingface.co/Lightricks/LTX-2.3), under the LTX-2 Community License
  Agreement. Non-commercial use is free; entities with annual revenue of $10M or more need a
  separate commercial license from Lightricks.
- **LTX-2.5**, by [Lightricks](https://huggingface.co/Lightricks/LTX-2.5), under its published
  model license.
- **Gemma 3 12B**, by Google: Lightricks'
  [paired QAT-unquantized release](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
  by default, or Google's [instruction-tuned release](https://huggingface.co/google/gemma-3-12b-it).
- **Gemma 4 12B**, by [Google](https://huggingface.co/google/gemma-4-12B-it), shipped as a
  prepared text encoder inside the LTX-2.5 pack.

Both Gemma models are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

KinoMLX contains no code from the Lightricks LTX-2 repository. Its LTX2 implementation is built
from checkpoint metadata, the published architecture, and these sources:

- [Diffusers](https://github.com/huggingface/diffusers) LTX-2 modules, written jointly by
  Lightricks and Hugging Face (Apache-2.0).
- [Transformers](https://github.com/huggingface/transformers) Gemma implementations (Apache-2.0).
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT).
- The Sol-Attn routing of NVIDIA's [Sana Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine)
  (Apache-2.0).

See [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) for license texts. KinoMLX's own code
is [MIT-licensed](../../LICENSE).
