# Krea 2 model files

KinoMLX reads Krea 2 from files already on your Mac; it never downloads during a run. This page
lists what to download and how KinoMLX finds it.

- [Get access](#get-access)
- [Download Turbo and Raw](#download-turbo-and-raw)
- [Optional files](#optional-files)
- [How files are found](#how-files-are-found)
- [Replace one component](#replace-one-component)

## Get access

Sign in to Hugging Face, open [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) and
[Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw), and read their model cards and terms. Accept
or request access where a repository is gated; signing in to the `hf` CLI does not grant access
by itself.

## Download Turbo and Raw

Download either variant, or both:

```bash
hf download krea/Krea-2-Turbo --exclude turbo.safetensors
hf download krea/Krea-2-Raw --exclude raw.safetensors
```

The exclusion skips only the root monolith, a second copy of the transformer in another layout.
The rest of each repository is what KinoMLX reads. Download the monolith only to pass it as an
explicit `--transformer-path` or for an outside comparison.

## Optional files

| File | Repository | Use |
| --- | --- | --- |
| `krea2_identity_edit_v1_2_r128.safetensors` | [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) | The default [edit adapter](editing.md) |
| `krea2_turbo_4step_rank_64_lora.safetensors` | [lvladikov/Krea2-Turbo-Distill-4step-LoRA](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-4step-LoRA) | [`--turbo-4step`](adapters.md#four-step-turbo) |
| `krea2_turbo_sda_v1.0_comfy.safetensors` | [F16/krea2-turbo-sda](https://huggingface.co/F16/krea2-turbo-sda) | [`--turbo-sda`](adapters.md#seed-diversity-sda) |
| `Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors` | [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) | [`--vae-decoder upscale2x`](generation.md#decode-at-twice-the-size) |

Download exactly these files into the Hugging Face cache, where their options find them:

```bash
hf download conradlocke/krea2-identity-edit krea2_identity_edit_v1_2_r128.safetensors
hf download lvladikov/Krea2-Turbo-Distill-4step-LoRA krea2_turbo_4step_rank_64_lora.safetensors
hf download F16/krea2-turbo-sda krea2_turbo_sda_v1.0_comfy.safetensors
hf download spacepxl/Wan2.1-VAE-upscale2x Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors
```

The SDA repository also ships a Diffusers-style conversion that loses 32 of its 264 targets;
`--turbo-sda` uses only the file above. The 2x decoder repository also carries a Diffusers copy of
the same decoder, which is not needed.

## How files are found

KinoMLX reads each repository's local `main` snapshot in the Hugging Face cache selected by
`HF_HOME` or `--hf-home`. It never falls back to the network or to an older snapshot. Turbo is the
default; `--variant raw` selects Raw.

`--source-path` uses a local folder instead of the cache, such as one made with
`hf download --local-dir`:

```bash
kinomlx --model krea2 --source-path /path/to/Krea-2-Turbo --prompt "A painted landscape."
```

`--dry-run` finds every file a run would read, checks them, and exits without loading weights:

```bash
kinomlx --model krea2 --dry-run --variant raw
kinomlx --model krea2 --dry-run --source-path /path/to/Krea-2-Turbo --json
```

The JSON form lists each component's path, identity, dtype, and layout, and the recipe a run
would use.

## Replace one component

`--tokenizer-path`, `--text-encoder-path`, `--transformer-path`, and `--vae-path` replace one
component and keep the rest. Each is checked against the others before loading. Components load
from their original files and prepare their layouts in memory, so no converted copy is written to
disk.

Keep each component complete: editing also uses the VAE's encoder and the text encoder's vision
model.
