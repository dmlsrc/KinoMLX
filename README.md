# KinoMLX

Native multimodal inference on Apple Silicon through
[MLX](https://github.com/ml-explore/mlx). KinoMLX runs four model families through one CLI and
Python APIs:

- **LTX2**: LTX-2.3 and LTX-2.5 text/image-to-video generation, optional stereo audio, HDR
  workflows, adapters, and staged restart.
- **Krea 2**: Raw and Turbo text-to-image generation, reference editing with a compatible
  adapter, lossless PNG/TIFF output, and saved-stage restart.
- **FLUX.2 Klein 9B**: base, distilled, and KV text/image generation, ordered reference editing,
  adapters, lossless PNG/TIFF output, and saved-stage restart.
- **GMNet**: SDR-to-HDR still expansion with EXR and PQ HEIC output and optional gain maps.

KinoMLX is an independent implementation built from published architecture descriptions and
implementations, checkpoint metadata, and independent validation. You do not need the original
model implementations to run it.

## Status

Pre-alpha. All four families have end-to-end CLI and Python workflows, and interfaces and
defaults can change between commits.

| Model | Input and output | Python entry point | Guide |
| --- | --- | --- | --- |
| `ltx2` | Text and optional images -> video and optional audio | `generate_ltx2()` / `LTX2Runner` | [LTX2](docs/ltx2/README.md) |
| `krea2` | Text and optional references -> PNG/TIFF stills | `generate_krea2()` / `Krea2Runner` | [Krea 2](docs/krea2/README.md) |
| `flux2` | Text and ordered references -> PNG/TIFF stills | `generate_flux2()` / `Flux2Runner` | [Klein](docs/flux2/README.md) |
| `gmnet` | SDR still -> HDR still files | `expand_gmnet()` / `GMNetRunner` | [GMNet](docs/gmnet/README.md) |

## Requirements

- Apple Silicon (M1 family or later)
- macOS 27 or later
- Python 3.15
- Enough unified memory for the model and output size

LTX2 runs on 32 GB with transformer streaming; 64 GB is recommended for its default settings and
larger videos. Krea 2 and Klein load one model at a time by default, but their memory still grows
with image size and reference count. GMNet needs much less and scales with the input image.

## Install

```bash
git clone https://github.com/dmlsrc/KinoMLX.git
cd KinoMLX
uv venv --python 3.15
source .venv/bin/activate
uv pip install -e .
kinomlx --help
```

KinoMLX never downloads model weights, and none are bundled. Each model guide's model files page
lists the files to get and the exact commands. LTX2, Krea 2, and Klein come from Hugging Face;
GMNet publishes its checkpoints on GitHub.

The Hugging Face downloads use the optional `hf` CLI, which you can install as an isolated uv
tool:

```bash
uv tool install hf
hf auth login
```

If `hf` is not on `PATH` afterward, run `uv tool update-shell` and open a new shell. The
[Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) also
documents `uvx hf`. Signing in does not grant access to a gated model: accept or request access
on its model page first. You can also download the files through the model pages instead.

## Quick start

Once the model files are downloaded:

```bash
# LTX-2.3 video with generated audio
kinomlx --model ltx2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 --generate-audio

# Krea 2 Turbo still
kinomlx --model krea2 \
  --prompt "A ceramic cup on a walnut table, soft window light" \
  --width 1024 --height 1024 --seed 42

# FLUX.2 Klein distilled still
kinomlx --model flux2 \
  --prompt "A ceramic cup on a walnut table, soft window light" \
  --width 1024 --height 1024 --seed 42

# GMNet HDR expansion of a photograph, using converted weights
kinomlx --model gmnet --image photo.jpg --output-dir hdr/ --save-gain-map
```

`kinomlx --model MODEL --help` lists a model's options, and
`kinomlx config init --model MODEL --output settings.toml` writes an annotated config file with
every option and its default.

## Documentation

Each model has a folder of guides; start with its README:

| Model | Start here | Every argument |
| --- | --- | --- |
| LTX2 | [Overview](docs/ltx2/README.md) | [Arguments](docs/ltx2/arguments.md) |
| Krea 2 | [Overview](docs/krea2/README.md) | [Arguments](docs/krea2/arguments.md) |
| FLUX.2 Klein | [Overview](docs/flux2/README.md) | [Arguments](docs/flux2/arguments.md) |
| GMNet | [Overview](docs/gmnet/README.md) | [Arguments](docs/gmnet/arguments.md) |

Guides shared by the models:

- [Running a model](docs/USAGE.md): prompts, seeds, output files, run records, stopping, and
  restarting.
- [Configuration](docs/CONFIG.md): config files, environment variables, `--set`, and the
  arguments every model shares.
- [Adapters](docs/ADAPTERS.md): formats, strength, alpha, exclusions, and checks.
- [Denoise schedules](docs/SCHEDULES.md): setting noise levels and varying adapter strength over
  a run.
- [Model files and loading](docs/WEIGHTS.md): where models load from and what loading costs.
- [Performance and memory](docs/PERFORMANCE.md): memory controls, High Power, precision, and
  comparing runs.
- [Changelog](CHANGELOG.md): user-facing changes.

Contributors: [Project conventions](docs/CONVENTIONS.md) covers code, documentation, and
testing rules, and [Project structure](docs/STRUCTURE.md) maps the package.

## Attribution and license

KinoMLX builds on Apple's [MLX](https://github.com/ml-explore/mlx) framework and on published
architecture implementations and checkpoints from Lightricks, the GMNet authors, Krea, Black
Forest Labs, Hugging Face, and Google. Model weights are not redistributed and remain governed by
their upstream terms; each model guide names its licenses.

KinoMLX's own code is [MIT-licensed](LICENSE). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for included third-party license texts and
notices.
