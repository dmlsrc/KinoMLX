# KinoMLX

Native multimodal inference on Apple Silicon through
[MLX](https://github.com/ml-explore/mlx). KinoMLX currently provides two focused
model families behind one CLI and one public-library style:

- distilled LTX-2.3 and LTX-2.5 SDR text/image-to-video generation, conditioned
  HDR image/video workflows, joint audio, LoRA composition, and staged restart;
- GMNet SDR-to-HDR still expansion with transactional EXR, PQ HEIC, and optional
  gain-map output.

KinoMLX is an independent implementation built from published architecture
descriptions and implementations, checkpoint metadata, and independently
constructed behavioral fixtures. Reference repositories are specification
sources, not runtime dependencies.

## Status

Pre-alpha. The end-to-end LTX distilled pipeline is implemented and tested for
both LTX-2.3 and LTX-2.5, including native video and optional stereo-audio
delivery. GMNet provides a separate still-image path using native ImageIO with
half-float EXR and 10-bit PQ HEIC output. Interfaces and defaults can still
change between commits.

| Model | Input and output | Public recipe | Runner |
| --- | --- | --- | --- |
| `ltx2` | Text/optional image -> video and optional audio | `generate_distilled()` | `LTX2Runner` |
| `gmnet` | Display-referred SDR still -> HDR still artifacts | `expand_gmnet()` | `GMNetRunner` |

The [LTX2 model guide](docs/LTX2.md) contains the complete capability matrix and
version-specific behavior. In particular, automatic duration and generated
keyframe slots are LTX-2.5 features, LTX-2.3 owns the validated SDR-video-to-HDR
profile, and LTX-2.5 owns native HDR-still-to-video generation.

## Requirements

- Apple Silicon (M1 family or later)
- macOS
- Python 3.14
- 32 GB unified memory minimum for streamed LTX generation; 64 GB recommended
  for the default resident-transformer path and larger geometries

GMNet has a much smaller working set that scales primarily with source-image
dimensions.

## Install

```bash
git clone https://github.com/dmlsrc/KinoMLX.git
cd KinoMLX
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
kinomlx --help
```

KinoMLX does not download model weights at runtime. The LTX2 model class uses
Hugging Face model repositories. For cache-managed downloads, optionally install
the `hf` CLI as an isolated uv tool and verify it:

```bash
uv tool install hf
hf --help
```

The official [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
also documents the no-install `uvx hf` form. If `hf` is not on `PATH` after a
tool install, run `uv tool update-shell` and open a new shell. You can skip the
CLI entirely and download the approved files through the model pages instead.

## LTX2 model access

Open the applicable model pages while signed in and inspect their model cards,
files, licenses, and access terms:

- [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
- [LTX-2.3 FP8](https://huggingface.co/Lightricks/LTX-2.3-fp8)
- [LTX-2.3 HDR IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR)
- [LTX-2.3 paired QAT-unquantized Gemma 3](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
- [Optional vanilla Gemma 3](https://huggingface.co/google/gemma-3-12b-it)
- [LTX-2.5 split component pack](https://huggingface.co/Lightricks/LTX-2.5)
- [Gemma 4 12B model page](https://huggingface.co/google/gemma-4-12B-it); LTX-2.5
  supplies its prepared text artifact inside the component pack

Accept or request access for every repository shown as gated before downloading.
CLI authentication alone does not grant access to a gated repository. After
access is approved, either use `hf auth login` for cache-managed downloads or
download the required files through each page's Files and versions tab and place
them in a local folder. The [LTX2 model guide](docs/LTX2.md#ltx-model-downloads)
lists the exact files, the one-directory LTX-2.5 form, and the complete
[HDR IC-LoRA workflow](docs/LTX2.md#ltx-23-hdr-ic-lora). It also shows how to
override individual LTX-2.3 or LTX-2.5 components, convert an FP8 source into a
BF16 or FP16 cache, compose stage-specific LoRAs with target knockouts, and run
the 32 GB block-streaming profile.

GMNet checkpoints are published by the
[GMNet project](https://github.com/qtlark/GMNet) rather than Hugging Face. The
[GMNet model guide](docs/GMNET.md) explains the restricted, torch-free conversion
workflow.

Model weights are not bundled in the repository, source distribution, or wheel.

## Quick start

Once the corresponding local model files are available:

```bash
# LTX-2.3 using the monolithic distilled checkpoint family
kinomlx --model ltx2 \
  --weights-path /path/to/ltx-2.3-22b-distilled-1.1.safetensors \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --frames 121 --generate-audio

# LTX-2.5 using one local component-pack directory
kinomlx --model ltx2 --model-generation 2.5 \
  --weights-path /models/ltx25 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 576 --auto-duration --generated-keyframes 1
```

For LTX-2.5, the directory may retain the official repository subdirectories
or contain the canonical component filenames directly. Cache discovery remains
available when `--weights-path` is omitted.

Use `kinomlx config init` to write an annotated LTX starter configuration or
`kinomlx config init --model gmnet --output gmnet.toml` for GMNet. Run
`kinomlx --model ltx2 --help`, `kinomlx --model gmnet --help`, and
`kinomlx weights --help` for the executable command surfaces.

## Documentation

- [LTX2 model guide](docs/LTX2.md) - gated downloads, LTX-2.3/LTX-2.5 behavior,
  configuration, restart, outputs, HDR, performance, and the LTX2 library API
- [GMNet model guide](docs/GMNET.md) - checkpoint conversion, variants, still
  expansion, transactional output, sidecars, and the GMNet library API
- [Project structure](docs/STRUCTURE.md) - package map, layer responsibilities,
  public API surface, and implementation conventions
- [LTX composition example](examples/ltx2_distilled.py) - explicit component
  schedule and VideoToolbox output ownership
- [GMNet composition example](examples/gmnet_expand.py) - compact still-expansion
  and transactional-output composition

## Performance

On an M1 Max with 64 GB, matched LTX-2.3 and LTX-2.5 768x448, 121-frame joint
audio/video runs each completed in about 2 minutes 30 seconds end to end. A
one-off LTX-2.3 1024x576, 481-frame production run measured about 23 minutes.
Detailed decoder timing and memory context are in the
[LTX2 model guide](docs/LTX2.md#performance).

## Scope and limitations

- The public LTX recipe is the distilled two-stage 8 + 3 pipeline. Dev CFG/STG,
  DFR, general retake/video-to-video, and alternative public recipes remain out
  of scope.
- The validated video-to-video profile is LTX-2.3 SDR-to-HDR conversion with a
  full-length aligned SDR reference and declared LogC3 HDR IC-LoRA.
- GMNet expands SDR stills only; it is not temporal video expansion and does not
  accept scene-linear EXR input.
- Frame counts are `8*k+1`; `--duration` rounds up to the next valid count.
- The generic converter accepts zip-format tensor-only PyTorch checkpoints;
  legacy stream checkpoints and float64 storage are refused.
- Apple Silicon Macs only.

## Attribution and license

KinoMLX builds on Apple's [MLX](https://github.com/ml-explore/mlx) framework and
the published architecture implementations and checkpoints from Lightricks,
Hugging Face, Google, and the GMNet authors. KinoMLX does not redistribute model
weights. Each model remains governed by its upstream terms; the detailed model
links and license notes are in the [LTX2](docs/LTX2.md#model-weights) and
[GMNet](docs/GMNET.md#model-weights) model guides.

KinoMLX's own code is [MIT-licensed](LICENSE). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for included third-party
license texts and notices.
