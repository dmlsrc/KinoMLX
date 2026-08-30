# GMNet Model Guide

This document owns the setup, checkpoint conversion, command-line, variant,
output, sidecar, and library API reference for KinoMLX's `gmnet` model class.
For the project overview and shared installation, see the
[root README](../README.md). LTX2 has a separate [model guide](LTX2.md).

GMNet ("Learning Gain Map for Inverse Tone Mapping", ICLR 2025) predicts a gain
map plus a Qmax scalar from one display-referred SDR still. KinoMLX reconstructs
a normalized scene-linear HDR image and can publish a half-float EXR master,
10-bit BT.2100 PQ HEIC, and self-describing gain-map sidecar as one transaction.

## Status

The published `realworld` and `synthetic` checkpoints are both implemented and
tested. Native ImageIO owns input and output; the model path does not depend on
PyTorch, NumPy, Pillow, or the Hugging Face runtime.

| Model | Input and output | Public recipe | Runner |
| --- | --- | --- | --- |
| `gmnet` | Display-referred SDR still -> HDR still artifacts | `expand_gmnet()` | `GMNetRunner` |

GMNet is a much smaller model than LTX2. Its working set scales primarily with
the source-image dimensions. KinoMLX still requires Apple Silicon, macOS, and
Python 3.14.

## Install and configure GMNet

Complete the [shared installation](../README.md#install), then inspect the GMNet
command and generate an annotated starter configuration:

```bash
kinomlx --model gmnet --help
kinomlx weights --help
kinomlx config init --model gmnet --output gmnet.toml
```

The generated TOML documents every GMNet field, accepted value, built-in
default, and environment variable. It refuses to replace an existing file.

## Model weights

KinoMLX does not bundle GMNet weights. Both published checkpoints are
MIT-licensed files in the upstream
[GMNet repository](https://github.com/qtlark/GMNet):

| Variant | Published checkpoint | Source SHA-256 |
| --- | --- | --- |
| `realworld` | [G_realworld.pth](https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth) | `83bf27bcdbf6eacfdef37f0e24ed6d79152b7386620c012ae509a59a895c875f` |
| `synthetic` | [G_synthetic.pth](https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth) | `887c940d492424cd44f029c6b09dd3bbe1bbec07126f15d41192828ff95e6880` |

Download either file directly from the table, or use `curl`, and place it under
`weights-src/gmnet/` with its published filename:

```bash
mkdir -p weights-src/gmnet

curl -L -o weights-src/gmnet/G_realworld.pth \
  https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth

curl -L -o weights-src/gmnet/G_synthetic.pth \
  https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth
```

Convert the selected source from the repository root. Bare filenames resolve
under `weights-src/`, and published checkpoints are recognized by SHA-256:

```bash
kinomlx weights convert gmnet G_realworld.pth
kinomlx weights convert gmnet G_synthetic.pth
```

The first conversion creates the canonical file; use `--force` only when
deliberately reconverting that same destination. In an editable Git checkout,
canonical converted weights stay beside the model-owned conversion notes at
`kinomlx/models/gmnet/weights/`. In a non-checkout install, the default is
`$KINO_CACHE_DIR/weights/gmnet/`, so conversion never assumes the installed
package is writable. An explicit converter `--output` or runtime
`KINO_GMNET_WEIGHTS`/`--weights-path` override wins in either layout.

The model-specific converter statically scans the embedded pickle, rebuilds
tensors through a restricted torch-free reader, requires the complete generator
state dict, stamps variant and source metadata, and verifies the result through
the GMNet loader. Outputs refuse to clobber existing files unless `--force` is
explicit, and a candidate is verified before a forced replacement is published.

KinoMLX also has a value- and layout-preserving converter for plain tensor state
dicts:

```bash
kinomlx weights convert checkpoint.pth -o checkpoint.safetensors
```

Use the generic path only when no model-owned converter is available. It can
select a nested mapping with `--param-key`, filter and strip key prefixes, and
refuses to guess between `params` and `params_ema`; it does not transpose tensors
or claim a model-specific contract. Legacy stream checkpoints and float64
storage are refused.

## Usage

The default `realworld` variant targets photographed pairs with 203-nit SDR
white, up to 5x peak, and a half-resolution local branch. The `synthetic`
variant targets HDR-video-derived content with 100-nit SDR white, up to 8x peak,
and a full-resolution local branch:

```bash
# Default realworld variant
kinomlx --model gmnet \
  --image photo.jpg --output-dir hdr/ --save-gain-map

# Synthetic variant with its corresponding converted weights
kinomlx --model gmnet --variant synthetic \
  --image video-frame.png --output-dir hdr/ --save-gain-map
```

The variant follows the usual precedence:
`KINO_GMNET_VARIANT`, TOML `[model_settings].variant`, `--variant`, then
`--set model_settings.variant=...`. `KINO_GMNET_WEIGHTS` and `--weights-path`
override the converted-weights location.

Those 203-nit and 100-nit figures describe each checkpoint's training and
reconstruction contract. Both variants return a normalized scene-linear plate
where 1.0 is SDR diffuse white. The HEIC terminal maps that 1.0 level to its
fixed 203-nit PQ reference white. Selecting `synthetic` therefore changes the
model prior and expansion range, not the HEIC terminal's reference white.

The EXR output can be used as the HDR conditioning plate for LTX-2.5 native HDR
image-to-video; see the [LTX2 HDR contract](LTX2.md#hdr-generation).

### Output selection and publication

Without an exact `--output`, GMNet derives the input stem and writes both EXR
and HEIC under `outputs/`. An exact `.exr` or `.heic` path selects exactly that
primary artifact; add `--heic` or `--exr` to request its sibling explicitly.
`--save-gain-map` adds a safetensors sidecar carrying the normalized gain map
and its reconstruction law.

All selected targets are reserved before inference with hidden
`.<name>.kinomlx-reservation` peers rather than zero-byte final artifacts. Every
artifact is completed in a private peer directory, and the bundle is published
together. Existing artifacts remain untouched on an encoder or publication
failure, including forced replacements. If a process is terminated without
cleanup, the next refusal names the stale marker that the user can inspect and
remove after confirming no run is active.

GMNet exposes `--save-effective-config`, `--save-console-log`, and
`--save-run-log` individually. `--save-all-sidecars` enables all three plus the
normalized gain map. They share the resolved output stem and join the
pre-inference target reservation, so an existing selected sidecar is refused
alongside an existing EXR or HEIC. `--force` explicitly authorizes replacement
of the complete selected bundle.

The model scalar can select GMNet from TOML without a CLI `--model` flag:

```toml
model = "gmnet"

[expand]
image = "photo.jpg"

[model_settings]
variant = "realworld"

[output]
directory = "hdr"
save_gain_map = true
```

## Library API

GMNet exposes a typed request, immutable prepared resources, injectable
component leases, a stateless recipe, a runner, and a transactional output
plan:

```python
from pathlib import Path

from kinomlx import (
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    GMNetSettings,
    Settings,
    expand_gmnet,
    plan_gmnet_output,
    prepare_gmnet_resources,
)

resources = prepare_gmnet_resources(
    GMNetSettings.from_env(),
    infrastructure=Settings.from_env(),
)
request = GMNetRequest(Path("photo.jpg"))
plan = plan_gmnet_output(
    request,
    GMNetOutputConfig(directory=Path("hdr"), save_gain_map=True),
)
runner = GMNetRunner(resources=resources)

with plan.reserve() as reservation:
    result = runner.run(expand_gmnet, request)
    artifacts = GMNetOutputSink(plan).write(result, reservation=reservation)
```

`GMNetRunner.expand(request)` is the convenience form of
`runner.run(expand_gmnet, request)`. Hosts with their own terminal policy can
consume the returned scene-linear `ExpansionResult` directly. See
[`examples/gmnet_expand.py`](../examples/gmnet_expand.py) for the complete small
composition.

## Limitations

- GMNet is still-image SDR-to-HDR expansion only. It is not temporal video
  expansion and does not accept scene-linear EXR input.
- Input is one display-referred SDR still; variant selection must match the
  converted checkpoint.
- Pre-alpha interfaces and defaults can change between commits.
- Apple Silicon Macs only.

## Attribution and license

The architecture, paper, and published checkpoints are from
[GMNet](https://github.com/qtlark/GMNet), "Learning Gain Map for Inverse Tone
Mapping" (Liao et al., ICLR 2025), under the upstream MIT license. KinoMLX does
not redistribute the checkpoints.

KinoMLX's own code is [MIT-licensed](../LICENSE). See
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for included license texts
and notices.
