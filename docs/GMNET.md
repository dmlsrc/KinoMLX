# GMNet Model Guide

Use GMNet to expand an SDR photograph into an HDR still. This guide covers the
one-time weight conversion, realworld/synthetic model choice, output formats,
and Python API. Start with the [shared installation](../README.md#install).

GMNet ("Learning Gain Map for Inverse Tone Mapping", ICLR 2025) predicts a gain
map plus a Qmax scalar from one display-referred SDR still. KinoMLX reconstructs
a normalized scene-linear HDR image and can publish a half-float EXR master,
10-bit BT.2100 PQ HEIC, and self-describing gain-map sidecar as one transaction.

In this guide: [model weights](#model-weights), [generation](#usage),
[output files](#output-selection-and-publication), [Python API](#library-api),
and [limitations](#limitations).

## Status

Both `realworld` and `synthetic` checkpoints are supported. Input and output use
native macOS image handling; you do not need PyTorch, NumPy, Pillow, or Hugging
Face packages to run the model.

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

Conversion creates a reusable safetensors file; use `--force` only when
deliberately replacing that destination. In an editable Git checkout,
converted weights are saved under
`kinomlx/models/gmnet/weights/`. In a non-checkout install, the default is
`~/Library/Caches/KinoMLX/weights/gmnet/`, so conversion never assumes the installed
package is writable. An explicit converter `--output` or runtime
`KINO_GMNET_WEIGHTS_PATH`/`--weights-path` override wins in either layout.

Use the GMNet converter for the published GMNet files: it checks the variant,
requires all generator weights, and verifies that the result can load before
publishing it. It reads tensor data through a restricted reader without executing
the checkpoint's Python code or requiring PyTorch. With `--force`, verification
still happens before the existing destination is replaced.

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
`--set model_settings.variant=...`. `KINO_GMNET_WEIGHTS_PATH` and `--weights-path`
override the converted-weights location.

Those 203-nit and 100-nit figures describe each checkpoint's training and
reconstruction contract. Both variants return a normalized scene-linear plate
where 1.0 is SDR diffuse white. The HEIC terminal maps that 1.0 level to its
fixed 203-nit PQ reference white. Selecting `synthetic` therefore changes the
model prior and expansion range, not the HEIC terminal's reference white.

The EXR output can be used as the HDR conditioning plate for LTX-2.5 native HDR
image-to-video; see the [LTX2 HDR contract](LTX2.md#hdr-generation).

### Output selection and publication

Without an exact `--output`, GMNet uses `<input-stem>_YYYYMMDD_HHMMSS` and writes
both EXR and HEIC, following the
[shared app output naming policy](../README.md#output-names-and-sidecars).
`--output-prefix` replaces the input stem but keeps the timestamp. The directory
defaults to `KINO_GMNET_OUTPUT_DIR`, then `KINO_OUTPUT_DIR`, then `outputs/`.
`--output-dir` overrides it.
Same-second collisions add `_2`, `_3`, and so on to the whole bundle's stem.
An exact `.exr` or `.heic` path selects exactly that primary artifact without
a timestamp; add `--heic` or `--exr` to request its sibling explicitly.
`--save-gain-map` adds a safetensors sidecar carrying the normalized gain map
and its reconstruction law.

KinoMLX protects existing outputs and prepares all requested files before publishing
them together. An ordinary encoding or publication failure leaves previous files
intact, including when using `--force`.

If a run is terminated without cleanup, its reserved output name may remain occupied.
Generated filenames skip that name automatically. For an exact output path, the error
identifies a hidden `.<name>.kinomlx-reservation` marker. Confirm that no run is still
using the output before removing the named stale marker and retrying.

GMNet exposes `--save-effective-config`, `--save-console-log`, and
`--save-run-log` individually. `--save-all-sidecars` enables all three plus the
normalized gain map. They share the resolved output stem and join the
pre-inference target reservation. A collision with any selected artifact or
sidecar advances the entire generated stem. With an exact `--output`, existing
targets are refused unless `--force` explicitly authorizes replacement of the
complete selected bundle.

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

from kinomlx import Settings
from kinomlx.models.gmnet import (
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    GMNetSettings,
    expand_gmnet,
    plan_gmnet_output,
    prepare_resources,
)

resources = prepare_resources(
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

Installed-package conversion storage and native signpost builds use
`--app-cache-dir` / `settings.app_cache_dir`. The environment lookup order is
`KINO_APP_CACHE_DIR`, then `XDG_CACHE_HOME/KinoMLX`, then
`~/Library/Caches/KinoMLX`.
Conversion remains an explicit manual command; normal generation never opens
a pickle checkpoint. Editable checkout weights remain beside the model code.

### Memory policy for API calls

The runner and public recipe functions apply the resource inventory's shared
memory policy before model work. Set `mlx_cache_limit_gb` and
`mlx_wired_limit_gb` through infrastructure `Settings`. Reusing resources
preserves those choices; each invocation reapplies them. For standalone
component loading, call `kinomlx.apply_memory_policy(resources.memory_policy)`
before allocating inputs or loading weights. See
[public API memory policy](PERFORMANCE.md#public-api-memory-policy).
