# GMNet: HDR from an SDR photograph

GMNet ("Learning Gain Map for Inverse Tone Mapping", ICLR 2025) expands one ordinary SDR still into
an HDR image. It predicts a gain map and a peak scale, Qmax, from the photograph; KinoMLX rebuilds
a scene-linear HDR image from them and writes it as a half-float EXR master and a 10-bit PQ HEIC
you can view in Photos.

```bash
kinomlx --model gmnet --image photo.jpg --output-dir hdr/ --save-gain-map
```

GMNet is much smaller than the generative models, and its memory grows with the image size.

Image input and output use native macOS frameworks, so PyTorch, NumPy, Pillow, and the Hugging
Face packages are not needed.

## Choose a variant

| `--variant` | Trained on | SDR white | Peak | Local branch |
| --- | --- | --- | --- | --- |
| `realworld` (default) | Photographed SDR and HDR pairs | 203 nits | Up to 5x | Half resolution |
| `synthetic` | Content derived from HDR video | 100 nits | Up to 8x | Full resolution |

The run log shows the variant's values as "Model SDR reference white", "Model peak over SDR
white", and "Network scale", and after expanding it reports the predicted Qmax and peak.

Both variants write the same kind of plate: scene-linear, with 1.0 at SDR diffuse white. The HEIC
maps 1.0 to the same 203-nit reference white for both, so the variant changes what the model
expects and how far it expands highlights, not the HEIC's white level.

Each variant needs its own converted weights, and the variant you run must match them: KinoMLX
does not check which variant a weights file was converted from.

## Quick start

1. Download and convert a checkpoint once; see [Weights](weights.md).
2. Check that KinoMLX finds it:

   ```bash
   kinomlx --model gmnet --dry-run
   kinomlx --model gmnet --dry-run --variant synthetic
   ```

3. Expand a photograph:

   ```bash
   kinomlx --model gmnet --image photo.jpg --output-dir hdr/
   kinomlx --model gmnet --variant synthetic --image video-frame.png --output-dir hdr/
   ```

The EXR works as the HDR still for [LTX-2.5 HDR video](../ltx2/hdr.md#ltx-25-animate-an-hdr-still).

## Config files

A config file can select GMNet with `model = "gmnet"`, so `--model` is not needed:

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

`kinomlx config init --model gmnet` writes a template with every field and its default; see
[Configuration](../CONFIG.md).

## Guides

| Page | Read it to |
| --- | --- |
| [Weights](weights.md) | Download and convert the published checkpoints. |
| [Output files](output.md) | Choose EXR, HEIC, and the gain map, and name the files. |
| [Python API](api.md) | Expand images from Python. |
| [Arguments](arguments.md) | Look up any argument. |

## Limits

- Still images only: GMNet does not expand video over time.
- Input is one display-referred SDR still; scene-linear EXR input is not accepted.
- Apple Silicon Macs only.
- Interfaces and defaults can change between commits.

## Licenses and credits

The architecture, paper, and published checkpoints are from
[GMNet](https://github.com/qtlark/GMNet) (Liao et al., ICLR 2025), under the upstream MIT license.
KinoMLX does not redistribute the checkpoints. KinoMLX's own code is
[MIT-licensed](../../LICENSE); see [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) for
included license texts.
