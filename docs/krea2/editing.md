# Image editing

Krea 2 edits one or two reference images from an instruction, with a compatible edit adapter.
The default adapter is the community identity-edit adapter, found in the Hugging Face cache once
downloaded; see [optional files](downloads.md#optional-files).

```bash
kinomlx --model krea2 \
  --source-image source.png \
  --prompt "Change the jacket to blue, preserving the person and background." \
  --output edited.png
```

- [The edit adapter](#the-edit-adapter)
- [Several results](#several-results)
- [Raw editing](#raw-editing)
- [Two references](#two-references)
- [Output size](#output-size)
- [Fit the reference](#fit-the-reference)
- [Focus the reference on regions](#focus-the-reference-on-regions)
- [Describe the source differently](#describe-the-source-differently)
- [Saving and restarting edits](#saving-and-restarting-edits)
- [Config files](#config-files)

## The edit adapter

Without `--edit-adapter`, KinoMLX uses `krea2_identity_edit_v1_2_r128.safetensors` from the cached
`main` snapshot of its repository. It never downloads the file or picks an older snapshot.
`--edit-adapter PATH` uses another one.

`--edit-adapter-strength` (default 1) and `--edit-adapter-exclude` control the edit adapter on its
own, apart from any style adapters. Add those with `--adapter` as usual: they do not replace the
edit adapter, and listing the edit adapter again under `--adapter` is refused, so it is never
applied twice.

```bash
kinomlx --model krea2 --source-image source.png \
  --prompt "Restyle the scene while preserving the subject." \
  --adapter /path/to/style.safetensors --adapter-strength 0.35 \
  --output edited.png
```

This applies the edit adapter at 1 and the style adapter at 0.35. Their updates add up; they are
not averaged.

## Several results

An edit takes one prompt and at most one negative prompt. A seed list or `--images-per-prompt`
makes several edited images of it, encoding the references only once; see
[seeds](../USAGE.md#seeds).

## Raw editing

Turbo edits in eight steps without guidance. For the community Raw editing recipe, add
`--variant raw --steps 20 --guidance 2`. Both guidance branches see the reference images.

## Two references

Repeat `--source-image` for two references: the scene first, then the subject.

| Option | Controls | Default |
| --- | --- | --- |
| `--reference-strength` | How strongly the output attends to the sole or second reference | 1 |
| `--scene-reference-strength` | How strongly the output attends to the first of two | 1 |
| `--edit-adapter-strength` | How strongly the edit adapter changes the weights | 1 |

The reference strengths act on attention, the adapter strength on the weights, so they combine.

## Output size

Without other options, the output is `--width` by `--height`, 1024x1024 by default. A source of
another shape is then centered in that canvas, and the sides are filled in.

`--match-source-aspect` keeps the source's shape instead. Its value says which side you set:

| Value | Pinned side | Its size comes from |
| --- | --- | --- |
| `long` (the bare flag) | The longer side | `--width` or `--height`, else 1024 |
| `short` | The shorter side | `--width` or `--height`, else 1024 |
| `width` | The width | `--width` |
| `height` | The height | `--height` |

Give one size at most. The pinned side rounds up and the other to the nearest multiple of 16. A
720x1280 phone photo edits at 576x1024 with the bare flag, and each step costs about a third less
than at 1024x1024, because both the reference and the output shrink.

## Fit the reference

`--reference-fit fit`, the default, keeps the reference's aspect ratio and centers it in the
output grid; a nearly matching shape is filled after a small center crop. `--reference-fit crop`
fills the output with a center crop, for older adapters trained that way.

The vision model sees the source at most 768 pixels on its longest side. `--grounding-px 0` lets it
see the full size.

## Focus the reference on regions

`--reference-strength` normally applies to the whole last reference. A mask or shapes can limit it
to parts of the image, for example to hold a face or a garment closer to the source:

```bash
kinomlx --model krea2 --source-image source.png \
  --prompt "Change the jacket to blue." \
  --reference-strength 1.5 --reference-boost-mask subject-mask.png \
  --output edited.png
```

`--reference-boost-mask` takes an image the size of the last source: white selects, black leaves
attention unchanged. Color masks use the channel mean, and alpha is ignored. It is cropped like
the source, resized to the reference's token grid, and selects where its value is above 0.5, so
mid-gray and antialiased edges count only past half. A missing mask is refused before anything
loads.

`--reference-boost-region` selects shapes instead, with numbers from 0 to 100 percent of the
reference as the model sees it, measured from its top-left corner; a `%` sign is optional. Repeat
it to combine shapes, with each other and with a mask:

| Shape | Example | Numbers |
| --- | --- | --- |
| Rectangle | `--reference-boost-region "rect:35,15,65,55"` | left, top, right, bottom |
| Circle | `--reference-boost-region "circle:50,30,15"` | center x, center y, radius as a share of the shorter side |
| Ellipse | `--reference-boost-region "ellipse:50,30,15,20"` | center x, center y, then radii as shares of the width and the height |
| Polygon | `--reference-boost-region "polygon:30,15;70,15;80,60;20,60"` | x,y corners separated by semicolons |

`rectangle` and `poly` also work. Quote the value so the shell leaves semicolons alone. Polygons
may be concave and wind either way; a self-crossing outline fills by the even-odd rule. Circles and
ellipses are clipped at the edge of the view, and tiny shapes can miss every token and select
nothing.

A selection needs a reference strength other than 1; at strength 1 it is refused. It focuses
attention on the reference and is not an inpainting mask, so other parts of the output can still
change. The first of two references always uses its whole image.

## Describe the source differently

Editing describes the source images to the text encoder with its own system instruction.
`--grounding-system-prompt` replaces it, for example to stress identity or clothing:

```bash
--grounding-system-prompt "Describe facial identity and clothing details."
```

An empty or whitespace-only value keeps the training default. The instruction applies to both the
positive and the negative branch, and it steers the language conditioning only; the vision model
is unchanged.

## Saving and restarting edits

`--save-all-sidecars` saves the starting noise, the final latents, and the run records for an
edit, so a [decode restart](restart.md#decode-again) works as usual. The run record lists the
source images and reference settings.

An edit cannot restart from text conditioning, because that lacks the encoded references. Edit
runs therefore do not save it, and an explicit `--save-text-conditioning` is refused, even beside
`--save-all-sidecars`.

The references are always encoded with the stock Krea VAE. `--vae-decoder` and `--vae-path`
change only how the output is decoded.

Python requests edit through `Krea2Request(..., source_images=(Path("source.png"),))`; see
[the Python API](api.md).

## Config files

The edit options keep their own names in TOML; this restyles a source at its own aspect ratio:

```toml
model = "krea2"

[generate]
prompt = "Restyle the scene while preserving the subject."
seed = 42
source_images = ["source.png"]
match_source_aspect = "long"

[[generate.adapters]]
path = "style.safetensors"
strength = 0.35

[output]
path = "edited.png"
save_all_sidecars = true
```

`kinomlx config init --model krea2` lists the other editing fields, such as `edit_adapter_path`
and `reference_boost_regions`.
