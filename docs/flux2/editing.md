# Image editing

Klein edits from one or more reference images, in the order you give them. Any variant can edit;
`kv` is faster with references, at a memory cost described in
[performance](performance.md#reference-kv).

```bash
kinomlx --model flux2 --variant kv --source-image reference.jpg \
  --prompt "Turn the ceramic vase into blue glass, preserving the window and light." \
  --match-source-aspect --height 1024
```

- [References](#references)
- [Output size](#output-size)
- [Fit the references](#fit-the-references)
- [What the log reports](#what-the-log-reports)
- [Saving edits](#saving-edits)

## References

Repeat `--source-image` for several references; their order matters to the model. Only the output
is denoised: references condition every step and are never changed.

EXIF orientation is applied when a reference is read, and references are resized with bicubic
resampling. Another implementation that uses Lanczos gives different conditioning pixels from the
same image, so feed both the same preprocessed tensors when isolating model differences.

## Output size

With one reference and no size, Klein matches the reference's shape at 1024 pixels on its longer
side. `--match-source-aspect` sets which side you pin, with the same values as Krea 2:

| Value | Pinned side | Its size comes from |
| --- | --- | --- |
| `long` (the bare flag) | The longer side | `--width` or `--height`, else 1024 |
| `short` | The shorter side | `--width` or `--height`, else 1024 |
| `width` | The width | `--width`, else 1024 |
| `height` | The height | `--height`, else 1024 |

Give at most one size when matching the aspect, on the side you pin; both sides then round to
multiples of 16, the pinned one up. With both `--width` and `--height` given, Klein keeps exactly
that grid, which must already be a multiple of 16. With several references,
`--match-source-aspect` follows the first one.

## Fit the references

`--reference-fit` decides how a reference sits in the output grid. It does not crop the output.

| `--reference-fit` | Behavior | Choose it when |
| --- | --- | --- |
| `fit` (default with one reference) | The whole reference, centered | Keeping all of the source matters |
| `crop` | A centered crop that fills the grid | Filling the canvas matters more |

A reference of a very different shape from the output can leave much empty space under `fit`.

With several references and no explicit fit, each keeps its own native grid, capped at about
1024x1024 in area. Native grids need sources at least 64 pixels on each side and no more than 8
times as long as they are wide.

## What the log reports

Before generating, the log shows the canvas, each reference's grid, crop box and position offset,
and the bicubic resizing used. The KV variant also reports the reference token count and the
estimated cache size; see [reference memory](performance.md#reference-kv).

In the run record, `resized_size` is a reference's size before its `crop_box`, and
`reference_size` is the final image given to the VAE.

## Saving edits

`--save-text-conditioning` saves the encoded prompt together with the encoded references, their
positions, and their geometry. A [denoise restart](restart.md) therefore replays an edit without
opening the source images again. Reference KV is never saved: a replay extracts it afresh on its
first step.
