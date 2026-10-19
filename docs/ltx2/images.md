# Image conditioning

`--image` makes a frame of the video follow an image: the first frame, the last, or any frame in
between. Repeat it to condition several frames, for example to animate between a start and an
end picture:

```bash
kinomlx --model ltx2 \
  --image start.png@first --image end.png@last \
  --prompt "A slow cinematic push through the illuminated scene" \
  --width 768 --height 448 --frames 121
```

- [Write a condition](#write-a-condition)
- [Set members with separate flags](#set-members-with-separate-flags)
- [Config files and Python](#config-files-and-python)
- [How frame 0 is applied](#how-frame-0-is-applied)
- [What a keyframe costs](#what-a-keyframe-costs)
- [Restarting](#restarting)

## Write a condition

Each `--image` is one condition:

```text
--image PATH[@POSITION][:STRENGTH]
--image PATH[@POSITION]:STAGE1:STAGE2
```

| Part | Meaning | Default |
| --- | --- | --- |
| `@POSITION` | `first`, `last`, or a frame number counted from 0 | `first` |
| `:STRENGTH` | How closely the frame follows the image, from 0 to 1, in both stages | `1.0` |
| `:STAGE1:STAGE2` | Separate strengths for stage 1 and stage 2 | Both follow the common strength |

```bash
--image first-frame.png              # frame 0 at strength 1
--image last-frame.png@120           # frame 120
--image last-frame.png@last:0.8      # the last frame at 0.8
--image last-frame.png@last:0.9:0.5  # 0.9 in stage 1, 0.5 in stage 2
```

`first` and `0` are the same frame. `last` is resolved after the length is known, so it works
with `--duration` and `--auto-duration` too. Two conditions on one frame are refused, even when
spelled differently: on a 17-frame run, `last` and `16` are the same frame.

A suffix counts only when it is `first`, `last`, digits, or a number, so a path such as
`photo@2x.png` or `shots/take:2.png` is read as written. A third numeric suffix is refused, and
files are not opened while the arguments are read.

## Set members with separate flags

These flags change the condition started by the latest `--image`, the same way the adapter
flags work:

```bash
--image first-frame.png --image last-frame.png --image-frame-index last --image-strength 0.8
--image last-frame.png@120 --image-stage1-strength 0.9 --image-stage2-strength 0.5
```

| Flag | Sets |
| --- | --- |
| `--image-frame-index` | The frame |
| `--image-strength` | The strength in both stages |
| `--image-stage1-strength` | The stage-1 strength |
| `--image-stage2-strength` | The stage-2 strength |

A flag affects only its own condition. Using one before any `--image`, or setting a member the
shorthand already set, is an error.

## Config files and Python

In TOML, `images` is a list of records under `[generate]`. Each record has `path`,
`frame_index`, `strength`, `stage1_strength`, and `stage2_strength`:

```toml
[generate]
images = [
  {path = "first-frame.png"},
  {path = "last-frame.png", frame_index = "last", strength = 0.8, stage2_strength = 0.5},
]
```

A bare number for `frame_index` is the same as its quoted form: `frame_index = 120` reads back as
`"120"`. `KINO_LTX2_IMAGES` takes the same list. Change one item with
`--set 'generate.images[0].strength=0.9'`, or clear the list with `--set 'generate.images=[]'`.

Python requests take `LTX2Request(images=(LTX2ImageConditioningConfig(...), ...))`; both types
come from `kinomlx.models.ltx2`.

## How frame 0 is applied

A condition at frame 0 can be applied two ways, chosen with `--first-frame-mode`:

| Mode | Frame 0 | Use it when |
| --- | --- | --- |
| `anchor` (default) | Held at the encoded image throughout | The video must start exactly on your picture. |
| `guide` | Denoised like the rest, steered toward the image | An adapter should change the first frame too. |

With `anchor` and strength 1, the first latent frame is exactly the encoded image. The decoded
frame is the image after one trip through the VAE, so it is very close to, but not identical
with, your file.

`anchor` matters with adapters that change the image, such as colorize, upscale, or style
adapters. The anchored frame 0 stays untouched while frames 1 and later change, so the video
visibly jumps at frame 1.

`guide` removes that jump; frame 0 then comes out close to the image but not equal to it:

```bash
--image first-frame.png --first-frame-mode guide --adapter style.safetensors:0.8
```

The mode affects only a frame-0 condition: conditions on other frames always guide.

## What a keyframe costs

A condition at any frame other than 0 adds extra tokens the model attends to: one latent frame
in each stage. That is `(height / 32) * (width / 32)` tokens in stage 2, and a quarter as many in
half-size stage 1:

| Video | Stage-2 video tokens | One keyframe adds |
| --- | ---: | ---: |
| 768x448, 121 frames | 5,376 | 336, about 6% |
| 1920x1088, 121 frames | 32,640 | 2,040, about 6% |
| 768x448, 721 frames | 30,576 | 336, about 1% |

Attention time grows faster than the token count.

One latent frame spans eight video frames. Keyframes closer together than about eight frames
compete for the same stretch of video.

## Restarting

A [stage-2 restart](restart.md) keeps each condition's path, frame, and stage-1 strength, so it
may change only stage-2 strengths. A decode restart allows no image changes. Respelling `last` as
`120` on a 121-frame run is the same frame and counts as no change.
