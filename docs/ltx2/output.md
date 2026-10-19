# Upscaling, frame rate, and encoding

After [decoding](decode.md), frames pass through optional VideoToolbox upscaling and frame-rate
conversion, then into a 10-bit HEVC MP4 with the audio. This page covers those steps and the
files a run writes.

- [Upscale](#upscale)
- [Change the frame rate](#change-the-frame-rate)
- [Scene cuts](#scene-cuts)
- [Encode](#encode)
- [Output files](#output-files)
- [Sidecars](#sidecars)

## Upscale

`--vsr-spatial-mode` enlarges the decoded video with macOS's VideoToolbox super resolution. It is
far cheaper than generating at the larger size:

| Mode | Scale | Input size limit | How it works |
| --- | --- | --- | --- |
| `off` (default) | 1x | | No upscaling |
| `fast` | 2x | 960x960 | Low-latency, one frame at a time |
| `balanced` | 4x | 1920x1080 | Uses the previous frame: slightly crisper moving edges, slightly more frame-to-frame variation |
| `image` | 4x | 1920x1080 | One frame at a time: the smoothest from frame to frame |

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise, cinematic light" \
  --width 768 --height 448 --vsr-spatial-mode balanced
```

The first `balanced` or `image` run may wait while macOS downloads its super-resolution model.

`balanced` falls back to `image` when a side is under 128 pixels, where its frame history is
unreliable. HDR runs cannot be upscaled yet.

## Change the frame rate

`--target-fps` interpolates new frames after upscaling, for example to deliver 60 fps from a
24 fps generation:

```bash
kinomlx --model ltx2 --prompt "A quiet coastal sunrise, cinematic light" --target-fps 60
```

On macOS 27, VideoToolbox frame-rate conversion has known problems:

- At small frame sizes, up to about 704x512, its new frames come out close to a blend of their
  neighbors instead of interpolated motion. The default 1024x576 is not affected, but its
  half-size stage-1 decode, 512x288, is.
- When both sides of the frames it receives are between 1441 and 1920 pixels, it fails to start.
  The failure comes after generation, so save the latents on long runs you plan to interpolate:
  a [decode restart](restart.md) can then retry without regenerating.
- `--vsr-temporal-mode high` drifts on static shots. Keep the default `normal`.

HDR runs keep the generation frame rate.

## Scene cuts

`balanced` upscaling and frame-rate conversion use neighboring frames. Across a hard cut in the
generated video, that could carry the previous shot into the next one. Cut detection prevents it,
and is on by default:

| `--cut-detect-mode` | Behavior |
| --- | --- |
| `simple` (default) | A light frame-difference detector |
| `hist` | A histogram detector, steadier on motion-heavy footage |
| `off` | No detection |

`--cut-detect-threshold` sets how different two frames must be to count as a cut: lower finds
more. The defaults are 0.25 for `simple` and 0.5 for `hist`. Detection costs nothing when no step
uses neighboring frames.

`--vsr-save-original` also keeps the video as it was before upscaling or interpolation, as
`<stem>_orig.mp4`, with the same audio. `--save-all-sidecars` turns it on.

## Encode

The video is HEVC, 10-bit, tagged BT.709; HDR runs encode BT.2020 HLG. `--encode-quality` sets
VideoToolbox's quality from 0 to 1, 0.65 by default: higher is larger and closer to the decoded
frames.

Audio, when generated, is muxed as Apple Lossless (`--audio-codec alac`) or AAC
(`--audio-codec aac`).

`--audio-onset-trim` repairs a click that generated audio can have at its very start:

| Value | Behavior |
| --- | --- |
| `auto` (default) | When a click is detected, silence the first 120 ms |
| `off` | Leave the start untouched |
| `MS` | Always silence this many milliseconds; `0` is the same as `off` |

The audio keeps its length either way, so it stays in sync. In TOML a millisecond value may be
bare or quoted, `audio_onset_trim = 120` or `"120"`.

## Output files

Without `--output`, the video is named `<prefix>_YYYYMMDD_HHMMSS.mp4` in the output directory,
following the [shared naming rules](../USAGE.md#output-files). The prefix defaults to
`ltx2-distilled-seed<seed>` and the directory to `outputs/`.

`--output /exact/path.mp4` writes exactly that file. It must end in `.mp4`, checked before
generation starts, and an existing file is refused: LTX2 has no replacement option.

Every sidecar shares the video's directory and stem.

## Sidecars

| Option | Writes | Why you would want it |
| --- | --- | --- |
| `--save-run-log` | `<stem>_run.json` | Timings, memory, settings, and the input to a restart |
| `--save-console-log` | `<stem>_console.log` | The log as shown |
| `--save-effective-config` | `<stem>_config.toml` | Rerun the same invocation |
| `--save-latents` | `<stem>_stage1.safetensors`, `<stem>.safetensors` | [Decode again or rerun stage 2](restart.md) |
| `--save-text-conditioning` | `<stem>_text.safetensors` | Skip text encoding next time |
| `--save-media-conditioning` | `<stem>_stage1_conditioning.safetensors`, `_stage2_` | The exact encoded image conditions |
| `--save-audio-sidecar` | `<stem>.wav` | The audio on its own |
| `--save-vae-frames` | `<stem>_vae_frames/` | Lossless PNG frames straight from the decoder |
| `--save-hdr-heic-frames` | `<stem>_heic/` | Viewable HDR frames |
| `--vsr-save-original` | `<stem>_orig.mp4` | The video before upscaling or interpolation |

`--save-all-sidecars` selects all of them except the two frame sequences, which can be very
large. Sidecars that do not apply are skipped: no audio file for a silent run, and no media
conditioning without image conditions. An individual option still overrides save-all, so
`--no-save-console-log` drops just the log.

The effective config is written before generation, so it survives a failed run. It refuses to
replace an existing file, and reloading one whose video already exists is refused: pick a new
`--output`.

`--save-vae-frames` writes each decoded SDR frame as PNG before upscaling, interpolation, or
encoding, with a `manifest.json` describing the 8-bit mapping. Use it to inspect the decoder's
output without codec loss. HDR runs already write a lossless EXR sequence and refuse it.
