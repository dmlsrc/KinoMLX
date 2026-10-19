# HDR video

LTX2 makes HDR video in two validated ways, one per version. Each needs an HDR source to anchor
the signal, so text alone is refused: a prompt cannot establish highlights brighter than SDR
white. Combinations outside these two workflows are refused rather than approximated.

| Version | Input | Workflow |
| --- | --- | --- |
| LTX-2.3 | An SDR video | [Convert SDR video to HDR](#ltx-23-convert-sdr-video-to-hdr) with the HDR IC-LoRA |
| LTX-2.5 | An HDR still (EXR) | [Animate an HDR still](#ltx-25-animate-an-hdr-still) |

## What an HDR run writes

`--hdr PROFILE` selects the working space of the EXR master:

| `--hdr` | EXR master |
| --- | --- |
| `SRGB_LINEAR` | Scene-linear Rec.709 |
| `ACESCG` | Scene-linear ACEScg |
| `ACESCCT` | ACEScct working space |

Every HDR run writes a half-float EXR frame sequence, `<stem>_exr/`, as the master, and a
BT.2020 HLG HEVC video for viewing. `--save-hdr-heic-frames` adds `<stem>_heic/`: 10-bit
BT.2100 PQ HEIC frames you can open in Preview, Quick Look, and Photos. They map scene-linear 1.0
to the 203-nit HDR reference white. HEIC is lossy, so keep the EXR as the master; it is a second
large sequence, so `--save-all-sidecars` does not select it.

HDR runs cannot be upscaled or change frame rate yet. `--save-vae-frames` is refused, because the
EXR sequence is already lossless.

## LTX-2.3: convert SDR video to HDR

Lightricks' HDR IC-LoRA converts an SDR video into an HDR one, frame for frame. It lives in its
own gated repository,
[Lightricks/LTX-2.3-22b-IC-LoRA-HDR](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR),
under the LTX-2 Community License; accept access on that page first. KinoMLX uses one file from
it:

```bash
hf download Lightricks/LTX-2.3-22b-IC-LoRA-HDR \
  ltx-2.3-22b-ic-lora-hdr-0.9.safetensors --local-dir /models/ltx23-hdr
```

There is no install step: pass the file to `--adapter`, with the SDR source as `--hdr-reference`:

```bash
kinomlx --model ltx2 --model-generation 2.3 \
  --adapter /models/ltx23-hdr/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors \
  --hdr-reference /videos/source-sdr.mov \
  --hdr SRGB_LINEAR \
  --prompt "Preserve the source scene with natural motion and detailed highlights" \
  --width 768 --height 448 --frames 121 \
  --output-dir outputs --output-prefix source-hdr
```

The reference must contain at least the requested number of frames. KinoMLX reads that many from
its start, resizes them to the generation size, keeping the aspect ratio, and pads the rest. A
shorter reference is refused, because the adapter misbehaves on frames without one.

Before loading the model, KinoMLX checks the adapter's own metadata. The run needs exactly one
adapter that declares `hdr_transform=logc3` and `reference_downscale_factor=1`; the official file
declares both. The repository's separate `scene-emb` file is not used.

The HDR adapter must have a positive strength in both stages, cannot take exclusions, and must
place on every target it carries. Its strength and the reference's default to 1.0;
`--adapter-strength` and `--hdr-reference-strength` change them deliberately.

The recipe is video only, as validated: it refuses `--generate-audio`, image conditions, and
generated keyframes. Other adapters may stack beside the HDR one, but the validated recipe uses it
alone, without streaming or quantization.

## LTX-2.5: animate an HDR still

LTX-2.5 makes HDR video without an adapter, from an HDR EXR image at frame 0:

```bash
kinomlx --model ltx2 --model-generation 2.5 \
  --image /images/hdr-plate.exr --hdr ACESCG \
  --prompt "A slow cinematic push through the illuminated scene" \
  --width 768 --height 448 --frames 121 \
  --output-dir outputs --output-prefix hdr-plate-motion
```

The model keeps the exposure of the image, so the EXR must really contain highlights above SDR
white. An EXR converted from an SDR picture gives an SDR-looking video in an HDR file, and
KinoMLX warns when it sees one. [GMNet](../gmnet/README.md) expands SDR photographs into suitable
HDR plates.

Later keyframes may be SDR images or EXRs; only frame 0 sets the HDR signal.
