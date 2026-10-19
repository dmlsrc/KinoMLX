# Model files and loading

KinoMLX reads model files already on your Mac. It never downloads weights during a run, and it
never writes a second, converted copy of a model to disk. Download the files for a model before
running it.

- [Get the right files](#get-the-right-files)
- [Choose where models load from](#choose-where-models-load-from)
- [What loading costs](#what-loading-costs)
- [Precision and checks](#precision-and-checks)
- [Files, caches, and saved results](#files-caches-and-saved-results)
- [Convert a checkpoint](#convert-a-checkpoint)

## Get the right files

| Model | Download instructions |
| --- | --- |
| LTX-2.3 / LTX-2.5 | [LTX2 model files](ltx2/downloads.md): transformer, text encoder, VAEs, and upscaler |
| Krea 2 Raw / Turbo | [Krea 2 model files](krea2/downloads.md): the model folder, plus optional decoder and adapters |
| FLUX.2 Klein | [Klein model files](flux2/downloads.md): one folder per variant |
| GMNet | [GMNet weights](gmnet/weights.md): download and convert a checkpoint once |

Keep Hugging Face downloads in their original layout. A sharded model needs every shard its index
names, plus its configuration and tokenizer files. Signing in does not grant access to a gated
model: accept or request access on its model page first.

## Choose where models load from

KinoMLX finds downloaded snapshots in the Hugging Face cache selected by `HF_HOME` or `--hf-home`.
To load from elsewhere:

- LTX2 takes `--weights-path`, plus component paths when needed.
- Krea 2 and Klein take `--source-path` for a model folder, or component paths.
- GMNet finds its converted weights, or takes `--weights-path`.

An explicit file must fit the selected model: a missing or mismatched file is never silently
replaced by another download. Each model's downloads page has examples, and
`kinomlx --model MODEL --dry-run` checks the result without loading weights.

## What loading costs

Each model part loads when its stage needs it and is released afterward: the text encoder, then
the denoiser, then the decoder. Because they are never all resident at once, the peak is lower. A
large single-file checkpoint is not loaded whole either; each part reads only its own weights.

Precision and layout changes are prepared lazily, on first use. The first denoising step can
therefore include loading the transformer and merging adapters, so the initial loading message
understates the cost: compare complete runs or phase timings.

With [LTX2 streaming](ltx2/performance.md#run-on-32-gb), only some transformer blocks stay loaded,
and the rest are read again each step. Reads overlap computation, but once the checkpoint no
longer fits in the file cache, the drive's speed becomes part of every step.

## Precision and checks

LTX2 runs its transformer in FP16 by default, and Krea 2 and Klein in BF16; see
[precision](PERFORMANCE.md#precision). Both use two bytes per weight. Choosing a dtype never
rewrites the source file, and FP16 from a BF16 source cannot restore precision the source lacks.
Supported FP8 LTX sources are decoded into the chosen dtype.

Before using a component, KinoMLX checks its tensor names, shapes, and storage dtypes. Generation
also checks every denoising step's latents for nonfinite values, and adapter-changed weights get a
[first-use check](ADAPTERS.md#first-use-weight-checks).

If loading fails, read the component and file named in the error. Check that the download is
complete, that the variant is right, and that an explicit path does not mix incompatible
versions. If FP16 produces nonfinite values, try BF16 and check adapter strengths. Do not turn
checks off to make an incompatible file appear to work.

Loading records each source's identity, but it does not hash every weight, so a successful load is
not a full integrity check. Verify download hashes separately when you suspect corruption.

## Files, caches, and saved results

| Storage | Holds | Affected by `--mlx-cache-limit-gb`? |
| --- | --- | --- |
| Hugging Face cache or your model folder | The downloaded model files | No |
| MLX allocator cache | Freed memory kept for reuse | Yes; `0` keeps none |
| Application cache | Converted GMNet weights and the profiling helper | No |
| Saved latents and conditioning | Optional sidecars that a restart reads back | No |
| Klein reference KV | Reference keys and values, within one generation | No |

The application cache is `~/Library/Caches/KinoMLX` by default, or `XDG_CACHE_HOME/KinoMLX` when
that is set; `--app-cache-dir`, `[settings].app_cache_dir`, and `KINO_APP_CACHE_DIR` move it. Keep
the original model files for restarts: saved latents and conditioning do not include the remaining
weights. Memory limits are in [Performance and memory](PERFORMANCE.md).

## Convert a checkpoint

Runs load safetensors only. GMNet publishes PyTorch checkpoints, so it has its own converter; see
[GMNet weights](gmnet/weights.md). For any other plain PyTorch state dict, the generic converter
copies its tensors unchanged:

```bash
kinomlx weights convert checkpoint.pth -o checkpoint.safetensors
```

Use it only when no model has its own converter: it does not transpose tensors or check them
against any model. Non-tensor entries are dropped with a warning, and the prefix options below
choose and rename keys.

It reads the checkpoint with a restricted reader that never runs code from it, and refuses old
stream-format checkpoints and float64 storage. A bare filename is also looked up under
`weights-src/` in the current directory.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>-o</code>, <code>--output</code></td><td><em>the input with a .safetensors suffix</em></td><td><code>PATH.safetensors</code></td><td>Where to write the converted file. Required for a source under <code>weights-src/</code>, so converted files never land there.</td></tr>
<tr><td><code>--param-key</code></td><td><em>automatic</em></td><td><code>KEY</code></td><td>The nested mapping to extract. A single <code>state_dict</code>, <code>model</code>, <code>net</code>, <code>weights</code>, <code>params</code>, or <code>params_ema</code> mapping is found automatically; name one when there are several, or when tensors also sit at the top level.</td></tr>
<tr><td><code>--strip-prefix</code></td><td><code>module.</code></td><td><code>PREFIX</code></td><td>Remove this prefix from key names where present; <code>''</code> keeps them.</td></tr>
<tr><td><code>--only-prefix</code></td><td><em>every key</em></td><td><code>PREFIX</code></td><td>Keep only keys that start with this prefix, matched before <code>--strip-prefix</code> removes anything.</td></tr>
<tr><td><code>--force</code>, <code>--no-force</code></td><td><em>off</em></td><td><em>flag</em></td><td>Replace an existing output; otherwise refuse before loading.</td></tr>
<tr><td><code>--allow-suspicious-globals</code>, <code>--no-allow-suspicious-globals</code></td><td><em>off</em></td><td><em>flag</em></td><td>Convert a checkpoint you trust that the static scan refused; the restricted reader still refuses anything outside its allowlist.</td></tr>
</tbody>
</table>

`kinomlx weights --help` lists the converters. They also take the memory, logging, profiling, and
location options among the [shared arguments](CONFIG.md#shared-arguments).
