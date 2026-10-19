# Krea 2 arguments

Every `kinomlx --model krea2` argument, grouped by task. Arguments every model shares, such as
memory limits and `--json`, are in [shared arguments](../CONFIG.md#shared-arguments).

Most arguments also have a TOML field and an environment variable, and the adapter flags set members
of an `adapters` record instead: `kinomlx config init --model krea2` writes them all with their
defaults, and [Configuration](../CONFIG.md) explains the naming. The [last
section](#config-file-names) lists the names that differ from their flags.

- [Prompt](#prompt)
- [Size and count](#size-and-count)
- [Sampling](#sampling)
- [Adapters](#adapters)
- [Four-step and diversity profiles](#four-step-and-diversity-profiles)
- [Image editing](#image-editing)
- [Decoder](#decoder)
- [Output files](#output-files)
- [Sidecars](#sidecars)
- [Model files and precision](#model-files-and-precision)
- [Restart](#restart)
- [Inspection](#inspection)
- [Config file names](#config-file-names)

## Prompt

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--prompt</code></td><td><em>none</em></td><td><code>TEXT</code></td><td>Describe the image. Repeat it for several prompts, run in order.</td></tr>
<tr><td><code>--prompt-file</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Read one complete prompt from a UTF-8 file, keeping its line breaks; not with <code>--prompt</code>. See <a href="../USAGE.md#prompts">prompts</a>.</td></tr>
<tr><td><code>--prompt-unconditional</code>, <code>--no-prompt-unconditional</code></td><td><em>off</em></td><td><em>flag</em></td><td>Generate from a deliberately empty prompt; text encoding still runs.</td></tr>
<tr><td><code>--negative-prompt</code></td><td><em>empty</em></td><td><code>TEXT</code></td><td>What to steer away from, when guidance is above 0 (Raw's default). One value for every prompt, or one per prompt.</td></tr>
<tr><td><code>--system-prompt</code></td><td><em>Krea's training instruction</em></td><td><code>TEXT</code></td><td>Replace the Qwen system instruction for text-to-image generation. <code>""</code> keeps an empty one. Not for editing.</td></tr>
<tr><td><code>--text-conditioning</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Reuse a saved encoded prompt (<code>&lt;image&gt;_text.safetensors</code>) and skip text encoding. Its prompts come with it; see <a href="restart.md#reuse-an-encoded-prompt-without-a-run-record">replay</a>.</td></tr>
</tbody>
</table>

## Size and count

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--width</code></td><td><code>1024</code></td><td><code>PIXELS</code></td><td>Output width. Rounded up to a multiple of 16 for generation, then cropped back exactly.</td></tr>
<tr><td><code>--height</code></td><td><code>1024</code></td><td><code>PIXELS</code></td><td>Output height, aligned the same way.</td></tr>
<tr><td><code>--seed</code>, <code>--seeds</code></td><td><code>0</code></td><td><code>N [N ...]</code></td><td>One seed or an ordered list. See <a href="../USAGE.md#seeds">seeds</a>.</td></tr>
<tr><td><code>--images-per-prompt</code></td><td><code>1</code></td><td><code>N</code></td><td>Make N images per prompt, with consecutive seeds starting at <code>--seed</code>.</td></tr>
<tr><td rowspan="3" valign="top"><code>--seed-mode</code></td><td rowspan="3" valign="top"><code>sequential</code></td><td><code>sequential</code></td><td>Each seed runs through every stage with normal model loading.</td></tr>
<tr><td><code>stage</code></td><td>Denoise every seed with one transformer load, then decode every seed.</td></tr>
<tr><td><code>resident</code></td><td>Keep the transformer and decoder loaded across seeds.</td></tr>
</tbody>
</table>

## Sampling

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--steps</code></td><td><code>8</code> (Turbo), <code>52</code> (Raw)</td><td><code>N</code></td><td>Denoising steps.</td></tr>
<tr><td><code>--guidance</code></td><td><code>0</code> (Turbo), <code>3.5</code> (Raw)</td><td><code>G</code></td><td>Guidance strength. <code>0</code> skips the negative branch entirely. Krea's guidance is an offset: 2 equals conventional CFG 3.</td></tr>
<tr><td><code>--mu</code></td><td><code>1.15</code> (Turbo), <em>by resolution</em> (Raw)</td><td><code>MU</code></td><td>Scheduler shift; setting it turns off Raw's resolution-dependent shift.</td></tr>
<tr><td><code>--sigmas</code></td><td><em>derived</em></td><td><code>S,S,...</code></td><td>Set the noise levels directly; their count fixes the step count. See <a href="../SCHEDULES.md#setting-the-schedule">schedules</a>.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-backend</code></td><td rowspan="2" valign="top"><code>mlx</code></td><td><code>mlx</code></td><td>KinoMLX's own noise; the normal choice.</td></tr>
<tr><td><code>torch-mps</code></td><td>Reproduce PyTorch MPS noise, to compare against tools that use it.</td></tr>
</tbody>
</table>

## Adapters

See [Adapters](adapters.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--adapter</code>, <code>--lora</code></td><td><em>none</em></td><td><code>PATH[:S]</code></td><td>Add an adapter, optionally with its strength. Repeat to stack.</td></tr>
<tr><td><code>--adapter-strength</code>, <code>--lora-strength</code></td><td><code>1.0</code></td><td><code>S</code></td><td>Strength for the latest <code>--adapter</code>. Zero and negative values are allowed.</td></tr>
<tr><td><code>--adapter-exclude</code>, <code>--lora-exclude</code></td><td><em>none</em></td><td><code>CATEGORY,...</code></td><td>Leave these parts of the model untouched by the latest <code>--adapter</code>. See <a href="adapters.md#exclude-parts-of-the-model">categories</a>.</td></tr>
<tr><td><code>--adapter-step-strengths</code>, <code>--lora-step-strengths</code></td><td><em>constant</em></td><td><code>S,S,...</code></td><td>One strength per denoising step for the latest <code>--adapter</code>. See <a href="../SCHEDULES.md#one-strength-per-step">strength over a run</a>.</td></tr>
<tr><td><code>--adapter-sigma-window</code>, <code>--lora-sigma-window</code></td><td><em>always on</em></td><td><code>LOW[:HIGH][@S],...</code></td><td>Apply the latest <code>--adapter</code> only at these noise levels. See <a href="../SCHEDULES.md#strength-by-noise-level">strength by noise level</a>.</td></tr>
</tbody>
</table>

`--check-adapter-weights` is a [shared argument](../CONFIG.md#checks-logging-and-profiling).

## Four-step and diversity profiles

Two community adapters have their own switches; see [the profiles](adapters.md#four-step-turbo).
Each other option here needs its switch, and a profile's step list refuses its `-strength` option.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--turbo-4step</code>, <code>--no-turbo-4step</code></td><td><em>off</em></td><td><em>flag</em></td><td>Use the community four-step Turbo LoRA: 4 steps, no guidance, <code>mu</code> 1.15. On Raw it warns that a standard 8-step Turbo LoRA is also needed.</td></tr>
<tr><td><code>--turbo-4step-strength</code></td><td><code>1.0</code></td><td><code>S</code> (positive)</td><td>Strength of the four-step adapter.</td></tr>
<tr><td><code>--turbo-4step-exclude</code></td><td><em>none</em></td><td><code>CATEGORY,...</code> or <code>none</code></td><td>Exclusions for the four-step adapter only.</td></tr>
<tr><td><code>--turbo-4step-step-strengths</code></td><td><em>constant</em></td><td><code>S,S,...</code></td><td>One strength per step for the four-step adapter.</td></tr>
<tr><td><code>--turbo-4step-sigma-window</code></td><td><em>always on</em></td><td><code>LOW[:HIGH][@S],...</code></td><td>Noise levels at which the four-step adapter applies.</td></tr>
<tr><td><code>--turbo-sda</code>, <code>--no-turbo-sda</code></td><td><em>off</em></td><td><em>flag</em></td><td>Use the community SDA adapter, which restores seed-to-seed variety to Turbo. By default it applies at noise levels from 0.93, its training floor, up to 1.</td></tr>
<tr><td><code>--turbo-sda-strength</code></td><td><code>1.0</code></td><td><code>S</code></td><td>Strength of the SDA adapter inside its window.</td></tr>
<tr><td><code>--turbo-sda-exclude</code></td><td><em>none</em></td><td><code>CATEGORY,...</code> or <code>none</code></td><td>Exclusions for the SDA adapter only.</td></tr>
<tr><td><code>--turbo-sda-step-strengths</code></td><td><em>window</em></td><td><code>S,S,...</code></td><td>One strength per step for the SDA adapter, instead of its window.</td></tr>
<tr><td><code>--turbo-sda-sigma-window</code></td><td><em>0.93 up to 1</em></td><td><code>LOW[:HIGH][@S],...</code></td><td>Noise levels at which the SDA adapter applies.</td></tr>
</tbody>
</table>

## Image editing

See [Image editing](editing.md). The options after `--source-image` need it.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--source-image</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Edit this image. Repeat for two references, scene first, then subject.</td></tr>
<tr><td><code>--edit-adapter</code></td><td><em>cached default</em></td><td><code>PATH</code></td><td>The edit adapter. Without it, the community identity-edit adapter is found in the Hugging Face cache.</td></tr>
<tr><td><code>--edit-adapter-strength</code></td><td><code>1.0</code></td><td><code>S</code> (nonzero)</td><td>Strength of the edit adapter.</td></tr>
<tr><td><code>--edit-adapter-exclude</code></td><td><em>none</em></td><td><code>CATEGORY,...</code> or <code>none</code></td><td>Exclusions for the edit adapter only.</td></tr>
<tr><td rowspan="4" valign="top"><code>--match-source-aspect</code></td><td rowspan="4" valign="top"><em>off</em></td><td><code>long</code></td><td>Pin the longer side to the one size you give (<code>--width</code> or <code>--height</code>, default 1024); the other follows the source. The bare flag means <code>long</code>.</td></tr>
<tr><td><code>short</code></td><td>Pin the shorter side the same way.</td></tr>
<tr><td><code>width</code></td><td>Pin the width to <code>--width</code>, else 1024; the height follows the source.</td></tr>
<tr><td><code>height</code></td><td>Pin the height to <code>--height</code>, else 1024; the width follows the source.</td></tr>
<tr><td rowspan="2" valign="top"><code>--reference-fit</code></td><td rowspan="2" valign="top"><code>fit</code></td><td><code>fit</code></td><td>Keep the source's aspect ratio, centered in the output grid.</td></tr>
<tr><td><code>crop</code></td><td>Fill the output with a centered crop, for older adapters trained that way.</td></tr>
<tr><td><code>--reference-strength</code></td><td><code>1.0</code></td><td><code>S</code> (nonnegative)</td><td>How strongly the output attends to the sole or second reference.</td></tr>
<tr><td><code>--scene-reference-strength</code></td><td><code>1.0</code></td><td><code>S</code> (nonnegative)</td><td>How strongly the output attends to the first of two references; needs two <code>--source-image</code>.</td></tr>
<tr><td><code>--reference-boost-mask</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Limit <code>--reference-strength</code> to the white areas of this mask, the size of the last source image.</td></tr>
<tr><td><code>--reference-boost-region</code></td><td><em>none</em></td><td><code>SHAPE:COORDS</code></td><td>Limit <code>--reference-strength</code> to a rectangle, circle, ellipse, or polygon in percent. Repeat to combine.</td></tr>
<tr><td><code>--grounding-system-prompt</code></td><td><em>training default</em></td><td><code>TEXT</code></td><td>Replace the Qwen system instruction used to describe the source images.</td></tr>
<tr><td><code>--grounding-px</code></td><td><code>768</code></td><td><code>PIXELS</code></td><td>Longest side of the source as the vision model sees it; <code>0</code> keeps its full size.</td></tr>
</tbody>
</table>

## Decoder

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--vae-decoder</code></td><td rowspan="2" valign="top"><code>qwen</code>, or as detected from <code>--vae-path</code></td><td><code>qwen</code></td><td>The stock Krea decoder.</td></tr>
<tr><td><code>upscale2x</code></td><td>A community decoder that outputs twice the width and height. See <a href="generation.md#decode-at-twice-the-size">the 2x decoder</a>.</td></tr>
<tr><td rowspan="3" valign="top"><code>--vae-tiling</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Decode the whole image if it fits, otherwise in the largest tiles that fit.</td></tr>
<tr><td><code>single</code></td><td>Decode the whole image at once.</td></tr>
<tr><td><code>custom</code></td><td>Use <code>--vae-tile-size</code> and <code>--vae-tile-stride</code>.</td></tr>
<tr><td><code>--vae-tile-size</code></td><td><code>256</code></td><td><code>PIXELS</code> (multiple of 8)</td><td>Custom tile size in pixels of the generated image, larger than the stride.</td></tr>
<tr><td><code>--vae-tile-stride</code></td><td><code>192</code></td><td><code>PIXELS</code> (multiple of 8)</td><td>Custom tile stride, smaller than the tile size.</td></tr>
<tr><td><code>--vae-dtype</code></td><td><code>bfloat16</code></td><td><code>bfloat16</code>, <code>float16</code>, <code>float32</code></td><td>Decoder precision.</td></tr>
</tbody>
</table>

## Output files

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--output</code></td><td><em>timestamped name</em></td><td><code>PATH</code></td><td>Write exactly this <code>.png</code>, <code>.tif</code>, or <code>.tiff</code> file. For several images it is the stem the numbered names start from.</td></tr>
<tr><td><code>--output-dir</code></td><td><code>outputs</code></td><td><code>PATH</code></td><td>Directory for timestamped names; see <a href="../USAGE.md#output-files">output files</a> for the fallbacks.</td></tr>
<tr><td><code>--output-prefix</code></td><td><code>krea2-&lt;variant&gt;-seed&lt;seed&gt;</code></td><td><code>TEXT</code></td><td>Name before the timestamp.</td></tr>
<tr><td><code>--force</code>, <code>--no-force</code></td><td><em>off</em></td><td><em>flag</em></td><td>Replace existing files at an exact <code>--output</code>; refused without one. Source images and masks are never replaced.</td></tr>
</tbody>
</table>

## Sidecars

Each sidecar follows `--save-all-sidecars` unless set itself. Tensor sidecars are written once
per output image.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--save-all-sidecars</code>, <code>--no-save-all-sidecars</code></td><td><em>off</em></td><td><em>flag</em></td><td>Save everything below that applies to the run.</td></tr>
<tr><td><code>--save-run-log</code>, <code>--no-save-run-log</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_run.json</code>: settings, timings, memory, and outputs. A restart reads it.</td></tr>
<tr><td><code>--save-console-log</code>, <code>--no-save-console-log</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_console.log</code>: the human-readable log.</td></tr>
<tr><td><code>--save-effective-config</code>, <code>--no-save-effective-config</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_config.toml</code>: the resolved invocation.</td></tr>
<tr><td><code>--save-text-conditioning</code>, <code>--no-save-text-conditioning</code>, <code>--save-text-embeddings</code>, <code>--no-save-text-embeddings</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_text.safetensors</code>: the encoded prompt, for a denoise restart. Text-to-image only.</td></tr>
<tr><td><code>--save-initial-noise</code>, <code>--no-save-initial-noise</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_initial_noise.safetensors</code>: the starting noise.</td></tr>
<tr><td><code>--save-final-latents</code>, <code>--no-save-final-latents</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_final_latents.safetensors</code>: the final latent, for a decode restart.</td></tr>
</tbody>
</table>

## Model files and precision

See [Model files](downloads.md) and [Performance and precision](performance.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--variant</code></td><td rowspan="2" valign="top"><code>turbo</code>, or as declared by <code>--source-path</code></td><td><code>turbo</code></td><td>The distilled model: fast, eight steps, no guidance.</td></tr>
<tr><td><code>raw</code></td><td>The base model: 52 steps with guidance, and the one adapters are trained on.</td></tr>
<tr><td><code>--source-path</code></td><td><em>Hugging Face cache</em></td><td><code>PATH</code></td><td>A local model folder to use instead of the cache.</td></tr>
<tr><td><code>--tokenizer-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the tokenizer.</td></tr>
<tr><td><code>--text-encoder-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the Qwen text encoder.</td></tr>
<tr><td><code>--transformer-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the transformer.</td></tr>
<tr><td><code>--vae-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace the VAE; its decoder type is detected from the file.</td></tr>
<tr><td rowspan="3" valign="top"><code>--transformer-dtype</code></td><td rowspan="3" valign="top"><code>bfloat16</code></td><td><code>bfloat16</code></td><td>Krea's default precision.</td></tr>
<tr><td><code>float16</code></td><td>Faster, with a narrower range; not with <code>--reference-denoise</code>.</td></tr>
<tr><td><code>float32</code></td><td>Full precision, at twice the memory.</td></tr>
<tr><td><code>--reference-denoise</code>, <code>--no-reference-denoise</code></td><td><em>off</em></td><td><em>flag</em></td><td>Denoise with the original padded, masked layout, to compare against reference implementations. Slower; needs a BF16 or FP32 transformer.</td></tr>
</tbody>
</table>

## Restart

See [Restart from saved stages](restart.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--restart</code></td><td><em>none</em></td><td><code>RUN.json</code></td><td>Restart from a saved run record.</td></tr>
<tr><td rowspan="2" valign="top"><code>--restart-from</code></td><td rowspan="2" valign="top"><code>decode</code></td><td><code>decode</code></td><td>Decode the saved final latent again.</td></tr>
<tr><td><code>denoise</code></td><td>Denoise again from the saved encoded prompt.</td></tr>
<tr><td><code>--restart-latents</code></td><td><em>recorded in the run</em></td><td><code>PATH</code></td><td>Decode restarts only: decode this latent file instead of the recorded one.</td></tr>
<tr><td><code>--restart-index</code></td><td><em>the only image</em></td><td><code>N</code></td><td>Which image of a multi-image run to resume, counting from 1.</td></tr>
</tbody>
</table>

## Inspection

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--dry-run</code></td><td><em>off</em></td><td><em>flag</em></td><td>Find and check every model file the run would use, then exit without loading weights.</td></tr>
</tbody>
</table>

## Config file names

Most fields are the flag's name with underscores, in the table `kinomlx config init` shows. These
differ; the shared ones, such as `--output`, are in
[Configuration](../CONFIG.md#names-that-differ-from-the-flag).

| Flags | Config fields | Environment variables |
| --- | --- | --- |
| `--turbo-4step-exclude`, `--turbo-sda-exclude`, `--edit-adapter-exclude` | `generate.turbo_4step_exclusions`, `generate.turbo_sda_exclusions`, `generate.edit_adapter_exclusions`, each a comma-separated string | `KINO_KREA2_TURBO_4STEP_EXCLUSIONS`, `KINO_KREA2_TURBO_SDA_EXCLUSIONS`, `KINO_KREA2_EDIT_ADAPTER_EXCLUSIONS` |
| `--edit-adapter` | `generate.edit_adapter_path` | `KINO_KREA2_EDIT_ADAPTER_PATH` |
| `--source-image` | `generate.source_images`, a list | `KINO_KREA2_SOURCE_IMAGES` |
| `--reference-boost-region` | `generate.reference_boost_regions`, a list | `KINO_KREA2_REFERENCE_BOOST_REGIONS` |
| `--adapter` and the flags that modify it | `generate.adapters` records with `path`, `strength`, `exclude`, `step_strengths`, `sigma_window` | `KINO_KREA2_ADAPTERS` |
