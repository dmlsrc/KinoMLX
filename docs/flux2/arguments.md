# FLUX.2 Klein arguments

Every `kinomlx --model flux2` argument, grouped by task. Arguments every model shares, such as
memory limits and `--json`, are in [shared arguments](../CONFIG.md#shared-arguments).

Most arguments also have a TOML field and an environment variable, and the adapter flags set members
of an `adapters` record instead: `kinomlx config init --model flux2` writes them all with their
defaults, and [Configuration](../CONFIG.md) explains the naming. The [last
section](#config-file-names) lists the names that differ from their flags.

- [Prompt](#prompt)
- [Size and count](#size-and-count)
- [Sampling](#sampling)
- [References](#references)
- [Adapters](#adapters)
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
<tr><td><code>--prompt-file</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Read one complete prompt from a UTF-8 file, keeping its line breaks. See <a href="../USAGE.md#prompts">prompts</a>.</td></tr>
<tr><td><code>--prompt-unconditional</code>, <code>--no-prompt-unconditional</code></td><td><em>off</em></td><td><em>flag</em></td><td>Generate from a deliberately empty prompt; text encoding still runs.</td></tr>
<tr><td><code>--negative-prompt</code></td><td><em>empty</em></td><td><code>TEXT</code></td><td>Base with guidance above 1 only: what to steer away from. One value for every prompt, or one per prompt.</td></tr>
</tbody>
</table>

## Size and count

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--width</code></td><td><code>1024</code></td><td><code>PIXELS</code> (at least 64)</td><td>Output width; a multiple of 16 unless the reference's aspect is matched, which rounds it up. With one reference and no size, the output follows the reference's shape.</td></tr>
<tr><td><code>--height</code></td><td><code>1024</code></td><td><code>PIXELS</code> (at least 64)</td><td>Output height, aligned the same way.</td></tr>
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
<tr><td><code>--steps</code></td><td><code>50</code> (base), <code>4</code> (distilled, KV)</td><td><code>N</code></td><td>Denoising steps. Fewer steps on base need a Turbo LoRA; see <a href="adapters.md#turbo-loras-for-base">Turbo LoRAs</a>.</td></tr>
<tr><td><code>--guidance</code></td><td><code>4</code> (base), <code>1</code> (distilled, KV)</td><td><code>G</code></td><td>Guidance scale for base, at least 1. Distilled and KV take no external guidance, so it must stay 1.</td></tr>
<tr><td rowspan="2" valign="top"><code>--cfg-mode</code></td><td rowspan="2" valign="top"><code>sequential</code></td><td><code>sequential</code></td><td>Base guidance: run the positive and negative branches one after the other, using less memory.</td></tr>
<tr><td><code>batched</code></td><td>Base guidance: run both branches together, with more memory and slightly different rounding. Refused on distilled and KV.</td></tr>
<tr><td><code>--sigmas</code></td><td><em>derived</em></td><td><code>S,S,...</code></td><td>Set the noise levels directly; their count fixes the step count. See <a href="../SCHEDULES.md#setting-the-schedule">schedules</a>.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-backend</code></td><td rowspan="2" valign="top"><code>mlx</code></td><td><code>mlx</code></td><td>KinoMLX's own noise; the normal choice.</td></tr>
<tr><td><code>torch-mps</code></td><td>Reproduce PyTorch MPS noise. A PyTorch CPU generator gives different noise.</td></tr>
</tbody>
</table>

## References

See [Image editing](editing.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--source-image</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Add a reference image. Repeat for several; their order matters.</td></tr>
<tr><td rowspan="4" valign="top"><code>--match-source-aspect</code></td><td rowspan="4" valign="top"><em>automatic with one reference</em></td><td><code>long</code></td><td>Pin the longer side to the one size you give (<code>--width</code> or <code>--height</code>, default 1024); the other follows the first reference. The bare flag means <code>long</code>.</td></tr>
<tr><td><code>short</code></td><td>Pin the shorter side the same way.</td></tr>
<tr><td><code>width</code></td><td>Pin the width to <code>--width</code>, else 1024; the height follows the reference. Refused with <code>--height</code>.</td></tr>
<tr><td><code>height</code></td><td>Pin the height to <code>--height</code>, else 1024; the width follows the reference. Refused with <code>--width</code>.</td></tr>
<tr><td rowspan="2" valign="top"><code>--reference-fit</code></td><td rowspan="2" valign="top"><em>fit with one reference</em></td><td><code>fit</code></td><td>Fit the whole reference into the output grid, centered.</td></tr>
<tr><td><code>crop</code></td><td>Fill the output grid with a centered crop of the reference.</td></tr>
<tr><td><code>--recompute-reference-kv</code>, <code>--no-recompute-reference-kv</code></td><td><em>off</em></td><td><em>flag</em></td><td>KV variant only: recompute the cached reference keys and values every step. Needed when an adapter's strength varies by step or sigma window while editing; slower.</td></tr>
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

## Output files

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--output</code></td><td><em>timestamped name</em></td><td><code>PATH</code></td><td>Write exactly this <code>.png</code>, <code>.tif</code>, or <code>.tiff</code> file. For several images it is the stem the numbered names start from.</td></tr>
<tr><td><code>--output-dir</code></td><td><code>outputs</code></td><td><code>PATH</code></td><td>Directory for timestamped names; see <a href="../USAGE.md#output-files">output files</a> for the fallbacks.</td></tr>
<tr><td><code>--output-prefix</code></td><td><code>flux2-&lt;variant&gt;-seed&lt;seed&gt;</code></td><td><code>TEXT</code></td><td>Name before the timestamp.</td></tr>
<tr><td><code>--force</code>, <code>--no-force</code></td><td><em>off</em></td><td><em>flag</em></td><td>Replace existing files at an exact <code>--output</code>. A run's inputs are never replaced, even through a link.</td></tr>
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
<tr><td><code>--save-text-conditioning</code>, <code>--no-save-text-conditioning</code>, <code>--save-text-embeddings</code>, <code>--no-save-text-embeddings</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_text.safetensors</code>: the encoded prompt, negative prompt, and references, for a denoise restart.</td></tr>
<tr><td><code>--save-initial-noise</code>, <code>--no-save-initial-noise</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_initial_noise.safetensors</code>: the starting noise and its schedule.</td></tr>
<tr><td><code>--save-final-latents</code>, <code>--no-save-final-latents</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;image&gt;_final_latents.safetensors</code>: the final latent, for a decode restart.</td></tr>
</tbody>
</table>

## Model files and precision

See [Model files](downloads.md) and [Performance and memory](performance.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="3" valign="top"><code>--variant</code></td><td rowspan="3" valign="top"><code>distilled</code></td><td><code>distilled</code></td><td>Four steps, no external guidance.</td></tr>
<tr><td><code>kv</code></td><td>A separate four-step checkpoint that caches reference keys and values, for faster editing.</td></tr>
<tr><td><code>base</code></td><td>The undistilled model: 50 steps with guidance, and the target of Turbo LoRAs.</td></tr>
<tr><td><code>--source-path</code></td><td><em>Hugging Face cache</em></td><td><code>PATH</code></td><td>A local model folder to use instead of the cache.</td></tr>
<tr><td><code>--tokenizer-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the tokenizer.</td></tr>
<tr><td><code>--text-encoder-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the text encoder.</td></tr>
<tr><td><code>--transformer-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the transformer.</td></tr>
<tr><td><code>--vae-path</code></td><td><em>from the source</em></td><td><code>PATH</code></td><td>Replace only the VAE.</td></tr>
<tr><td rowspan="2" valign="top"><code>--transformer-dtype</code></td><td rowspan="2" valign="top"><code>bfloat16</code></td><td><code>bfloat16</code></td><td>The bundled weights' own precision.</td></tr>
<tr><td><code>float16</code></td><td>Optional speed setting with a narrower range; no memory saving.</td></tr>
<tr><td rowspan="2" valign="top"><code>--vae-dtype</code></td><td rowspan="2" valign="top"><code>bfloat16</code></td><td><code>bfloat16</code></td><td>The bundled decoder precision.</td></tr>
<tr><td><code>float32</code></td><td>FP32 decoding, for diagnostic comparisons. A run decoded in FP32 cannot be restarted.</td></tr>
</tbody>
</table>

## Restart

See [Restart from saved stages](restart.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--restart</code></td><td><em>none</em></td><td><code>RUN.json</code></td><td>Restart from a saved run record.</td></tr>
<tr><td><code>--text-conditioning</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Replay saved conditioning, prompt and references included, without a run record; or replace it at a denoise restart. Refuses new references and more than one prompt.</td></tr>
<tr><td rowspan="2" valign="top"><code>--restart-from</code></td><td rowspan="2" valign="top"><code>decode</code></td><td><code>decode</code></td><td>Decode the saved final latent again.</td></tr>
<tr><td><code>denoise</code></td><td>Denoise again from the saved conditioning, extracting reference KV afresh.</td></tr>
<tr><td><code>--restart-latents</code></td><td><em>recorded in the run</em></td><td><code>PATH</code></td><td>Decode this latent file instead of the recorded one.</td></tr>
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
| `--source-image` | `generate.source_images`, a list | `KINO_FLUX2_SOURCE_IMAGES` |
| `--adapter` and the flags that modify it | `generate.adapters` records with `path`, `strength`, `exclude`, `step_strengths`, `sigma_window` | `KINO_FLUX2_ADAPTERS` |
