# LTX2 arguments

Every `kinomlx --model ltx2` argument, grouped by task. Arguments every model shares, such as memory
limits and `--json`, are in [shared arguments](../CONFIG.md#shared-arguments).

Most arguments also have a TOML field and an environment variable, and the adapter and image flags
set members of `adapters` and `images` records instead: `kinomlx config init --model ltx2` writes
them all with their defaults, and [Configuration](../CONFIG.md) explains the naming. The [last
section](#config-file-names) lists the names that differ from their flags.

- [Prompt](#prompt)
- [Size, length, and frame rate](#size-length-and-frame-rate)
- [Sampling](#sampling)
- [Image conditioning](#image-conditioning)
- [Adapters](#adapters)
- [Audio](#audio)
- [HDR](#hdr)
- [Video decoder](#video-decoder)
- [Upscaling, frame rate, and encoding](#upscaling-frame-rate-and-encoding)
- [Output files](#output-files)
- [Sidecars](#sidecars)
- [Model files](#model-files)
- [Memory and speed](#memory-and-speed)
- [Expert transformer tuning](#expert-transformer-tuning)
- [Restart](#restart)
- [Inspection](#inspection)
- [Config file names](#config-file-names)

## Prompt

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--prompt</code></td><td><em>none</em></td><td><code>TEXT</code></td><td>Describe the video. Required unless saved text conditioning, a restart, or <code>--prompt-unconditional</code> supplies the prompt.</td></tr>
<tr><td><code>--prompt-file</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Read one complete prompt from a UTF-8 file, keeping its line breaks. See <a href="../USAGE.md#prompts">prompts</a>.</td></tr>
<tr><td><code>--prompt-unconditional</code>, <code>--no-prompt-unconditional</code></td><td><em>off</em></td><td><em>flag</em></td><td>Generate from a deliberately empty prompt; text encoding still runs.</td></tr>
<tr><td><code>--text-conditioning</code></td><td><em>none</em></td><td><code>PATH</code></td><td>Reuse saved prompt tensors (<code>&lt;stem&gt;_text.safetensors</code>) and skip text encoding.</td></tr>
<tr><td><code>--reference-prompt-padding</code></td><td><em>on</em></td><td><em>flag</em></td><td>Pad the text encoder's input to 1,024 tokens, as the reference implementation does.</td></tr>
<tr><td><code>--compact-prompt</code></td><td><em>off</em></td><td><em>flag</em></td><td>Encode only the prompt's own tokens. This changes the conditioning, so the video differs from a padded run.</td></tr>
</tbody>
</table>

## Size, length, and frame rate

`--frames`, `--duration`, and `--auto-duration` are three ways to set the length; give only one.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--width</code></td><td><code>1024</code></td><td><code>N</code> (multiple of 64)</td><td>Output width in pixels.</td></tr>
<tr><td><code>--height</code></td><td><code>576</code></td><td><code>N</code> (multiple of 64)</td><td>Output height in pixels.</td></tr>
<tr><td><code>--frames</code></td><td><code>121</code></td><td><code>N</code> (<code>8*k+1</code>)</td><td>Frame count: 121 is five seconds at 24 fps.</td></tr>
<tr><td><code>--duration</code></td><td><em>none</em></td><td><code>SECONDS</code></td><td>Length in seconds; rounded up to the next valid frame count.</td></tr>
<tr><td><code>--auto-duration</code></td><td><em>off</em></td><td><em>flag</em></td><td>LTX-2.5 only: let the model's duration head choose a natural length for the prompt.</td></tr>
<tr><td><code>--fps</code></td><td><code>24</code></td><td><code>FPS</code></td><td>Frame rate the model generates for, and the rate the video plays at.</td></tr>
</tbody>
</table>

## Sampling

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--seed</code>, <code>--seeds</code></td><td><code>42</code></td><td><code>N [N ...]</code></td><td>One seed or an ordered list; each seed makes its own video. See <a href="../USAGE.md#seeds">seeds</a>.</td></tr>
<tr><td rowspan="3" valign="top"><code>--stage1-sampler</code>, <code>--sampler-stage-1</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>The sampler the checkpoint was distilled for: deterministic on LTX-2.3, ancestral on LTX-2.5.</td></tr>
<tr><td><code>deterministic</code></td><td>Deterministic Euler.</td></tr>
<tr><td><code>ancestral</code></td><td>RF ancestral Euler, which adds fresh noise each step.</td></tr>
<tr><td><code>--stage2-sampler</code>, <code>--sampler-stage-2</code></td><td><code>auto</code></td><td><code>auto</code>, <code>deterministic</code>, <code>ancestral</code></td><td>The same choice for stage 2. A stage-2 restart can change it.</td></tr>
<tr><td><code>--sigmas-stage-1</code>, <code>--stage1-sigmas</code></td><td><em>pinned schedule</em></td><td><code>S,S,...</code></td><td>Replace stage 1's noise levels; the count sets its step count. See <a href="../SCHEDULES.md#ltx2-has-two-of-them">schedules</a>.</td></tr>
<tr><td><code>--sigmas-stage-2</code>, <code>--stage2-sigmas</code></td><td><em>pinned schedule</em></td><td><code>S,S,...</code></td><td>Replace stage 2's noise levels.</td></tr>
<tr><td rowspan="2" valign="top"><code>--noise-backend</code></td><td rowspan="2" valign="top"><code>mlx</code></td><td><code>mlx</code></td><td>KinoMLX's own noise; the normal choice.</td></tr>
<tr><td><code>torch-mps</code></td><td>Reproduce PyTorch MPS noise, to compare against tools that use it.</td></tr>
<tr><td><code>--generated-keyframes</code></td><td><code>0</code></td><td><code>N</code></td><td>LTX-2.5 only: add N keyframes the model generates itself to shape the video. See <a href="generation.md#generated-keyframes">generated keyframes</a>.</td></tr>
</tbody>
</table>

## Image conditioning

See [Image conditioning](images.md) for the `--image` shorthand and how frame 0 differs from
later frames.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--image</code></td><td><em>none</em></td><td><code>PATH[@POSITION][:S]</code></td><td>Condition a frame on an image. Repeat for more frames. <code>POSITION</code> is <code>first</code>, <code>last</code>, or a frame number.</td></tr>
<tr><td><code>--image-frame-index</code></td><td><code>first</code></td><td><code>first</code>, <code>last</code>, <code>N</code></td><td>Frame for the latest <code>--image</code>.</td></tr>
<tr><td><code>--image-strength</code></td><td><code>1.0</code></td><td><code>0..1</code></td><td>Strength for the latest <code>--image</code>, in both stages.</td></tr>
<tr><td><code>--image-stage1-strength</code></td><td><em>common strength</em></td><td><code>0..1</code></td><td>Stage-1 strength for the latest <code>--image</code>.</td></tr>
<tr><td><code>--image-stage2-strength</code></td><td><em>common strength</em></td><td><code>0..1</code></td><td>Stage-2 strength for the latest <code>--image</code>.</td></tr>
<tr><td rowspan="2" valign="top"><code>--first-frame-mode</code></td><td rowspan="2" valign="top"><code>anchor</code></td><td><code>anchor</code></td><td>Hold frame 0 exactly at the encoded image.</td></tr>
<tr><td><code>guide</code></td><td>Denoise frame 0 too, guided by the image, so an adapter also changes the first frame.</td></tr>
</tbody>
</table>

## Adapters

See [Adapters](adapters.md) for stage strengths, exclusion categories, and recipes.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--adapter</code>, <code>--lora</code></td><td><em>none</em></td><td><code>PATH[:S]</code> or <code>PATH:S1:S2</code></td><td>Add an adapter. Repeat to stack; one number sets both stages, two set stage 1 and stage 2.</td></tr>
<tr><td><code>--adapter-strength</code>, <code>--lora-strength</code></td><td><code>1.0</code></td><td><code>S</code></td><td>Strength for the latest <code>--adapter</code>, in both stages.</td></tr>
<tr><td><code>--adapter-stage1-strength</code>, <code>--lora-stage1-strength</code></td><td><em>common strength</em></td><td><code>S</code></td><td>Stage-1 strength for the latest <code>--adapter</code>; <code>0</code> skips it in stage 1.</td></tr>
<tr><td><code>--adapter-stage2-strength</code>, <code>--lora-stage2-strength</code></td><td><em>common strength</em></td><td><code>S</code></td><td>Stage-2 strength for the latest <code>--adapter</code>; <code>0</code> skips it in stage 2.</td></tr>
<tr><td><code>--adapter-exclude</code>, <code>--lora-exclude</code></td><td><em>none</em></td><td><code>CATEGORY,...</code></td><td>Leave these parts of the model untouched by the latest <code>--adapter</code>. See <a href="adapters.md#exclude-parts-of-the-model">categories</a>.</td></tr>
<tr><td rowspan="3" valign="top"><code>--adapter-accumulation</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>FP16 fusion on a streamed FP16 transformer, FP32 otherwise.</td></tr>
<tr><td><code>float32</code></td><td>FP32 fusion everywhere; use it to compare streamed and resident output.</td></tr>
<tr><td><code>float16</code></td><td>FP16 fusion; needs <code>--transformer-dtype float16</code>.</td></tr>
</tbody>
</table>

`--check-adapter-weights` is a [shared argument](../CONFIG.md#checks-logging-and-profiling).

## Audio

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--generate-audio</code>, <code>--no-generate-audio</code></td><td><em>off</em></td><td><em>flag</em></td><td>Generate a stereo soundtrack with the video.</td></tr>
<tr><td><code>--reference-aligned-audio</code>, <code>--no-reference-aligned-audio</code></td><td><em>off</em></td><td><em>flag</em></td><td>Use the reference implementation's rounded audio length, which can end slightly before the video. For parity tests.</td></tr>
<tr><td rowspan="2" valign="top"><code>--audio-codec</code></td><td rowspan="2" valign="top"><code>alac</code></td><td><code>alac</code></td><td>Apple Lossless audio in the MP4.</td></tr>
<tr><td><code>aac</code></td><td>Smaller, lossy AAC audio.</td></tr>
<tr><td rowspan="3" valign="top"><code>--audio-onset-trim</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>When a click is detected at the very start, silence the first 120 ms.</td></tr>
<tr><td><code>off</code></td><td>Leave the start untouched.</td></tr>
<tr><td><code>MS</code></td><td>Always silence this many milliseconds at the start; <code>0</code> is the same as <code>off</code>. The audio keeps its length and sync.</td></tr>
</tbody>
</table>

## HDR

See [HDR video](hdr.md) for the two supported workflows.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="3" valign="top"><code>--hdr</code></td><td rowspan="3" valign="top"><em>off (SDR)</em></td><td><code>SRGB_LINEAR</code></td><td>Write scene-linear Rec.709 EXR masters.</td></tr>
<tr><td><code>ACESCG</code></td><td>Write scene-linear ACEScg EXR masters.</td></tr>
<tr><td><code>ACESCCT</code></td><td>Write ACEScct working-space EXR masters.</td></tr>
<tr><td><code>--hdr-reference</code></td><td><em>none</em></td><td><code>PATH</code></td><td>LTX-2.3 only: the SDR video the HDR IC-LoRA converts. It must cover every requested frame.</td></tr>
<tr><td><code>--hdr-reference-strength</code></td><td><code>1.0</code></td><td><code>0..1</code></td><td>Strength of the SDR reference.</td></tr>
</tbody>
</table>

## Video decoder

See [Video decoding](decode.md) for choosing a decoder and handling memory.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--video-vae</code></td><td rowspan="2" valign="top"><code>conv</code></td><td><code>conv</code></td><td>The convolutional decoder: fast, and the default.</td></tr>
<tr><td><code>diffusion</code></td><td>The LTX-2.5 diffusion decoder: slower, draws its own fine texture. Works with LTX-2.3 latents too.</td></tr>
<tr><td rowspan="4" valign="top"><code>--vae-decode-dtype</code></td><td rowspan="4" valign="top"><code>auto</code></td><td><code>auto</code></td><td>FP16, decoding again in BF16 if the first frames overflow.</td></tr>
<tr><td><code>float16</code></td><td>FP16 only; an overflow stops the run.</td></tr>
<tr><td><code>bfloat16</code></td><td>BF16: FP32's range at half its memory.</td></tr>
<tr><td><code>float32</code></td><td>FP32.</td></tr>
<tr><td rowspan="3" valign="top"><code>--vae-tiling</code></td><td rowspan="3" valign="top"><code>auto</code></td><td><code>auto</code></td><td>Choose tiles and cache size from free memory.</td></tr>
<tr><td><code>single</code></td><td>Decode in one pass.</td></tr>
<tr><td><code>custom</code></td><td>Use the tile sizes below.</td></tr>
<tr><td><code>--vae-temporal-tile-frames</code></td><td><em>auto</em>; <code>256</code> with <code>custom</code></td><td><code>N</code> (<code>0</code> or a multiple of 8, at least 16)</td><td>Frames per temporal tile; <code>0</code> disables temporal tiling.</td></tr>
<tr><td><code>--vae-temporal-overlap-frames</code></td><td><code>24</code></td><td><code>N</code> (multiple of 8)</td><td>Frames shared by neighboring temporal tiles.</td></tr>
<tr><td><code>--vae-spatial-tile-pixels</code></td><td><em>auto</em></td><td><code>N</code> (<code>0</code> or a multiple of 32, at least 64)</td><td>Pixels per spatial tile side; <code>0</code> disables spatial tiling.</td></tr>
<tr><td><code>--vae-spatial-overlap-pixels</code></td><td><code>64</code></td><td><code>N</code> (multiple of 32)</td><td>Pixels shared by neighboring spatial tiles.</td></tr>
<tr><td><code>--vae-cache-limit-gb</code></td><td><em>auto</em></td><td><code>GiB</code></td><td>Allocator cache while decoding. Unset, it is chosen with the tile plan.</td></tr>
<tr><td><code>--vae-sharpen</code></td><td><code>0</code></td><td><code>0..10</code></td><td>Diffusion decoder only: sharpen what it draws. About 1 is subtle; 3 to 4 is strong.</td></tr>
<tr><td><code>--vae-sharpen-radius</code></td><td><code>2</code></td><td><code>PIXELS</code> (at most 8)</td><td>Blur radius for <code>--vae-sharpen</code>; larger radii sharpen coarser shapes and can outline them.</td></tr>
</tbody>
</table>

## Upscaling, frame rate, and encoding

See [Upscaling, frame rate, and encoding](output.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="4" valign="top"><code>--vsr-spatial-mode</code></td><td rowspan="4" valign="top"><code>off</code></td><td><code>off</code></td><td>Keep the generated size.</td></tr>
<tr><td><code>fast</code></td><td>VideoToolbox 2x upscale with low latency; input up to 960x960.</td></tr>
<tr><td><code>balanced</code></td><td>VideoToolbox 4x upscale that uses the previous frame for crisper moving edges; input up to 1920x1080.</td></tr>
<tr><td><code>image</code></td><td>VideoToolbox 4x upscale, one frame at a time, the smoothest from frame to frame; input up to 1920x1080.</td></tr>
<tr><td><code>--target-fps</code></td><td><em>off</em></td><td><code>FPS</code></td><td>Interpolate new frames to reach this frame rate after upscaling.</td></tr>
<tr><td rowspan="2" valign="top"><code>--vsr-temporal-mode</code></td><td rowspan="2" valign="top"><code>normal</code></td><td><code>normal</code></td><td>Normal-quality frame interpolation.</td></tr>
<tr><td><code>high</code></td><td>Higher-quality interpolation; on macOS 27 it drifts on static shots, so prefer <code>normal</code>.</td></tr>
<tr><td rowspan="3" valign="top"><code>--cut-detect-mode</code></td><td rowspan="3" valign="top"><code>simple</code></td><td><code>simple</code></td><td>Detect hard cuts, so upscaling and interpolation do not blend the shots on either side of one.</td></tr>
<tr><td><code>hist</code></td><td>Histogram detection, steadier on motion-heavy footage.</td></tr>
<tr><td><code>off</code></td><td>No cut detection.</td></tr>
<tr><td><code>--cut-detect-threshold</code></td><td><code>0.25</code> (<code>simple</code>), <code>0.5</code> (<code>hist</code>)</td><td><code>FLOAT</code></td><td>How different two frames must be to count as a cut; lower finds more cuts.</td></tr>
<tr><td><code>--vsr-save-original</code>, <code>--no-vsr-save-original</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td>Also keep the video before upscaling or interpolation, as <code>&lt;stem&gt;_orig.mp4</code>.</td></tr>
<tr><td><code>--encode-quality</code></td><td><code>0.65</code></td><td><code>0..1</code></td><td>HEVC quality; higher is larger and closer to the decoded frames.</td></tr>
</tbody>
</table>

## Output files

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--output</code></td><td><em>timestamped name</em></td><td><code>PATH.mp4</code></td><td>Write exactly this file; it must end in <code>.mp4</code>. An existing file is refused.</td></tr>
<tr><td><code>--output-dir</code></td><td><code>outputs</code></td><td><code>PATH</code></td><td>Directory for timestamped names; see <a href="../USAGE.md#output-files">output files</a> for the fallbacks.</td></tr>
<tr><td><code>--output-prefix</code></td><td><code>ltx2-distilled-seed&lt;seed&gt;</code></td><td><code>TEXT</code></td><td>Name before the timestamp.</td></tr>
</tbody>
</table>

## Sidecars

Every sidecar except the frame sequences follows `--save-all-sidecars` unless set itself. See
[restart](restart.md#what-to-save) for which ones a restart needs.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--save-all-sidecars</code>, <code>--no-save-all-sidecars</code>, <code>--save-debug-sidecars</code>, <code>--no-save-debug-sidecars</code></td><td><em>off</em></td><td><em>flag</em></td><td>Save everything below except the two frame sequences.</td></tr>
<tr><td><code>--save-run-log</code>, <code>--no-save-run-log</code>, <code>--save-metadata</code>, <code>--no-save-metadata</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_run.json</code>: settings, timings, memory, and outputs. A restart reads it.</td></tr>
<tr><td><code>--save-console-log</code>, <code>--no-save-console-log</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_console.log</code>: the human-readable log.</td></tr>
<tr><td><code>--save-effective-config</code>, <code>--no-save-effective-config</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_config.toml</code>: the resolved invocation, written before generation starts.</td></tr>
<tr><td><code>--save-latents</code>, <code>--no-save-latents</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_stage1.safetensors</code> and <code>&lt;stem&gt;.safetensors</code>: the stage-1 and final latents.</td></tr>
<tr><td><code>--save-text-conditioning</code>, <code>--no-save-text-conditioning</code>, <code>--save-text-embeddings</code>, <code>--no-save-text-embeddings</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_text.safetensors</code>: the encoded prompt, reusable with <code>--text-conditioning</code>.</td></tr>
<tr><td><code>--save-media-conditioning</code>, <code>--no-save-media-conditioning</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_stage1_conditioning.safetensors</code> and <code>_stage2_</code>: the encoded image and reference conditions.</td></tr>
<tr><td><code>--save-audio-sidecar</code>, <code>--no-save-audio-sidecar</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;.wav</code>: the generated audio.</td></tr>
<tr><td><code>--save-vae-frames</code>, <code>--no-save-vae-frames</code></td><td><em>off</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_vae_frames/</code>: every decoded SDR frame as PNG, before upscaling or encoding.</td></tr>
<tr><td><code>--save-hdr-heic-frames</code>, <code>--no-save-hdr-heic-frames</code></td><td><em>off</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_heic/</code>: viewable 10-bit PQ HEIC frames of an HDR run.</td></tr>
</tbody>
</table>

## Model files

See [Model files](downloads.md) for what to download and how files are found.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="2" valign="top"><code>--model-generation</code>, <code>--ltx-generation</code></td><td rowspan="2" valign="top"><code>2.3</code></td><td><code>2.3</code></td><td>Find and run the LTX-2.3 checkpoint family.</td></tr>
<tr><td><code>2.5</code></td><td>Find and run the LTX-2.5 component pack.</td></tr>
<tr><td><code>--weights-path</code></td><td><em>Hugging Face cache</em></td><td><code>PATH</code></td><td>The LTX-2.3 checkpoint file, or the LTX-2.5 pack folder.</td></tr>
<tr><td><code>--transformer-path</code></td><td><em>from the baseline</em></td><td><code>PATH</code></td><td>Replace the transformer, with its audio and video connectors.</td></tr>
<tr><td><code>--text-encoder-path</code></td><td><em>discovered</em></td><td><code>PATH</code></td><td>Replace the text encoder. LTX-2.3: a local Gemma 3 folder holding <code>config.json</code>, <code>tokenizer.json</code>, and the model files. LTX-2.5: a prepared text encoder file, with its tokenizer data and projection. Without it, LTX-2.3 finds the Gemma that <code>--gemma-variant</code> names in the Hugging Face cache, and LTX-2.5 uses the pack's.</td></tr>
<tr><td rowspan="2" valign="top"><code>--gemma-variant</code></td><td rowspan="2" valign="top"><code>qat</code></td><td><code>qat</code></td><td>LTX-2.3: the QAT-unquantized Gemma 3 that Lightricks pairs with the release.</td></tr>
<tr><td><code>plain</code></td><td>LTX-2.3: Google's instruction-tuned Gemma 3.</td></tr>
<tr><td><code>--video-vae-path</code></td><td><em>selected by <code>--video-vae</code></em></td><td><code>PATH</code></td><td>Replace the video decoder with this file.</td></tr>
<tr><td><code>--audio-vae-path</code></td><td><em>from the baseline</em></td><td><code>PATH</code></td><td>Replace the audio VAE and vocoder.</td></tr>
<tr><td><code>--spatial-upscaler-path</code></td><td><em>found automatically</em></td><td><code>PATH</code></td><td>The 2x spatial latent upscaler.</td></tr>
<tr><td><code>--temporal-latent-upscaler-path</code></td><td><em>found automatically</em></td><td><code>PATH</code></td><td>The 2x temporal latent upscaler, for the Python component API.</td></tr>
<tr><td><code>--duration-head-path</code></td><td><em>found automatically</em></td><td><code>PATH</code></td><td>The LTX-2.5 duration head for <code>--auto-duration</code>.</td></tr>
</tbody>
</table>

## Memory and speed

See [Performance and memory](performance.md) before changing these.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td rowspan="3" valign="top"><code>--transformer-dtype</code></td><td rowspan="3" valign="top"><code>float16</code></td><td><code>float16</code></td><td>FP16: closer to FP32 than BF16, and faster. A step that overflows stops the run and names BF16.</td></tr>
<tr><td><code>bfloat16</code></td><td>BF16: a wider range, for when FP16 overflows.</td></tr>
<tr><td><code>float32</code></td><td>FP32: twice the memory.</td></tr>
<tr><td><code>--stream-transformer</code>, <code>--no-stream-transformer</code></td><td><em>off</em></td><td><em>flag</em></td><td>Keep a pool of 8 blocks in memory and stream all 48 transformer blocks through it each step. Needed below 64 GB.</td></tr>
<tr><td><code>--transformer-resident-blocks</code></td><td><em>all 48</em>; <code>8</code> with <code>--stream-transformer</code></td><td><code>1..48</code></td><td>Stream the transformer, keeping this many blocks loaded.</td></tr>
<tr><td><code>--transformer-retain</code></td><td><em>none</em></td><td><code>SELECTOR,...</code></td><td>While streaming, keep these tensors loaded across steps: <code>small</code>, <code>audio-attention</code>, <code>video-attention</code>, <code>audio-ff</code>, <code>video-ff</code>, or tensor-name globs. Refused without streaming, and each selector must match.</td></tr>
<tr><td rowspan="3" valign="top"><code>--transformer-quantize</code></td><td rowspan="3" valign="top"><code>off</code></td><td><code>off</code></td><td>Run the transformer's own weights.</td></tr>
<tr><td><code>mxfp8-blocks</code></td><td>Pack block weights to MXFP8: much less memory, slower, and a different result.</td></tr>
<tr><td><code>mxfp8-blocks-pretranspose</code></td><td>MXFP8 with transposed storage; experimental.</td></tr>
<tr><td><code>--transformer-quantize-exclude-layers</code></td><td><em>none</em></td><td><code>N,...</code> (0 to 47)</td><td>With <code>--transformer-quantize</code>, keep these blocks unquantized.</td></tr>
<tr><td><code>--dense-attention-kernel</code>, <code>--no-dense-attention-kernel</code></td><td><em>on</em></td><td><em>flag</em></td><td>KinoMLX's FP16 attention kernel; off runs every call on MLX's attention, which is slower.</td></tr>
<tr><td><code>--stage1-sparse-attention</code>, <code>--sparse-attention-stage-1</code></td><td><code>dense</code></td><td><code>dense</code>, <code>T,T,...</code>, <code>T</code>, <code>A-B=T</code></td><td>Approximate video self-attention on stage-1 steps. See <a href="performance.md#sparse-video-attention">sparse attention</a>.</td></tr>
<tr><td><code>--stage2-sparse-attention</code>, <code>--sparse-attention-stage-2</code></td><td><code>dense</code></td><td><code>dense</code>, <code>auto</code>, <code>T,T,...</code>, <code>T</code>, <code>A-B=T</code></td><td>Approximate video self-attention on stage-2 steps; <code>auto</code> uses the published three-step schedule.</td></tr>
<tr><td><code>--stage1-dense-layers</code>, <code>--dense-layers-stage-1</code></td><td><code>0</code></td><td><code>N,A-B,...</code> or <code>none</code></td><td>Layers that stay exact on sparse stage-1 steps.</td></tr>
<tr><td><code>--stage2-dense-layers</code>, <code>--dense-layers-stage-2</code></td><td><code>0</code></td><td><code>N,A-B,...</code> or <code>none</code></td><td>Layers that stay exact on sparse stage-2 steps.</td></tr>
</tbody>
</table>

## Expert transformer tuning

These choices exist for benchmarks and diagnostics. The defaults are the tuned configuration:
changing them can make a run slower, and the quantization options also change its output.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--fast-mode</code>, <code>--no-fast-mode</code></td><td><em>on</em></td><td><em>flag</em></td><td>Build each forward as one graph over all 48 blocks. Off evaluates every eight blocks.</td></tr>
<tr><td><code>--compile-attention</code>, <code>--no-compile-attention</code></td><td><em>on</em></td><td><em>flag</em></td><td>Compile each attention call with its head split and merge.</td></tr>
<tr><td><code>--compile-block-group-size</code></td><td><em>off</em></td><td><code>1..48</code></td><td>Compile a resident transformer in groups of this many blocks. Sparse steps run uncompiled.</td></tr>
<tr><td><code>--transformer-compile-group-size</code></td><td><em>off</em></td><td><code>N</code></td><td>Compile streamed block groups of this size; it must not exceed the resident block count.</td></tr>
<tr><td><code>--video-ff-layout-specs</code></td><td><code>project_out:pretranspose</code></td><td><code>TARGET:pretranspose,...</code></td><td>Store these video feed-forward weights transposed. Targets: <code>project_in</code>, <code>project_out</code>.</td></tr>
<tr><td><code>--video-ff-layout-layers</code></td><td><em>all</em></td><td><code>N,...</code> (0 to 47)</td><td>Layers the video feed-forward layouts apply to.</td></tr>
<tr><td><code>--video-attn-layout-specs</code></td><td><em>none</em></td><td><code>TARGET:pretranspose,...</code></td><td>Store these video attention weights transposed. Targets: <code>to_q</code>, <code>to_k</code>, <code>to_v</code>, <code>to_out</code>, <code>to_gate_logits</code>.</td></tr>
<tr><td><code>--video-attn-layout-layers</code></td><td><em>all</em></td><td><code>N,...</code> (0 to 47)</td><td>Layers the video attention layouts apply to.</td></tr>
<tr><td><code>--audio-layout-mirror</code>, <code>--no-audio-layout-mirror</code></td><td><em>on</em></td><td><em>flag</em></td><td>Give audio blocks the video layouts. Turn it off to set the audio layouts below, which are refused while it is on.</td></tr>
<tr><td><code>--audio-ff-layout-specs</code></td><td><em>none</em></td><td><code>TARGET:pretranspose,...</code></td><td>Audio feed-forward layouts.</td></tr>
<tr><td><code>--audio-ff-layout-layers</code></td><td><em>all</em></td><td><code>N,...</code> (0 to 47)</td><td>Layers the audio feed-forward layouts apply to.</td></tr>
<tr><td><code>--audio-attn-layout-specs</code></td><td><em>none</em></td><td><code>TARGET:pretranspose,...</code></td><td>Audio attention layouts.</td></tr>
<tr><td><code>--audio-attn-layout-layers</code></td><td><em>all</em></td><td><code>N,...</code> (0 to 47)</td><td>Layers the audio attention layouts apply to.</td></tr>
<tr><td><code>--adaln-pretranspose</code>, <code>--no-adaln-pretranspose</code></td><td><em>off</em></td><td><em>flag</em></td><td>Store the AdaLN modulation weights transposed.</td></tr>
<tr><td><code>--video-ff-quantize-specs</code></td><td><em>none</em></td><td><code>TARGET:MODE,...</code></td><td>Quantize selected video feed-forward weights; not with <code>--transformer-quantize</code>. Modes: <code>affine</code>, <code>mxfp4</code>, <code>mxfp8</code>, <code>nvfp4</code>.</td></tr>
<tr><td><code>--video-ff-quantize-layers</code></td><td><em>all</em></td><td><code>N,...</code> (0 to 47)</td><td>Layers the targeted quantization applies to.</td></tr>
<tr><td><code>--video-ff-quantize-group-size</code></td><td><em>mode default</em></td><td><code>N</code></td><td>Quantization group size for the specs above.</td></tr>
<tr><td><code>--video-ff-quantize-bits</code></td><td><em>mode default</em></td><td><code>N</code></td><td>Bit count for <code>affine</code> quantization.</td></tr>
</tbody>
</table>

## Restart

See [Restart from saved stages](restart.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--restart</code></td><td><em>none</em></td><td><code>RUN.json</code></td><td>Restart from a saved run record.</td></tr>
<tr><td rowspan="2" valign="top"><code>--restart-from</code></td><td rowspan="2" valign="top"><code>decode</code></td><td><code>decode</code></td><td>Decode a saved latent again.</td></tr>
<tr><td><code>stage-2</code></td><td>Run stage 2 again from the saved stage-1 latent and prompt.</td></tr>
<tr><td><code>--latent-stage</code></td><td><em>latest saved</em></td><td><code>stage-1</code>, <code>stage-2</code></td><td>Which latent a decode restart reads: stage 1's half-size latent or the final one.</td></tr>
<tr><td><code>--restart-latents</code></td><td><em>recorded in the run</em></td><td><code>PATH</code></td><td>Use this latent file instead of the one the run record names.</td></tr>
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
| `--stage1-sampler`, `--stage2-sampler` | `generate.stage_1_sampler`, `generate.stage_2_sampler` | `KINO_LTX2_STAGE_1_SAMPLER`, `KINO_LTX2_STAGE_2_SAMPLER` |
| `--sigmas-stage-1`, `--sigmas-stage-2` | `generate.stage_1_sigmas`, `generate.stage_2_sigmas` | `KINO_LTX2_STAGE_1_SIGMAS`, `KINO_LTX2_STAGE_2_SIGMAS` |
| `--stage1-sparse-attention`, `--stage2-sparse-attention` | `generate.stage_1_sparse_attention`, `generate.stage_2_sparse_attention` | `KINO_LTX2_STAGE_1_SPARSE_ATTENTION`, `KINO_LTX2_STAGE_2_SPARSE_ATTENTION` |
| `--stage1-dense-layers`, `--stage2-dense-layers` | `generate.stage_1_dense_layers`, `generate.stage_2_dense_layers` | `KINO_LTX2_STAGE_1_DENSE_LAYERS`, `KINO_LTX2_STAGE_2_DENSE_LAYERS` |
| `--reference-prompt-padding`, `--compact-prompt` | `generate.pad_prompt_to_max`, `true` or `false` | `KINO_LTX2_PAD_PROMPT_TO_MAX` |
| `--hdr-reference`, `--hdr-reference-strength` | `generate.hdr_reference.path`, `generate.hdr_reference.strength` | `KINO_LTX2_HDR_REFERENCE`, `KINO_LTX2_HDR_REFERENCE_STRENGTH` |
| `--vae-tiling` | `generate.vae_tiling.mode` | `KINO_LTX2_VAE_TILING` |
| `--vae-temporal-tile-frames` and the other tile sizes | `generate.vae_tiling.temporal_tile_frames` and its siblings | `KINO_LTX2_VAE_TEMPORAL_TILE_FRAMES` and its siblings |
| `--adapter` and the flags that modify it | `generate.adapters` records with `path`, `strength`, `exclude`, `stage1_strength`, `stage2_strength` | `KINO_LTX2_ADAPTERS` |
| `--image` and the flags that modify it | `generate.images` records with `path`, `frame_index`, `strength`, `stage1_strength`, `stage2_strength` | `KINO_LTX2_IMAGES` |
