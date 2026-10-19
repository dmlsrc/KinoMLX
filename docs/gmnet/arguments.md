# GMNet arguments

Every `kinomlx --model gmnet` argument. Each one also has a TOML field and an environment
variable: `kinomlx config init --model gmnet` writes them all with their defaults, and
[Configuration](../CONFIG.md) explains the naming.

The memory, logging, and location arguments are in [shared
arguments](../CONFIG.md#shared-arguments), and the names that differ from their flags, such as
`--output`, in [Configuration](../CONFIG.md#names-that-differ-from-the-flag). The converter's
arguments are on the [weights page](weights.md#gmnet-converter-arguments).

## Input and model

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--image</code></td><td><em>required</em></td><td><code>PATH</code></td><td>The SDR still to expand: PNG, JPEG, HEIC, or TIFF.</td></tr>
<tr><td rowspan="2" valign="top"><code>--variant</code></td><td rowspan="2" valign="top"><code>realworld</code></td><td><code>realworld</code></td><td>Trained on photographed pairs: 203-nit SDR white, up to 5x peak.</td></tr>
<tr><td><code>synthetic</code></td><td>Trained on content derived from HDR video: 100-nit SDR white, up to 8x peak.</td></tr>
<tr><td><code>--weights-path</code></td><td><em>the variant's converted weights</em></td><td><code>PATH</code></td><td>Use this converted safetensors file. It must have been converted for the selected <code>--variant</code>; KinoMLX cannot tell.</td></tr>
<tr><td><code>--dry-run</code></td><td><em>off</em></td><td><em>flag</em></td><td>Report the weights file and the variant's values without loading weights, then exit. It checks only that the file exists, and cannot be combined with <code>--image</code>.</td></tr>
</tbody>
</table>

## Output files

See [Output files](output.md).

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--output</code></td><td><em>timestamped name</em></td><td><code>PATH.exr</code> or <code>PATH.heic</code></td><td>Write this exact primary file, without a timestamp. Selected sidecars share its stem.</td></tr>
<tr><td><code>--output-dir</code></td><td><code>outputs</code></td><td><code>PATH</code></td><td>Directory for timestamped names; see <a href="output.md#names">output files</a> for the fallbacks.</td></tr>
<tr><td><code>--output-prefix</code></td><td><em>the input's name</em></td><td><code>TEXT</code></td><td>Name before the timestamp. Characters other than letters, digits, <code>-</code>, <code>_</code>, and <code>.</code> become <code>_</code>.</td></tr>
<tr><td><code>--exr</code>, <code>--no-exr</code></td><td><em>on</em>; <em>from the suffix</em> with <code>--output</code></td><td><em>flag</em></td><td>Write a half-float, scene-linear Rec.709 EXR master. <code>--no-exr</code> is refused with an exact <code>.exr</code> path.</td></tr>
<tr><td><code>--heic</code>, <code>--no-heic</code></td><td><em>on</em>; <em>from the suffix</em> with <code>--output</code></td><td><em>flag</em></td><td>Write a viewable 10-bit BT.2100 PQ HEIC. <code>--no-heic</code> is refused with an exact <code>.heic</code> path.</td></tr>
<tr><td><code>--force</code>, <code>--no-force</code></td><td><em>off</em></td><td><em>flag</em></td><td>Replace existing files at an exact <code>--output</code>; refused without one.</td></tr>
</tbody>
</table>

A run must write something: turning off the EXR, the HEIC, and the gain map together is refused.
A gain map alone is allowed.

## Sidecars

Each sidecar is off unless you select it or `--save-all-sidecars`.

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--save-all-sidecars</code>, <code>--no-save-all-sidecars</code></td><td><em>off</em></td><td><em>flag</em></td><td>Save everything below.</td></tr>
<tr><td><code>--save-gain-map</code>, <code>--no-save-gain-map</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;.gain_map.safetensors</code>: the normalized gain map and its reconstruction rule.</td></tr>
<tr><td><code>--save-run-log</code>, <code>--no-save-run-log</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_run.json</code>: settings, timings, and outputs.</td></tr>
<tr><td><code>--save-console-log</code>, <code>--no-save-console-log</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_console.log</code>: the human-readable log.</td></tr>
<tr><td><code>--save-effective-config</code>, <code>--no-save-effective-config</code></td><td><em>off; follows save-all</em></td><td><em>flag</em></td><td><code>&lt;stem&gt;_config.toml</code>: the resolved invocation.</td></tr>
</tbody>
</table>
