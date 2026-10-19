# GMNet weights

KinoMLX does not bundle GMNet's weights. The authors publish two MIT-licensed PyTorch
checkpoints; convert the one you want once into a safetensors file, which every run then loads.
Normal runs never open a PyTorch checkpoint.

- [Download](#download)
- [Convert](#convert)
- [Other checkpoints](#other-checkpoints)
- [Where converted weights go](#where-converted-weights-go)
- [GMNet converter arguments](#gmnet-converter-arguments)

## Download

| Variant | Checkpoint | SHA-256 |
| --- | --- | --- |
| `realworld` | [G_realworld.pth](https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth) | `83bf27bcdbf6eacfdef37f0e24ed6d79152b7386620c012ae509a59a895c875f` |
| `synthetic` | [G_synthetic.pth](https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth) | `887c940d492424cd44f029c6b09dd3bbe1bbec07126f15d41192828ff95e6880` |

Keep the published names, in a `weights-src/gmnet/` folder of the directory you run the converter
from; a source checkout has `weights-src/` at its root:

```bash
mkdir -p weights-src/gmnet
curl -L -o weights-src/gmnet/G_realworld.pth \
  https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth
curl -L -o weights-src/gmnet/G_synthetic.pth \
  https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth
```

## Convert

From that directory:

```bash
kinomlx weights convert gmnet G_realworld.pth
kinomlx weights convert gmnet G_synthetic.pth
```

A bare filename is looked up under `weights-src/`, and the two published files are recognized by
their SHA-256, which sets their variant.

The converter checks that every generator weight is present, writes a temporary file, and loads it
back through the model before publishing it. It reads the checkpoint with a restricted reader that
never runs code from it, and it needs no PyTorch.

An existing converted file is kept unless you pass `--force`. Even then, the new file is verified
before it replaces the old one.

## Other checkpoints

A checkpoint that is not one of the published two, such as a retrained one, cannot be recognized,
so tell the converter about it: `--declare-variant` records which variant's numeric contract it
follows, and `--output PATH.safetensors` gives it a file of its own. With neither, conversion is
refused.

`--declare-variant` only labels the file. Without `--output`, the file goes to that variant's
default location, the one normal runs load, and replaces an existing conversion there only with
`--force`. A declaration that contradicts a published file's own variant is refused. At run time,
`--variant` and `--weights-path` decide what is used.

## Where converted weights go

| Install | Default location |
| --- | --- |
| A Git source checkout, installed editable | `kinomlx/models/gmnet/weights/`, next to the model code |
| Any other install | The application cache: `~/Library/Caches/KinoMLX/weights/gmnet/` |

The application cache keeps an installed package from needing to be writable. `--app-cache-dir`,
`KINO_APP_CACHE_DIR`, or `XDG_CACHE_HOME/KinoMLX` move it.

To use a converted file elsewhere, pass the converter's `--output`, then point runs at it with
`--weights-path` or `KINO_GMNET_WEIGHTS_PATH`.

## GMNet converter arguments

`kinomlx weights convert gmnet CHECKPOINT` takes:

<table>
<thead><tr><th>Argument</th><th>Default</th><th>Value</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>-o</code>, <code>--output</code></td><td><em>the variant's default file</em></td><td><code>PATH.safetensors</code></td><td>Write the converted weights to this exact file.</td></tr>
<tr><td><code>--declare-variant</code></td><td><em>from the published file</em></td><td><code>realworld</code>, <code>synthetic</code></td><td>Label the numeric contract of a checkpoint that is not one of the published two. Refused when it contradicts a published file.</td></tr>
<tr><td><code>--force</code>, <code>--no-force</code></td><td><em>off</em></td><td><em>flag</em></td><td>Replace an existing converted file, after verifying the new one.</td></tr>
<tr><td><code>--allow-suspicious-globals</code>, <code>--no-allow-suspicious-globals</code></td><td><em>off</em></td><td><em>flag</em></td><td>Convert a checkpoint you trust that the static scan refused; the restricted reader still refuses anything outside its allowlist.</td></tr>
</tbody>
</table>

The converter also takes the shared logging, memory, and location options; see
[shared arguments](../CONFIG.md#shared-arguments). For other plain PyTorch state dicts,
`kinomlx weights convert CHECKPOINT` is the generic converter; see
[converting a checkpoint](../WEIGHTS.md#convert-a-checkpoint). Use it only when no model has its
own converter.
