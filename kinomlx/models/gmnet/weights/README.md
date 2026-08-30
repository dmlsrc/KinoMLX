# GMNet weights

Not bundled - download and convert; the `.safetensors` files are
gitignored. Both published checkpoints are MIT-licensed files committed
inside the upstream repository. SHA-256 is of the source `.pth`:

| variant | download | SHA-256 |
| --- | --- | --- |
| realworld (photographed pairs; 203-nit SDR white, 5x peak, half-resolution local branch) | <https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth> | `83bf27bcdbf6eacfdef37f0e24ed6d79152b7386620c012ae509a59a895c875f` |
| synthetic (HDR video frames; 100-nit SDR white, 8x peak, full-resolution local branch) | <https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth> | `887c940d492424cd44f029c6b09dd3bbe1bbec07126f15d41192828ff95e6880` |

Collect the downloads into the repo-root [`weights-src/`](../../../../weights-src/README.md)
collection (`weights-src/gmnet/`, upstream filenames, verify the SHA-256),
then convert from the repo root - published checkpoints are recognized
by their SHA-256. In an editable Git checkout, they land in this directory
under their canonical names. A non-checkout install uses
`$KINO_CACHE_DIR/weights/gmnet/` instead so conversion never assumes the
installed package is writable. An explicit converter `--output` or runtime
`KINO_GMNET_WEIGHTS` override wins in either layout:

```bash
mkdir -p weights-src/gmnet
curl -L -o weights-src/gmnet/G_realworld.pth https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth
kinomlx weights convert gmnet G_realworld.pth
```

That command refuses to replace an existing canonical file. Add `--force` only
when deliberately reconverting the same source after verifying its SHA-256.

The converter statically scans the embedded pickle, rebuilds tensors
through a restricted torch-free unpickler, requires exactly the
126-tensor generator state dict, stamps provenance metadata (variant,
source file, source SHA-256, license) into the safetensors header, and
verifies the output by loading it back through the model before
reporting success.

A retrained checkpoint that is not one of the published two still converts,
but its numeric contract is your claim: pass `--declare-variant` and an explicit
`--output` path. The
machine-readable variant facts (network scale, peak, reference white,
provenance) live in [`catalog.py`](../catalog.py).

License and paper attribution: [Attribution.md](Attribution.md).
