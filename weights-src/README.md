# Upstream source checkpoints

The collected, hash-verified upstream files every converted
`.safetensors` in this repo was produced from. One subfolder per
weights owner, files under their upstream names. Everything here except
this README is untracked; this file exists so git keeps the folder and
so the layout is documented, and so there is one place to reconvert
from when a converter changes.

`kinomlx weights convert` and its model-specific extensions look here: an
input that does not exist as given is resolved as `weights-src/<path>` and then
as a unique `weights-src/**/<name>` match (relative to the working directory),
so documented conversion commands work with bare filenames from the repo root:

```bash
kinomlx weights convert gmnet G_realworld.pth
```

## Contents

Per-checkpoint provenance (source URL, SHA-256, license, numeric
contract) is code in each owner's model package; conversion commands
live in each owner's `weights/README.md`. Every file placed here was
SHA-256-verified against the recorded upstream hash at collection time.

| folder | files | provenance and conversion |
| --- | --- | --- |
| gmnet/ | G_realworld.pth, G_synthetic.pth | [`kinomlx/models/gmnet/catalog.py`](../kinomlx/models/gmnet/catalog.py), [`kinomlx/models/gmnet/weights/README.md`](../kinomlx/models/gmnet/weights/README.md) |
