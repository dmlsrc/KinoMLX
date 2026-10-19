# GMNet output files

A GMNet run writes an EXR master and a viewable HEIC by default, and optionally the gain map and
execution sidecars. This page covers which files a run writes, their names, and how existing files
are protected.

- [What a run writes](#what-a-run-writes)
- [Names](#names)
- [Protecting existing files](#protecting-existing-files)
- [A leftover reservation](#a-leftover-reservation)

## What a run writes

| File | Contents | Selected by |
| --- | --- | --- |
| `<stem>.exr` | Half-float, scene-linear Rec.709 EXR: the HDR master | On by default; `--no-exr` turns it off |
| `<stem>.heic` | 10-bit BT.2100 PQ HEIC, viewable in Photos and Preview | On by default; `--no-heic` turns it off |
| `<stem>.gain_map.safetensors` | The normalized gain map and its reconstruction rule | `--save-gain-map` |
| `<stem>_run.json`, `<stem>_console.log`, `<stem>_config.toml` | The run record, log, and effective config | `--save-run-log`, `--save-console-log`, `--save-effective-config` |

`--save-all-sidecars` selects the gain map and the three execution files. A run must write at
least one of the EXR, the HEIC, and the gain map.

## Names

Without `--output`, the stem is `<input-stem>_YYYYMMDD_HHMMSS`. `--output-prefix` replaces the
input's name and keeps the timestamp. If any file of the bundle would collide, the whole stem gets
`_2`, `_3`, and so on.

Without `--output-dir` or `[output].directory`, files go to `KINO_GMNET_OUTPUT_DIR`, then
`KINO_OUTPUT_DIR`, then `outputs/`.

`--output photo.exr` or `--output photo.heic` sets the primary file exactly, with no timestamp;
selected sidecars share its stem. Add `--heic` or `--exr` to write the sibling format too.

## Protecting existing files

Every name is reserved, and every conflict found, before the model runs, so a collision never
costs an expansion. A generated name never replaces a file. With an exact `--output`, existing
targets are refused unless `--force` allows replacing the selected bundle.

The EXR, HEIC, and gain map are prepared first and published together. An ordinary encoding or
publication failure leaves the previous versions in place, even with `--force`.

The execution sidecars are written as the run goes instead. The effective config and console log
start before the model runs, and a failed run still leaves its run record, marked `failed`.

## A leftover reservation

A run that was killed without cleaning up can leave its output name reserved. Generated names
skip it. For an exact `--output`, the error names a hidden `.<name>.kinomlx-reservation` marker:
check that no run is still writing there, delete that marker, and retry.
