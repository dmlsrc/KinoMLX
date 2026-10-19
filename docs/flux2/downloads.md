# FLUX.2 Klein model files

KinoMLX reads Klein from files already on your Mac; it never downloads during inspection,
preparation, or generation. This page lists what to download and how KinoMLX finds it.

- [Get access](#get-access)
- [Download a variant](#download-a-variant)
- [How files are found](#how-files-are-found)
- [Replace components](#replace-components)

## Get access

Each variant is a separate repository:

| Variant | Repository | Default recipe |
| --- | --- | --- |
| `distilled` | [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 4 steps |
| `kv` | [FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) | 4 steps, reference KV reused within a generation |
| `base` | [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | 50 steps, guidance 4 |

Read the model card and the [license](README.md#licenses-and-credits) before downloading. Accept
the repository's access conditions on Hugging Face, then sign in with `hf auth login` if needed.

## Download a variant

Download only the variants you intend to use:

```bash
hf download black-forest-labs/FLUX.2-klein-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-base-9B --exclude "flux-2-klein-*.safetensors"
hf download black-forest-labs/FLUX.2-klein-9b-kv --exclude "flux-2-klein-*.safetensors"
```

The exclusion skips only the single-file transformer, a duplicate of the component folders
KinoMLX reads. The README, license, and sample images that remain are small.

These commands fill the Hugging Face cache selected by `HF_HOME`. Use
`hf download --local-dir` only when you intend to pass that folder as `--source-path`.

## How files are found

KinoMLX reads each repository's local `refs/main` snapshot in the Hugging Face cache, from
`HF_HOME` or `--hf-home`. `distilled` is the default variant.

```bash
kinomlx --model flux2 --variant kv --dry-run
kinomlx --model flux2 --variant base --source-path /path/to/klein-base --dry-run
```

`--dry-run` checks every file a run would read and exits without loading weights; add `--json`
for a machine-readable inventory.

A source folder holds `model_index.json`, `scheduler/scheduler_config.json`, `tokenizer/`,
`text_encoder/`, `transformer/`, and `vae/`. Each tensor component needs its `config.json` and
its safetensors file or shard index. Use the bundled BF16 9B components: other shapes, storage
dtypes, or configurations are refused rather than converted.

The variant selects the sampling recipe. Matching weight layouts cannot tell base, distilled, and
KV transformers apart, so choose the variant of replacement weights yourself; an explicitly
contradicting `is_distilled` declaration is refused.

## Replace components

`--tokenizer-path`, `--text-encoder-path`, `--transformer-path`, and `--vae-path` replace one
component each, and are checked before default discovery. With all four given, no baseline
snapshot is needed, and the selected variant's scheduler applies. A `--source-path` folder still
has its own pipeline and scheduler checked.

Components load from their original files and prepare their layouts in memory, so no converted
copy is written to disk.
