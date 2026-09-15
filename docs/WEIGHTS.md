# Model files and loading

KinoMLX reads model files already on your Mac. It does not download weights during
generation or create a second, converted model-weight cache on disk. Download the
components for the model you intend to use before starting a run.

## Get the right files

| Model | Download and selection instructions |
| --- | --- |
| LTX-2.3 / LTX-2.5 | [LTX downloads](LTX2.md#ltx-model-downloads): transformer, text encoder, VAE, and upscaler requirements |
| Krea 2 Raw / Turbo | [Krea model access](KREA2.md#model-access): complete model folder and optional decoder/adapter files |
| FLUX.2 Klein base / distilled / KV | [Klein checkpoints](FLUX2.md#local-checkpoint-selection): choose the variant and keep its component folders |
| GMNet | [GMNet weights](GMNET.md#model-weights): download and convert the original checkpoint once |

For Hugging Face downloads, keep the original snapshot layout. A sharded model
needs every shard named by its index, along with the required configuration and
tokenizer files. Authentication does not grant access to a gated model by itself;
accept or request access on its model page before downloading.

## Choose where models load from

KinoMLX discovers downloaded snapshots in the Hugging Face cache selected by
`HF_HOME`. You can instead pass explicit local sources:

- LTX uses `--weights-path` plus individual component overrides when needed.
- Krea and Klein use `--source-path` for a model folder, or individual component paths.
- GMNet discovers converted weights or accepts the paths described in its guide.

Use `kinomlx --model MODEL --help` for that model's path options. Model guides
include examples for cache-managed downloads and manually selected folders.
Explicit overrides must be compatible with the selected model; a missing or
incompatible file is not silently replaced by a different downloaded model.

## What happens during loading

Model components are loaded when needed and released after their part of the run:
text encoding, denoising, then decoding. This reduces peak memory because the text
encoder, transformer, and decoder do not all need to stay resident together.
A large monolithic checkpoint does not require every tensor in the file to be
resident at once; each component selects the weights it uses.

KinoMLX prepares weight dtype and layout changes lazily. Some preparation is
therefore paid on first use rather than when the file is initially inspected.
The first denoising step can include transformer materialization and adapter
fusion. Compare complete runs or phase timings rather than treating the initial
loading message as the full loading cost.

With [LTX block streaming](LTX2.md#block-streaming-on-32-gb-systems), only a subset
of transformer blocks stays resident. Reloading repeats reads and preparation,
including adapter fusion where applicable. Reads can overlap computation, but a
checkpoint that no longer fits in the filesystem cache can make storage bandwidth
an important part of step time.

## Precision and validation

LTX, Krea, and Klein default to BF16 transformer execution. FP16 is an explicit
speed option with the same bytes per weight and a narrower numerical range.
Selecting a compute dtype does not rewrite the source checkpoint. Supported FP8
LTX sources are decoded into the requested execution dtype; choosing FP16 from a
BF16 source cannot recover precision already absent from that source.

KinoMLX checks required tensor names, shapes, and storage dtypes before using a
component. Generation also checks denoising latents for nonfinite values.
Adapter-modified active floating-point weights receive a first-use check by
default; packed LTX targets rely on latent checks. See
[adapter checks](ADAPTERS.md#first-use-weight-checks) for coverage and opt-out behavior.

If loading fails, read the named component and file in the error. Check that the
download is complete, the selected variant is correct, and an explicit path has
not mixed incompatible component versions. If FP16 produces nonfinite values,
try BF16 and inspect adapter strengths and compatibility. Do not disable checks
to make an incompatible export appear supported.

Source identities and metadata are recorded for reproducibility, but normal
loading does not hash every weight payload. Successful loading is not a full-file
integrity audit; verify download hashes separately when investigating file corruption.

## Model files, caches, and saved results

| Storage | Purpose | What changing the MLX cache limit does |
| --- | --- | --- |
| Hugging Face cache or your model folder | Original downloaded model files | Nothing; these are disk files |
| MLX allocator cache | Unused in-memory allocations kept for reuse | Limits retained buffers; `0` disables caching |
| Application cache | Manual GMNet conversions and native profiling builds | Nothing; managed separately |
| Saved conditioning and latent sidecars | Optional outputs used to restart part of a run | Nothing; keep the files needed by your restart |
| Klein reference KV | Reference keys/values reused within one generation | Does not remove live reference tensors |

`--app-cache-dir`, `[settings].app_cache_dir`, or `KINO_APP_CACHE_DIR` selects the
application cache. Its default is `~/Library/Caches/KinoMLX`; `XDG_CACHE_HOME`
selects its `KinoMLX` subdirectory. Keep original model sources available for
restart: saved latents or conditioning do not bundle all remaining model weights.
For MLX allocator and wired-memory controls, see [Performance](PERFORMANCE.md).

## Public API memory policy

Prepared resources describe model files and execution choices; they do not keep
live model weights resident. Reuse resources or a runner when making sequential
calls, and close generation results before starting another call.

Runners and public recipes apply the selected infrastructure `Settings` memory
policy before model work. If you call component loaders directly, apply it yourself
with `kinomlx.apply_memory_policy(resources.memory_policy)` before loading or
allocating inputs. These controls affect the entire process; see
[API memory policy](PERFORMANCE.md#public-api-memory-policy) for lifetime and concurrency rules.
