# Klein Python API

Use the Python API to generate inside your own application, compare against another
implementation with explicit tensors, or run the text encoder and VAE on their own. Requests and
settings use the config files' field names, not the flag spellings;
`kinomlx config inspect --model flux2` shows each flag's field and where Python takes it.

- [Generate](#generate)
- [Restart](#restart)
- [Lower-level generation](#lower-level-generation)
- [Text encoder and VAE](#text-encoder-and-vae)
- [Reference preprocessing](#reference-preprocessing)
- [Memory policy](#memory-policy)

## Generate

```python
from kinomlx.models.flux2 import Flux2Request, Flux2Runner, Flux2Settings

runner = Flux2Runner(Flux2Settings(variant="distilled"))
output = runner.generate(
    Flux2Request(prompt="A ceramic vase beside a window.", width=512, height=512, seed=42)
)
image = output.images[0]
```

`Flux2Runner(resources=prepared, reporter=reporter)` uses resources prepared earlier, without
inspecting the files again; give either settings or resources, not both. The runner keeps
resource descriptions and reporting only, never model weights, results, or reference KV.
`generate(request)` runs `generate_flux2`, and `run(recipe, request)` runs your own recipe.

`output.images` holds `Flux2ImageResult` objects in prompt and variation order: HWC RGB pixels
in `[0, 1]`, the packed final latent, dimensions, reference geometry, adapter receipts, the
complete conditioning, the initial noise, and the schedule. For editing and recipe controls, set
`source_images`, `steps`, `guidance`, and `cfg_mode` on the request.

[`examples/flux2_generate.py`](../../examples/flux2_generate.py) reserves an output bundle before
building the runner and writes results through `Flux2OutputSink`.

## Restart

```python
from pathlib import Path

from kinomlx.models.flux2 import Flux2Request, Flux2Restart

output = runner.restart(
    Flux2Request(),
    Flux2Restart(stage="denoise", text_conditioning=Path("previous_text.safetensors")),
)
decoded = runner.restart(
    Flux2Request(),
    Flux2Restart(latents=Path("previous_final_latents.safetensors")),
)
```

A restart uses the runner's prepared resources. Saved conditioning supplies its own prompt, and a
conflicting explicit prompt is refused; a decode uses only the VAE and the latent. The CLI also
resolves run records and checks recorded identities before calling the same recipes.

`Flux2Error` is the error type for operational failures; missing files and cancellation keep
their usual exception types.

## Lower-level generation

`generate_image(resources, prompt, ...)` generates one image from explicit tensors and sampling
controls, for comparisons with another implementation. `capture_stages=True` keeps the
conditioning, initial noise, and schedule on the result, `conditioning=` supplies saved
conditioning, and `decode_latents()` decodes a final latent and grid with the VAE files. Each
model's lifetime ends before the next loads.

| Argument | Meaning |
| --- | --- |
| `images=(Path("reference.jpg"),)` | References, in order; one reference matches its shape by default |
| `width=`, `height=`, `match_source_aspect=`, `reference_fit=` | Output size and reference fitting, with the CLI's rules |
| `recipe=SamplingRecipe(...)` | Override steps, guidance, and `cfg_mode`; its variant must match the resources |
| `negative_prompt=` | Negative text for base guidance; empty by default |
| `negative_embeddings=` | An explicit `[1, 512, 12288]` negative tensor instead of text |
| `noise=` | Packed `[1, H/16 * W/16, 128]` initial noise, shared with another runtime |

`SamplingRecipe(Flux2Variant.KV, recompute_reference_kv=True)` matches
`--recompute-reference-kv`. Without it, the model's weights, adapters included, must not change
during a KV call, because the cached reference values were computed with the first step's
weights.

Negative conditioning needs active base guidance, and a nonempty negative prompt cannot be
combined with explicit embeddings. Without `noise=`, `seed` and `noise_backend` produce the noise;
equal seed numbers in different random generators do not give equal tensors.

## Text encoder and VAE

The component loaders run the text encoder or the VAE without the transformer. Apply the memory
policy first, and release each model when done:

```python
import mlx.core as mx

from kinomlx import apply_memory_policy
from kinomlx.models.flux2.resources import prepare_resources
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.flux2.source_loading import (
    component_weight_files,
    load_text_encoder,
    load_tokenizer,
)
from kinomlx.models.flux2.text import prepare_text
from kinomlx.settings import Settings

resources = prepare_resources(Flux2Settings(variant="kv"), infrastructure=Settings.from_env())
apply_memory_policy(resources.memory_policy)
tokenizer = load_tokenizer(resources.component("tokenizer"))
batch = prepare_text(tokenizer, ("A glass sphere on a wooden table.", ""))
encoder = load_text_encoder(component_weight_files(resources.component("text_encoder")))
conditioning = encoder(batch.input_ids, batch.attention_mask)
mx.eval(conditioning)
del encoder
```

Text uses the user-message template with thinking disabled, right-padded or truncated to 512 tokens,
and raw hidden-state taps 9, 18, and 27. `conditioning` is `[batch, 512, 12288]`; keep all 512
states, padding included. `batch.text_ids` gives the `[batch, 512, 4]` THWL coordinates. A negative
or empty prompt goes through the same encoder.

`load_vae(files, mode="both")` returns a BF16 `Flux2VAE`; `mode="encode"` or `mode="decode"` loads
only that half. `dtype=mx.float32` runs the BF16 weights in FP32 for diagnostics; it does not
recover FP32 weights.

| Method | Takes | Returns |
| --- | --- | --- |
| `encode()` | BCHW RGB in `[-1, 1]`, sides divisible by 16 | The raw 32-channel posterior mean |
| `encode_normalized()` | The same | 2x2 positions packed into 128 channels, normalized |
| `decode_normalized()` | A packed, normalized latent | Unclipped BCHW pixels; clip and map to `[0, 1]` yourself |

There is no posterior sampling or tiling here.

## Reference preprocessing

`image_processing.output_size()` resolves the output size, including the one-reference aspect
default. `reference_geometry()` records each reference's crop, fitted size, and fractional token
offsets. `load_reference_pixels()` applies EXIF orientation, bicubic resampling, and
normalization. `spatial_ids()` places the output at time 0 and reference `i` at time `10 * (i + 1)`,
keeping half-token offsets.

## Memory policy

Runners and recipes apply the memory policy from `Settings` (`mlx_cache_limit_gb` and
`mlx_wired_limit_gb`) before model work, on every call. When you load components directly, call
`kinomlx.apply_memory_policy(resources.memory_policy)` first, as above. These limits apply to the
whole process; see [API memory policy](../PERFORMANCE.md#public-api-memory-policy).
