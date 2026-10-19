# Krea 2 Python API

Use the Python API to generate inside your own application, encode text separately, or decode
latents yourself. Requests and settings use the config files' field names, not the flag spellings;
`kinomlx config inspect --model krea2` shows each flag's field and where Python takes it.

- [Run the example](#run-the-example)
- [Generate](#generate)
- [Restart](#restart)
- [Inspect a request before running it](#inspect-a-request-before-running-it)
- [Encode text on its own](#encode-text-on-its-own)
- [Decode and write images yourself](#decode-and-write-images-yourself)
- [Memory policy](#memory-policy)

## Run the example

[`examples/krea2_generate.py`](../../examples/krea2_generate.py) prepares resources, generates,
and writes PNG or TIFF images through `Krea2OutputSink`. Run it from a checkout with KinoMLX
installed:

```bash
python examples/krea2_generate.py "A quiet coastal sunrise" outputs/turbo.png
python examples/krea2_generate.py "A quiet coastal sunrise" outputs/raw.png --variant raw
```

It accepts `--source-path`, `--width`, `--height`, and `--seed`, refuses existing outputs, and
numbers the outputs of a batch. Its `generate_images()` function is reusable in a host.

## Generate

```python
from kinomlx.models.krea2 import Krea2Request, Krea2Runner, Krea2Settings

runner = Krea2Runner(Krea2Settings(variant="turbo"))
request = Krea2Request("A quiet coastal sunrise, cinematic light", width=1024, height=1024, seed=42)

with runner.generate(request) as result:
    print(result.images[0].pixels.shape)
    print(result.metadata["recipe"])
```

Results are closeable; use them inside `with`. Editing uses the same call with
`source_images=(Path("source.png"),)`, and `generate_krea2_edit` is the explicit edit recipe.
Adapters are typed records: `adapters=(Krea2AdapterConfig(...),)`.

`generate_krea2(request, resources, ...)` is the stateless recipe the runner uses. Both accept an
injected `Krea2ComponentsProvider` and `Krea2TextConditionerPort` for hosting and testing.

## Restart

```python
from pathlib import Path

from kinomlx.models.krea2 import Krea2Request, Krea2Restart

with runner.restart(
    request, Krea2Restart.decode(Path("source_final_latents.safetensors"))
) as result:
    print(result.images[0].pixels.shape)

saved_text = Path("source_text.safetensors")
replay = Krea2Request(text_conditioning=saved_text, seed=99)
with runner.restart(replay, Krea2Restart.denoise(saved_text)) as result:
    print(result.metadata["execution_mode"])
```

## Inspect a request before running it

Creating a `Krea2Request` checks each field without opening files. Generation checks the whole
request and reads saved conditioning before using it. To see the prompts a saved file supplies
ahead of time:

```python
from kinomlx.models.krea2.request_sources import resolve_request_sources

print(resolve_request_sources(replay).prompts)
```

Saved rows appear in the resolved request's `prompts` and `negative_prompts`, while the original
request's `prompt` stays `None`. A copied request resolves its own rows.

## Encode text on its own

`Krea2TextConditioner` produces text embeddings without generating an image. It takes one prompt
or a batch, and optionally a same-sized batch of negative prompts through `negative_prompts=`; it
never adds negatives by itself. Both branches share one load of the text encoder, which is
released before the call returns:

```python
from kinomlx import Settings, apply_memory_policy
from kinomlx.models.krea2 import Krea2Settings, Krea2TextConditioner, prepare_resources

resources = prepare_resources(Krea2Settings(variant="turbo"), infrastructure=Settings.from_env())
apply_memory_policy(resources.memory_policy)
conditioning = Krea2TextConditioner()("A quiet coastal sunrise, cinematic light", resources)

print(conditioning.positive.hidden_states.shape)  # (1, 512, 12, 2560)
print(conditioning.provenance.to_metadata())
```

Each product holds twelve tapped Qwen hidden states shaped `[batch, sequence, tap, width]`, the
attention mask, three-axis position IDs, the prompts, and a token identity. Keep its metadata with
saved conditioning so a replay can check compatibility; it holds no model weights.

To reuse embeddings elsewhere, keep the masks and positions, not just the prompt text. The
encoder sees `[34-token instruction | prompt | padding | 5-token suffix]`, 546 positions by
default, and the instruction is removed after encoding, leaving 512. Text uses hidden states 2, 5,
8, and so on up to 35. Another implementation matches only with the same template, tokenizer, and
layer selection; matching the prompt string alone is not enough.

Tokenizer tables come from the model's own JSON files, so Transformers and Tokenizers are not
needed. Editing a chat-template file does not change Krea's prompt layout; use `system_prompt`.

## Decode and write images yourself

`Krea2Components` hands out the denoiser and decoder as `ComponentLease` objects. Close one lease
before opening the next, and call the model through `lease.value(...)`, so only one large model
is loaded at a time. The provider keeps no model itself.

The decoder takes a normalized 16-channel latent in BCHW order and returns `Krea2DecodedRGB`:
unclamped floating RGB in BHWC order. Pass it straight to the output sink, which maps `[-1, 1]` to
display RGB, clamps, converts to bytes, and encodes; converting it yourself first would apply the
mapping twice. PNG, TIFF, and TIF are accepted, and lossy formats are refused.

```python
from pathlib import Path

from kinomlx import Settings, apply_memory_policy
from kinomlx.models.krea2 import (
    Krea2Components,
    Krea2Settings,
    Krea2StillOutputPlan,
    Krea2StillOutputSink,
    Krea2VAETilingConfig,
    prepare_resources,
)

resources = prepare_resources(
    Krea2Settings(variant="turbo", vae_dtype="bfloat16"),
    infrastructure=Settings.from_env(),
)
apply_memory_policy(resources.memory_policy)
plan = Krea2StillOutputPlan(Path("image.png"))

with plan.reserve() as reservation:
    decoder = Krea2Components().image_decoder(resources)
    try:
        decoded = decoder.value(final_latent, tiling=Krea2VAETilingConfig())
    finally:
        decoder.close()
    Krea2StillOutputSink(plan).write(decoded, reservation=reservation)
```

Omit `tiling=` to decode the whole image. `Krea2VAETilingConfig()` uses 256-pixel tiles at a
192-pixel stride, and only when a side exceeds the tile size.

## Memory policy

Runners and recipes apply the memory policy from `Settings` (`mlx_cache_limit_gb` and
`mlx_wired_limit_gb`) before model work, on every call. When you load components directly, call
`kinomlx.apply_memory_policy(resources.memory_policy)` first, as above. These limits apply to the
whole process; see [API memory policy](../PERFORMANCE.md#public-api-memory-policy).
