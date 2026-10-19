# LTX2 Python API

Use the Python API to consume frames directly or to put generation inside your own application.
Requests and settings use the config files' field names, not the flag spellings;
`kinomlx config inspect --model ltx2` shows each flag's field and where Python takes it.

- [Generate](#generate)
- [Several seeds](#several-seeds)
- [Reuse prepared resources](#reuse-prepared-resources)
- [Restart from Python](#restart-from-python)
- [Write video yourself](#write-video-yourself)
- [Memory policy](#memory-policy)

## Generate

`prepare_resources()` finds and checks the model files without loading any weights.
`generate_ltx2()` runs one request; consume its output inside the `with` block, which releases the
frames and models when it ends:

```python
from kinomlx import Settings
from kinomlx.models.ltx2 import LTX2Request, LTX2Settings, generate_ltx2, prepare_resources

resources = prepare_resources(LTX2Settings.from_env(), infrastructure=Settings.from_env())
request = LTX2Request(prompt="A quiet coastal sunrise, cinematic light")
with generate_ltx2(request, resources) as output:
    for frame in output.frames:
        consume(frame)
```

`LTX2Settings` holds the model files and runtime choices from `[model_settings]`, and `Settings`
the shared infrastructure from `[settings]`. Adapters and image conditions are typed records:
`LTX2Request(adapters=(LTX2AdapterConfig(...),), images=(LTX2ImageConditioningConfig(...),))`.

## Several seeds

`LTX2Runner.generate(request)` yields one `LTX2GenerationOutput` per seed. Close each output
before taking the next, and wrap the iterator in `contextlib.closing` when you may stop early, so
the shared conditioning is released promptly:

```python
from contextlib import closing

from kinomlx.models.ltx2 import LTX2Request, LTX2Runner

runner = LTX2Runner()
request = LTX2Request(prompt="A lighthouse at sunset", seeds=(42, 123, 999))
with closing(runner.generate(request)) as results:
    for result in results:
        with result:
            for frame in result.frames:
                consume(frame)
```

## Reuse prepared resources

A runner keeps prepared resources between calls, but not model weights, so an application can
host many calls without holding memory in between:

```python
from kinomlx.models.ltx2 import LTX2Runner, generate_ltx2

runner = LTX2Runner(resources=resources)
with runner.run(generate_ltx2, request) as output:
    consume(output)
```

`run(recipe, request)` accepts your own recipe with the same signature; see
[the complete example](../../examples/ltx2_distilled.py), which composes every stage explicitly and
writes the video with VideoToolbox.

## Restart from Python

`LTX2Runner.restart()` decodes saved latents or reruns stage 2. Choose an `LTX2Restart`
constructor and pass the saved files:

```python
from pathlib import Path

from kinomlx.models.ltx2 import LTX2Request, LTX2Restart

request = LTX2Request(width=768, height=448, frames=121, seed=43, generate_audio=True)

with runner.restart(request, LTX2Restart.decode(Path("outputs/source.safetensors"))) as output:
    consume(output)

stage_2 = LTX2Restart.stage_2(
    Path("outputs/source_stage1.safetensors"),
    text_conditioning=Path("outputs/source_text.safetensors"),
)
with runner.restart(request, stage_2) as output:
    consume(output)
```

Saved text conditioning records the model and text settings that made it. Direct reuse requires
them to match; a restart accepts a compatible replacement and records the differences.

## Write video yourself

The CLI writes video through `VideoToolboxGenerationSink`. A host can use it too:
`VideoToolboxGenerationSink(vae_frame_directory=...)` also saves the decoder's PNG frames, like
`--save-vae-frames`.

When you supply your own `GenerationSink`, pass an `OutputColorPlan` that describes the actual
signal. The plan labels the signal; it does not convert it, so labeling SDR pixels as HDR does not
make them HDR.

Lower-level pieces live with the model: `LatentState` in `kinomlx.models.ltx2.types` and
`GaussianNoiser` in `kinomlx.models.ltx2.noising`.

## Memory policy

Runners and recipes apply the memory policy from `Settings` (`mlx_cache_limit_gb` and
`mlx_wired_limit_gb`) before model work, on every call. The video decoder chooses its own cache
limit after loading unless `LTX2Request.vae_cache_limit_gb` sets one.

When you load components directly, apply the policy yourself first:

```python
from kinomlx import apply_memory_policy

apply_memory_policy(resources.memory_policy)
```

These limits apply to the whole process. See
[API memory policy](../PERFORMANCE.md#public-api-memory-policy) for lifetimes and concurrency.
