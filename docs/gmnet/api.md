# GMNet Python API

GMNet's Python API takes a typed request, prepared resources, and an output plan that writes the
bundle transactionally, like the CLI.

```python
from pathlib import Path

from kinomlx import Settings
from kinomlx.models.gmnet import (
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    GMNetSettings,
    expand_gmnet,
    plan_gmnet_output,
    prepare_resources,
)

resources = prepare_resources(GMNetSettings.from_env(), infrastructure=Settings.from_env())
request = GMNetRequest(Path("photo.jpg"))
plan = plan_gmnet_output(request, GMNetOutputConfig(directory=Path("hdr"), save_gain_map=True))
runner = GMNetRunner(resources=resources)

with plan.reserve() as reservation:
    result = runner.run(expand_gmnet, request)
    artifacts = GMNetOutputSink(plan).write(result, reservation=reservation)
```

Reserve the output before running, so a name conflict fails before any work.
`GMNetRunner.expand(request)` is shorthand for `runner.run(expand_gmnet, request)`. A host with
its own output handling can consume the returned scene-linear `ExpansionResult` directly.

`GMNetRunner(components=...)` accepts an injected `GMNetComponentsProvider`, which hands out the
network as a bounded lease, for hosting and tests.

[`examples/gmnet_expand.py`](../../examples/gmnet_expand.py) is the complete small example.

## Memory policy

Runners and recipes apply the memory policy from `Settings` (`mlx_cache_limit_gb` and
`mlx_wired_limit_gb`) before model work, on every call. Prepared resources carry the policy, so
reusing them keeps those choices. When you load components directly, call
`kinomlx.apply_memory_policy(resources.memory_policy)` first. These limits apply to the whole
process; see [API memory policy](../PERFORMANCE.md#public-api-memory-policy).
