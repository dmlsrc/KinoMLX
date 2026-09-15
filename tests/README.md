# Tests

Tests are grouped by model or shared subsystem and named for the behavior they verify.
Run commands from the repository root with the project environment active.

```bash
python -m pytest tests
python -m pytest tests --run-heavy -m heavy
```

Tests marked `heavy` load real checkpoints or run long GPU work. They are deselected by
default and selected with `--run-heavy`, which always runs them in one process; a heavy
test handed to a parallel worker by any other route refuses to start. Every other test
runs in parallel (`-n auto`) and must pass under arbitrary interleaving. `slow` labels the
heavy tests that take minutes, so `--run-heavy -m 'heavy and not slow'` runs the quick
checkpoint checks.

Real assets are resolved the way the product resolves them: the model settings from the
environment first, then the local Hugging Face cache. A cheap real-asset test (a header
read, tokenizer parity against the reference library, one small component load) runs in
the default suite and skips with the reason when the cache does not hold its input; the
heavy selection uses the same resolution, and individual tests explain missing inputs or
additional opt-ins in their skip messages. Pytest redirects generated application caches
into temporary test directories.

## Model test organization

- `models/krea2/test_generation.py`: request, sampler, batching, and component lifetime contracts.
- `models/krea2/_*_fixtures.py`: shared text-conditioning, transformer, VAE, and generation fixtures.
- `models/krea2/*_gate.py`: standalone text-conditioning, transformer, VAE, and generation checks.
- `models/ltx2/test_distilled_sampler.py`: injected-noise sampler parity and stage-2 determinism.
- `models/ltx2/test_real_ltx25_component_parity.py`: component loading and reference parity.
- `models/ltx2/test_real_ltx25_temporal_components.py`: duration and temporal component oracles.
- `models/ltx2/test_real_ltx25_pipeline.py`: public LTX-2.5 pipeline integration.
- `models/ltx2/test_real_ltx25_capabilities.py`: additional public LTX-2.5 capability checks.
- `models/ltx2/ltx25_*_gate.py`: standalone pipeline and capability checks.

Standalone gates accept `--help` and support source or installed-wheel validation. They are
explicit integration tools, not automatically collected unit tests.

## Additional LTX-2.5 integration inputs

| Variable | Purpose |
| --- | --- |
| `KINO_RUN_REAL_LTX25` | Set to `1` to enable the full public-pipeline gate. |
| `KINO_RUN_REAL_LTX25_CAPABILITIES` | Set to `1` for capability and temporal component gates. |
| `KINO_LTX25_COMPONENT_PARITY_FIXTURE_DIR` | Directory containing the pinned component reference fixtures. |
| `KINO_LTX25_RECEIPT_DIR` | Optional durable directory for the full integration receipts; the default is the test's temporary directory. |

These opt-ins supplement `--run-heavy`; they do not replace it. The opt-in Krea text-conditioning
parity test reads its pinned fixture directory from `KINO_KREA_ORACLE_TEXT_FIXTURE_DIR`.

## Keeping tests focused

Use small fixtures that exercise the real behavior under test. Load only the components
that behavior needs. Keep end-to-end preparation checks separate from metadata and replay tests.
Shared helpers belong under `tests/`; the runtime must not acquire test-only implementations.
Source policy checks share a parsed-source fixture, and one test owns repository ASCII coverage.

Real component tests load the selected original checkpoints. The full LTX-2.3 transformer
check additionally requires `KINO_RUN_REAL_LTX23_TRANSFORMER=1` because it evaluates a
large transformer and requires substantial unified memory.
