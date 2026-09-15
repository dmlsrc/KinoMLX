# Performance and memory

Start with the model's default settings and save a run log. Output dimensions,
video length, sampling steps, reference images, and adapters all affect time and
memory. Change one setting at a time and compare the phase that dominates your run.

- [Memory controls](#memory-controls): allocator caching, wiring, and zero values.
- [Choose a change](#choose-a-change): what to try for memory pressure or slow runs.
- [Example run times](#example-run-times): two measured LTX-2.3 runs as a planning anchor.
- [Compare runs](#compare-runs): capture settings, timings, and repeatability.
- [Public API memory policy](#public-api-memory-policy): configure runners and component calls.

## Memory controls

These controls apply to all four model families. Despite the `-gb` spelling,
values are GiB (`1024**3` bytes), and must be finite and non-negative.

| Control | Default | Meaning of `0` |
| --- | --- | --- |
| `--mlx-cache-limit-gb` | 1 GiB, including LTX streaming | Disable caching of unused MLX allocations |
| `--mlx-wired-limit-gb` | Current device recommended working set | Disable MLX wiring |

Neither setting caps total process memory or changes downloaded checkpoint files.
Live weights, activations, temporary tensors, CPU allocations, and filesystem
caching still need memory.

### Allocator cache

MLX can retain a freed tensor's allocation for another tensor to reuse. The cache
limit controls this unused-memory pool. It does not limit the size of live tensors:
a 1 GiB cache can coexist with tens of GiB of active model memory.

`--mlx-cache-limit-gb 0` passes zero to MLX and disables this cache. Generation
still works, but repeated allocations may cost more. It is not an unlimited or
automatic setting. MLX documents that lowering the limit reclaims excess cached
memory on the next allocation; changing the limit is not an immediate cache-clear
operation in an already running Python process.

A larger cache allows more buffer reuse while retaining more unused memory.
Streaming repeatedly reloads block weights and can be sensitive to this tradeoff.
KinoMLX honors your selected limit without imposing a streaming-specific minimum.
Start with 1 GiB; any speed benefit from changing it needs a comparison on your
workload. There is no established universal advantage to a 4 GiB streaming cache.

For example, to select a 2 GiB cache for one generation:

```bash
kinomlx --model krea2 --prompt "A ceramic cup in soft window light" \
  --mlx-cache-limit-gb 2 --save-run-log
```

The same setting can be supplied through TOML or the environment:

```toml
[settings]
mlx_cache_limit_gb = 2.0
```

```bash
export KINO_MLX_CACHE_LIMIT_GB=2
```

### Memory residency

Wiring allows MLX buffers to remain resident instead of being compressed or paged
out by macOS. The wired limit is a ceiling, not a reservation: setting it does not
allocate that much memory. Allocations can exceed it, so it is not a guarantee that
a generation fits or remains fully resident.

By default KinoMLX reads `max_recommended_working_set_size` from the device at each
invocation. You can select a lower limit with `--mlx-wired-limit-gb`,
`[settings].mlx_wired_limit_gb`, or `KINO_MLX_WIRED_LIMIT_GB`. A value above the
reported device limit is rejected. Zero disables wiring without disabling the
allocator cache.

On a 64 GiB M1 Max, the automatic system policy was measured at **51.84 GiB**,
about **81% of physical memory**, with `iogpu.wired_limit_mb=0`. This is an
observed default for that machine and macOS version, not a fixed percentage for
every Mac. The 60 GiB value in the table below is an explicit override.

To inspect your current policy and effective device recommendation, use the
Python environment where KinoMLX is installed:

```bash
sysctl iogpu.wired_limit_mb
python -c 'import mlx.core as mx; print(mx.device_info()["max_recommended_working_set_size"] / 1024**3, "GiB")'
```

If `sysctl` reports `0`, automatic policy is active and the Python command shows
its effective limit. A nonzero value means a system override is active; the
Python command then shows the effective limit under that override, not the
automatic default.

To measure the automatic default while an override is active, first record the
current `sysctl` value and stop generation. Restore automatic policy:

```bash
sudo sysctl iogpu.wired_limit_mb=0
```

Then rerun the Python check above in a fresh process. This changes the system
policy. Reapply the recorded override afterward if you want to keep it, and start
a fresh KinoMLX process before resuming generation.
Leave room for macOS, CPU-side model work, and other applications when considering
a larger wired budget.

#### Optional system override

If the automatic ceiling is the limiting factor, macOS provides the system-wide
`iogpu.wired_limit_mb` override. For example, 61440 MiB selects a 60 GiB ceiling
on a 64 GiB Mac. That leaves only 4 GiB outside the ceiling; whether this is enough
depends on the rest of the workload. This is an example, not a preset for every Mac.

The table leaves 4 GiB outside the GPU wired-memory ceiling, using
`(installed GiB - 4) * 1024` for the `iogpu.wired_limit_mb` value:

| Unified memory | GPU wired limit | `iogpu.wired_limit_mb` |
| --- | --- | --- |
| 8 GiB | 4 GiB | 4096 |
| 16 GiB | 12 GiB | 12288 |
| 18 GiB | 14 GiB | 14336 |
| 24 GiB | 20 GiB | 20480 |
| 32 GiB | 28 GiB | 28672 |
| 36 GiB | 32 GiB | 32768 |
| 48 GiB | 44 GiB | 45056 |
| 64 GiB | 60 GiB | 61440 |
| 96 GiB | 92 GiB | 94208 |
| 128 GiB | 124 GiB | 126976 |

These are optional ceilings, not memory reservations or a guarantee that 4 GiB is
enough for every workload. Leave more headroom when other applications or CPU-side
allocations need it. On smaller machines, a table value may be lower than the
automatic device recommendation.

With generation stopped, an administrator can apply that example with:

```bash
sudo sysctl iogpu.wired_limit_mb=61440
```

Start a fresh KinoMLX process afterward. The override resets on reboot unless a
separate launch configuration reapplies it. To restore the automatic system policy:

```bash
sudo sysctl iogpu.wired_limit_mb=0
```

The two zeros have different meanings: **system `iogpu.wired_limit_mb=0` restores
automatic policy; KinoMLX `--mlx-wired-limit-gb 0` disables MLX wiring**. Raising
the system ceiling does not reduce the generation's memory requirement.

#### Apply the override at boot with launchd

Try the temporary `sysctl` override first and use it through representative
generations alongside your usual applications. Once the chosen limit runs reliably
without excessive memory pressure, you can make it persistent with launchd.
Until then, leaving it temporary provides a straightforward recovery path: a reboot
restores the automatic macOS policy, provided no existing boot configuration
reapplies the override.

A system LaunchDaemon can apply the override at every boot, before user login.
It runs `sysctl` directly; no shell script or Python dependency is needed.
This is an optional macOS configuration, independent of the KinoMLX installation.

With generation stopped, paste the following into a terminal in a local working
directory. It creates the plist, validates it, installs it with root ownership and
normal LaunchDaemon permissions, and loads it into the system domain. The example
uses the 60 GiB ceiling for a 64 GiB Mac; replace `61440` with your chosen MiB value
from the table above before running it.

```bash
cat > com.local.sysctl-iogpu.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.sysctl-iogpu</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/sbin/sysctl</string>
        <string>iogpu.wired_limit_mb=61440</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
PLIST

plutil -lint com.local.sysctl-iogpu.plist &&
sudo install -o root -g wheel -m 644 com.local.sysctl-iogpu.plist /Library/LaunchDaemons/com.local.sysctl-iogpu.plist &&
sudo launchctl bootstrap system /Library/LaunchDaemons/com.local.sysctl-iogpu.plist
```

Installation and loading proceed only if the preceding command succeeds. These
commands assume the service is not already installed; inspect an existing file
before replacing it. `RunAtLoad` applies the setting immediately on bootstrap and
again when the daemon loads at boot. `KeepAlive` is intentionally absent: this
command should run once and exit.

Verify the service and the live setting:

```bash
sudo launchctl print system/com.local.sysctl-iogpu
sysctl iogpu.wired_limit_mb
```

Look for `last exit code = 0` and the selected MiB value. A `not running` service
state is normal after this one-shot command completes. Start a fresh KinoMLX
process; the Python check above can confirm the effective device recommendation.

To change the persistent value later, stop generation, edit your local plist, and
unload the existing service with
`sudo launchctl bootout system/com.local.sysctl-iogpu`. Then repeat the validation,
installation, and bootstrap commands above.

To remove the boot override, stop generation and run:

```bash
sudo launchctl bootout system/com.local.sysctl-iogpu
sudo rm /Library/LaunchDaemons/com.local.sysctl-iogpu.plist
```

Unloading or deleting the daemon does not undo the already applied setting.
Restore automatic policy with the `sysctl` reset command above, or reboot after
removing the daemon. Start a fresh KinoMLX process after changing the policy.

## Choose a change

| Observation | Useful next step | Tradeoff or limit |
| --- | --- | --- |
| LTX transformer does not fit comfortably | Use [block streaming](LTX2.md#block-streaming-on-32-gb-systems) | Reloads weights and repeats preparation during denoising |
| Streamed LTX spends much of its time reading weights | Try [selective tensor retention](LTX2.md#retaining-selected-block-tensors) | Reduces repeated work by keeping more tensors resident |
| LTX weight memory is too high | Consider [transformer quantization](LTX2.md#transformer-quantization) | Changes weights and output; can be slower |
| Decode is the memory peak | Keep automatic VAE tiling for LTX or Krea; reduce geometry if needed | Tiling adds work; Klein currently has no VAE tiling |
| Successive runs get slower | Repeat the baseline after cooldown and check competing GPU work | Temperature, power limits, and memory pressure can confound timings |
| First-use adapter checking is expensive | Inspect the weight-check timing before opting out | Disabling the scan removes a validation step |

Reduce image dimensions, reference count, or video length when the total working
set is too large. Increasing a memory limit alone does not reduce that working set.

### Precision

LTX, Krea, and Klein default to BF16 transformer execution. Klein's default
preserves the bundled weights' precision; FP16 is an optional speed optimization.
Use `--transformer-dtype float16` to opt in or `--transformer-dtype bfloat16` to
select BF16 explicitly. Both types use two bytes per element. FP16 has more
mantissa precision but a narrower numerical range; it can overflow where BF16
would not. Speed depends on the model's kernels and workload.

LTX's STEEL attention kernels are BF16-only, so an FP16 LTX transformer runs
attention on stock MLX scaled dot-product attention instead. That is a fallback,
not an error, and it does not always cost time: for one measured pair of 30-second
runs the FP16 configuration finished sooner despite it. See
[example run times](#example-run-times) for that comparison. Precision changes the
sample, so a shared seed does not make FP16 and BF16 outputs comparable frame for
frame; judge the result as well as the clock.

Decoder precision is separate: Krea exposes `--vae-dtype`, and LTX exposes
`--vae-decode-dtype` with decoder-specific restrictions. See the
[Krea precision guide](KREA2.md#denoising-mode-and-precision) and
[LTX guide](LTX2.md) before changing them. Compare output as well as elapsed time.

### Adapter validation

`--check-adapter-weights` is enabled by default for the generative models. It
checks adapter-modified active floating-point weights after first use; streamed
LTX checks each newly used target before its group is evicted. A new adapter
profile can require another check. Packed LTX targets rely on the latent check.

`--no-check-adapter-weights` skips that weight scan while preserving per-step
latent finite checks. Use it only when you accept losing that diagnostic, such as
when repeating an already checked model, dtype, and adapter configuration.
See [weight validation](WEIGHTS.md#precision-and-validation) for coverage.

## Example run times

The two runs below are the same LTX-2.3 job on one machine, differing only in
transformer precision. Use them as a planning anchor for how long a run of this
shape takes, not as a benchmark or a performance guarantee. Different hardware, a
different macOS or MLX build, competing GPU work, or a warmer machine will produce
different numbers, and a single run does not establish a difference.

Measured with KinoMLX 0.0.1 on an Apple M1 Max with 64 GiB of unified memory,
macOS 27.0 and MLX 0.32.2, with the 60 GiB system wired-memory override applied.

Both runs used these settings; only `--transformer-dtype` differed:

```bash
kinomlx --model ltx2 --model-generation 2.3 --prompt-file PROMPT.txt \
  --width 768 --height 448 --duration 30 --seed 42 \
  --generate-audio --save-all-sidecars --output-prefix PREFIX
```

| Input | Value |
| --- | --- |
| Model | LTX-2.3 distilled 22B, `ltx-2.3-22b-distilled-1.1` checkpoint |
| Output | 768x448, 721 frames, 24 fps, 30.0 s timeline, joint 48 kHz stereo audio |
| Sampling | Distilled two-stage, 8 + 3 steps, deterministic Euler, seed 42 |
| Adapters | None |
| Execution | Fast mode, compiled attention, all blocks resident, no quantization |
| Decode | Automatic VAE tiling, conv video VAE |
| Memory limits | 1 GiB allocator cache, wired limit at the device recommendation |

| Phase | BF16 | FP16 |
| --- | --- | --- |
| Encode prompt with Gemma 3 | 3.6 s | 3.7 s |
| Project audio/video text context | 1.8 s | 1.8 s |
| Stage 1 denoise, 8 steps at half resolution | 287.6 s | 255.4 s |
| Upscale video latent | 4.7 s | 4.7 s |
| Stage 2 denoise, 3 steps at full resolution | 663.5 s | 594.4 s |
| Vocode audio with bandwidth extension | 14.1 s | 14.1 s |
| VAE decode tiles | 65.8 s | 65.5 s |
| Total | 1044 s (17.4 min) | 942 s (15.7 min) |
| Peak memory | 41.1 GiB | 41.1 GiB |

Denoising was about 90% of both runs, so the sampling stages are where a change to
geometry, length, or step count shows up. At this geometry the runs cost roughly
35 s (BF16) and 31 s (FP16) of wall time per second of finished video. Neither
number scales linearly to another resolution or duration: stage 2 runs at full
resolution over the whole timeline, and decode and audio work scale separately.

The FP16 run finished 9.7% sooner, and all of that came from denoising. Prompt
encoding, latent upscaling, vocoding, and decode matched the BF16 run to within a
few tenths of a second. Peak memory was unchanged, as expected for two types that
both use two bytes per element. Note that FP16 gave up the BF16-only STEEL
attention path and still finished sooner on this workload; see
[precision](#precision) before choosing it, and compare output, not only elapsed time.

## Compare runs

1. Save the resolved configuration and timings with `--save-effective-config` and
   `--save-run-log`. `--save-all-sidecars` also enables these, but saves additional
   model-dependent artifacts that can consume substantial disk space.
2. Keep model sources, prompt, seed, geometry, steps, adapters, precision, and both
   memory limits fixed except for the setting under comparison.
3. Close competing GPU workloads. Compare runs with similar starting temperature
   and memory pressure; filesystem caching can also make later runs faster.
4. Repeat the comparison and run the original baseline again at the end. If the
   baseline drift is comparable to the apparent improvement, the result is inconclusive.
5. Compare total time, phase timings, memory peaks, and output quality. A lower
   denoising peak is not necessarily a lower whole-run peak.

The run sidecar uses the output stem, for example `<stem>_run.json`.
`--json` emits a machine-readable result on stdout. The saved effective TOML
records the resolved configuration, including the selected memory limits.
For Instruments tracing, use `--profile-signposts`; add
`--profile-signpost-log PATH` to save the profiling log.

Sustained GPU work can heat the machine and reduce its available power. A slower
second run alone does not establish that a setting is slower. Likewise, a seed
makes inputs repeatable but does not guarantee identical outputs across changes
in dtype, quantization, kernels, or model recipe.

## Public API memory policy

`Settings.mlx_cache_limit_gb` and `Settings.mlx_wired_limit_gb` configure both
controls for every model. Resource inspection captures those choices without changing
MLX state. Runners and public recipes apply them before model work, including when
reusing prepared resources. Nested runner/recipe calls apply the policy once.

```python
from kinomlx import Settings
from kinomlx.models.krea2 import Krea2Runner, Krea2Settings

runner = Krea2Runner(
    Krea2Settings(variant="turbo"),
    infrastructure=Settings(mlx_cache_limit_gb=1.0, mlx_wired_limit_gb=None),
)
```

Custom recipes with `(request, resources, ...)` parameters can use
`@kinomlx.memory_managed` to establish the same boundary. They work both as
standalone calls and through a runner; see
[the explicit LTX recipe example](../examples/ltx2_distilled.py).

The wired default (`None`) reads the current device recommendation on each invocation.
Zero actively disables wiring, including after a previous run enabled it. A `None`
allocator limit leaves the host's current allocator setting unchanged; zero disables
allocator retention.

For standalone component loaders or custom model execution, apply the same policy
explicitly before loading weights or allocating inputs:

```python
from kinomlx import apply_memory_policy

apply_memory_policy(resources.memory_policy)
# Load components and run model work here.
```

`apply_memory_policy` also accepts `Settings` or `MLXMemoryPolicy` and returns `None`.
These are process-wide controls. Serialize model invocations,
consume and close lazy outputs before starting another, and change policies only
while MLX is idle. Limits remain in effect after the invocation; the next invocation
applies its selected policy.

## Model guides

- [LTX2](LTX2.md): streaming, retention, quantization, adapter accumulation, and VAE choices.
- [Krea 2](KREA2.md): Raw/Turbo steps, denoising precision, editing, and the optional 2x VAE.
- [Klein](FLUX2.md): base/distilled/KV recipes, reference count, and reference KV lifetime.
- [GMNet](GMNET.md): variant selection and HDR still output.

KinoMLX prepares weights from original checkpoints when components are first used
and releases components at model-defined phase boundaries. There is no generated
model-weight cache to build on disk. See [Weight loading](WEIGHTS.md) for source
selection, preparation, and component lifetime.
