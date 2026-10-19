# Performance and memory

Start with a model's defaults and save a run log. Output size, video length, steps, reference
images, and adapters all change time and memory. Change one setting at a time, and compare the
phase that dominates your run.

- [Memory controls](#memory-controls): the allocator cache, wiring, and what zero means.
- [Energy mode](#energy-mode): High Power keeps long runs from throttling.
- [Choose a change](#choose-a-change): what to try for memory pressure or slow runs.
- [Precision](#precision): FP16 against BF16.
- [Compare runs](#compare-runs): settings, timings, and repeatability.
- [Public API memory policy](#public-api-memory-policy): runners and component calls.

Each model's performance page covers its own levers: [LTX2](ltx2/performance.md),
[Krea 2](krea2/performance.md), and [Klein](flux2/performance.md).

## Memory controls

These apply to every model. Despite the `-gb` spelling, values are GiB (`1024**3` bytes), finite
and not negative.

| Control | Default | `0` means |
| --- | --- | --- |
| `--mlx-cache-limit-gb` | 1 GiB, LTX2 streaming included; LTX2's decoder chooses its own | Keep no freed allocations for reuse |
| `--mlx-wired-limit-gb` | The device's recommended working set | Wire nothing |

Neither caps the process's total memory or touches model files. Weights, activations, temporary
tensors, CPU allocations, and the file cache still need memory.

### Allocator cache

MLX can keep a freed buffer to reuse for the next tensor. The cache limit sets the size of that
pool of unused memory; it does not limit live tensors, so a 1 GiB cache can sit beside tens of GiB
of model memory.

`--mlx-cache-limit-gb 0` disables the cache. Generation still works, but repeated allocations may
cost more; zero is not "unlimited" or "automatic". Lowering the limit frees the excess at the next
allocation, not immediately.

A larger cache allows more reuse but keeps more unused memory. LTX2 streaming reloads blocks
repeatedly and can be sensitive to this trade. KinoMLX applies your limit without a
streaming-specific minimum. Start at 1 GiB and measure before changing it; no larger value is a
proven gain for every workload.

```bash
kinomlx --model krea2 --prompt "A ceramic cup in soft window light" \
  --mlx-cache-limit-gb 2 --save-run-log
```

The same setting is `mlx_cache_limit_gb = 2.0` under `[settings]` in TOML, or
`KINO_MLX_CACHE_LIMIT_GB=2` in the environment.

### Memory residency

Wiring keeps MLX buffers resident instead of letting macOS compress or page them out. The wired
limit is a ceiling, not a reservation: setting it allocates nothing, and allocations can exceed
it, so it does not guarantee a run fits or stays resident.

By default KinoMLX reads the device's recommended working set at each run. Choose a lower limit
with `--mlx-wired-limit-gb`, `[settings].mlx_wired_limit_gb`, or `KINO_MLX_WIRED_LIMIT_GB`; a value
above the device limit is refused, and zero turns wiring off without touching the cache.

macOS sets the automatic limit itself. One measurement:

| Mac | `iogpu.wired_limit_mb` | Automatic limit |
| --- | --- | ---: |
| M1 Max, 64 GB | `0` (automatic) | 51.84 GiB, about 81% of memory |

That is one machine and macOS version, not a fixed share for every Mac. To see your current policy
and the effective limit, in the Python environment where KinoMLX is installed:

```bash
sysctl iogpu.wired_limit_mb
python -c 'import mlx.core as mx; print(mx.device_info()["max_recommended_working_set_size"] / 1024**3, "GiB")'
```

`0` from `sysctl` means the automatic policy is active, and the Python line shows its limit. A
nonzero value means a system override is active, and the Python line shows the limit under it.

To measure the automatic default while an override is active, note the current value, stop any
generation, and restore automatic policy:

```bash
sudo sysctl iogpu.wired_limit_mb=0
```

Then run the Python check in a fresh process. This changes a system setting: reapply your
override afterward if you want it, and start a fresh KinoMLX process before generating.

#### Optional system override

If the automatic ceiling is what limits you, macOS's `iogpu.wired_limit_mb` raises it system-wide.
For example, 61440 MiB gives a 60 GiB ceiling on a 64 GiB Mac. That leaves 4 GiB outside the
ceiling, which may or may not be enough for the rest of your work.

This table leaves 4 GiB outside the ceiling, using `(installed GiB - 4) * 1024`:

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

These are optional ceilings, not reservations, and 4 GiB is not enough for every workload: leave
more when other applications need it. On small Macs a table value can be below the automatic
limit.

With generation stopped, an administrator applies the example with:

```bash
sudo sysctl iogpu.wired_limit_mb=61440
```

Start a fresh KinoMLX process afterward. The override lasts until reboot. To restore automatic
policy:

```bash
sudo sysctl iogpu.wired_limit_mb=0
```

The two zeros differ: **system `iogpu.wired_limit_mb=0` restores automatic policy, while KinoMLX
`--mlx-wired-limit-gb 0` turns MLX wiring off**. A higher ceiling does not make a run need less
memory.

#### Apply the override at boot with launchd

Try the temporary override first, through representative runs alongside your usual applications.
Make it permanent only once your chosen limit runs reliably without heavy memory pressure; until
then a reboot is an easy way back.

A system LaunchDaemon can apply the override at every boot, before login, by running `sysctl`
directly. It is an optional macOS setting, independent of KinoMLX.

With generation stopped, paste this into a terminal in a working folder. It writes the plist,
validates it, installs it owned by root, and loads it. It uses the 64 GiB example; replace `61440`
with your value from the table first.

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

Each step runs only if the one before succeeds. The commands assume the service is not installed
yet; inspect an existing file before replacing it. `RunAtLoad` applies the setting when loaded now
and at every boot, and there is no `KeepAlive`: the command runs once and exits.

Check the service and the live setting:

```bash
sudo launchctl print system/com.local.sysctl-iogpu
sysctl iogpu.wired_limit_mb
```

Look for `last exit code = 0` and your value. `not running` is normal after the command finishes.
Start a fresh KinoMLX process, and use the Python check above to confirm the effective limit.

To change the value later, stop generation, edit your plist, unload the service with
`sudo launchctl bootout system/com.local.sysctl-iogpu`, and repeat the validation, install, and
bootstrap commands.

To remove the boot override, stop generation and run:

```bash
sudo launchctl bootout system/com.local.sysctl-iogpu
sudo rm /Library/LaunchDaemons/com.local.sysctl-iogpu.plist
```

Removing the daemon does not undo the setting already applied: restore automatic policy with the
`sysctl` reset above, or reboot. Start a fresh KinoMLX process after changing the policy.

## Energy mode

On a Mac that offers High Power, choose it before long runs: System Settings > Battery > Energy
Mode > On power adapter > High Power, or System Settings > Energy on a desktop Mac.

Under Automatic, macOS throttles the GPU after a few minutes of full load: it lowers the GPU's
clock instead of running the fans faster, so a long run slows as it goes. High Power runs the fans
faster and keeps the full clock, at the cost of more fan noise.

A sustained GPU workload on an M1 Max:

| Energy mode | Speed |
| --- | --- |
| Automatic | Full speed for about 3.5 minutes, then about 20% longer for the same work |
| High Power | Full speed for the whole run |

## Choose a change

| Observation | Try | Trade-off |
| --- | --- | --- |
| The LTX2 transformer does not fit | [Streaming](ltx2/performance.md#run-on-32-gb) | Reloads weights every step |
| Streamed LTX2 spends its time reading weights | [Keeping some tensors loaded](ltx2/performance.md#keep-some-tensors-loaded) | More live memory |
| LTX2 weight memory is too high | [Quantization](ltx2/performance.md#use-less-transformer-memory) | Changes the output; slower |
| Decoding sets the peak | Keep automatic tiling ([LTX2](ltx2/decode.md#memory-tiling-and-cache), [Krea 2](krea2/performance.md#decoder-memory)), or a smaller size | Tiling adds work; Klein has none |
| A long run slows after a few minutes | [High Power](#energy-mode) on the power adapter | Louder fans |
| Later runs get slower | Rerun the baseline after a cooldown; close other GPU work | Heat, power limits, and memory pressure skew timings |
| The first-use adapter check is slow | Read its time in the log before turning it off | Turning it off loses a check |

When the whole working set is too large, use a smaller size, fewer references, or a shorter
video. Raising a memory limit does not shrink the working set.

## Precision

LTX2 runs its transformer in FP16 by default, and Krea 2 and Klein in BF16, the precision of their
weights. `--transformer-dtype` selects either. Both use two bytes per value: FP16 has more
mantissa precision but a narrower range, so it can overflow where BF16 does not. Which is faster
depends on the model's kernels; see [LTX2 precision](ltx2/performance.md#precision).

Precision changes the sample, so one seed does not make FP16 and BF16 images comparable pixel for
pixel. Judge the result as well as the clock.

Decoder precision is separate: LTX2 has `--vae-decode-dtype`, and Krea 2 and Klein `--vae-dtype`,
each with its own choices; see [LTX2 decoding](ltx2/decode.md#decode-precision), [Krea 2
precision](krea2/performance.md#precision), and [Klein precision](flux2/performance.md#precision).

## Compare runs

1. Save the settings and timings with `--save-effective-config` and `--save-run-log`.
   `--save-all-sidecars` includes them, but also saves large tensor sidecars.
2. Keep model files, prompt, seed, size, steps, adapters, precision, and both memory limits fixed,
   except the setting under test.
3. Close other GPU work. Start runs at a similar temperature and memory pressure, in the same
   [energy mode](#energy-mode); the file cache can also make later runs faster.
4. Repeat, and rerun the original baseline at the end. If the baseline drifts as much as the
   apparent gain, the result is inconclusive.
5. Compare total time, phase timings, memory peaks, and the output. A lower denoising peak is not
   necessarily a lower whole-run peak.

The run record, `<stem>_run.json`, and `--json` give the numbers in machine-readable form; the
saved TOML records the memory limits used. For Instruments traces, add `--profile-signposts`, and
`--profile-signpost-log PATH` to save the log.

Sustained GPU work heats the machine, and under Automatic macOS lowers the GPU's clock, so a
slower second run alone does not show a setting is slower.

A seed makes inputs repeatable, but not outputs across changes of dtype, quantization, kernels, or
recipe.

## Public API memory policy

`Settings.mlx_cache_limit_gb` and `Settings.mlx_wired_limit_gb` set both controls for every
model. Preparing resources records them without touching MLX. Runners and public recipes apply
them before model work, on every call, reused resources included; nested calls apply them once.

```python
from kinomlx import Settings
from kinomlx.models.krea2 import Krea2Runner, Krea2Settings

runner = Krea2Runner(
    Krea2Settings(variant="turbo"),
    infrastructure=Settings(mlx_cache_limit_gb=1.0, mlx_wired_limit_gb=None),
)
```

Your own recipes taking `(request, resources, ...)` can use `@kinomlx.memory_managed` to apply
the same policy, whether called directly or through a runner; see
[the explicit LTX recipe example](../examples/ltx2_distilled.py).

A wired limit of `None` reads the device recommendation at each call, and zero turns wiring off,
even after an earlier run turned it on. A cache limit of `None` leaves the current setting alone,
and zero disables the cache.

Component loaders and custom model code apply the policy themselves, before loading weights or
allocating inputs:

```python
from kinomlx import apply_memory_policy

apply_memory_policy(resources.memory_policy)
# Load components and run model work here.
```

`apply_memory_policy` also accepts `Settings` or `MLXMemoryPolicy`. These limits apply to the whole
process: run one model call at a time, consume and close lazy outputs before starting another, and
change policies only while MLX is idle. A policy stays in effect until the next call applies its
own.
