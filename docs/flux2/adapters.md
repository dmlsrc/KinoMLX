# Adapters with Klein

Klein takes LoRA, LoKr, LoHa, and the other [static
formats](../ADAPTERS.md#supported-static-export-formats), merged fresh for each generation. Choose
adapters made for your variant and recipe. The [shared adapter guide](../ADAPTERS.md) covers formats
and scaling in depth.

- [Add adapters](#add-adapters)
- [Turbo LoRAs for base](#turbo-loras-for-base)
- [Exclude parts of the model](#exclude-parts-of-the-model)
- [Scaling](#scaling)
- [Strength over a run](#strength-over-a-run)
- [Config files and Python](#config-files-and-python)

## Add adapters

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window" \
  --adapter style.safetensors:0.8 --adapter detail.safetensors:0.3
```

Each `--adapter` starts a new adapter, and the flags after it change only that one; `--lora` is
the same option. Strength defaults to 1. A numeric suffix after `.safetensors` sets the strength,
without checking whether the file exists; Klein takes one number, not a stage pair. A suffix that
is not a number, such as `:latest`, stays part of the path.

A filename does not decide the adapter type. An unsupported or incomplete export fails to load;
check its requirements in the [adapter guide](../ADAPTERS.md#keep-the-export-self-contained).

## Turbo LoRAs for base

A Turbo LoRA is an adapter trained to let base sample in far fewer steps. Use one made for Klein
base 9B, with its recommended steps and guidance. For a Turbo LoRA that supports four steps at
guidance 4, with a style adapter on top:

```bash
kinomlx --model flux2 --variant base --prompt "A painted landscape." \
  --negative-prompt "blurry" --guidance 4 --steps 4 \
  --adapter klein_turbo_lora.safetensors --adapter-strength 1 \
  --adapter style.safetensors --adapter-strength 0.8
```

The four steps come from the Turbo LoRA: `--steps 4` alone is not a usable base recipe, so keep
the default 50 steps without one. Turbo LoRAs are base recipes; distilled and KV take no external
guidance.

## Exclude parts of the model

`--adapter-exclude` leaves parts of the transformer unchanged by the latest adapter. Klein's
categories describe its own graph, so they differ from Krea 2's:

| Category | Leaves unchanged |
| --- | --- |
| `transformer` | The whole denoising transformer |
| `double`, `single` | The double-stream or the single-stream blocks |
| `input` | The image, text, and timestep embedders |
| `output` | The final norm and output projection |
| `attn` | Attention |
| `ff` | Feed-forward layers |
| `norm` | The query and key normalization |
| `modulation` | Modulation layers, including the final norm's |
| `bias` | Biases |
| `to_q`, `to_k`, `to_v`, `to_out` | Individual attention projections |

Exclusions act on whole stored weights. A single-stream block stores its query, key, value, and
feed-forward input as one fused weight, and its attention and feed-forward outputs as another. So
in single-stream blocks `ff` also leaves attention unchanged, `to_q`, `to_k`, or `to_v` leaves the
whole fused input weight unchanged, and `attn` covers the fused weights too.

## Scaling

Configured PEFT LoRA scales by `alpha/r`, or `alpha/sqrt(r)` for rank-stabilized LoRA, before your
strength. Other exports can use no rank scaling, notably dense LyCORIS and AI Toolkit LoKr. See
[strength, alpha, and scale](../ADAPTERS.md#strength-alpha-and-effective-scale) for the exact rules
and the handling of a saved alpha of 0.

Klein reads adapter configuration embedded by Diffusers in both its model-level and its
pipeline-level LoRA exports, including the pipeline's `transformer.` prefix.

## Strength over a run

`--adapter-step-strengths` and `--adapter-sigma-window` vary an adapter's strength over the
steps; see [Denoise schedules](../SCHEDULES.md#adapter-strength-over-a-run). A step list must
match the final step count.

The KV variant extracts its reference keys and values on the first step and reuses them. An
adapter whose strength changes after that would be read back at its first-step strength, so the
combination is refused while editing. Merge the adapter at one constant strength, or add
`--recompute-reference-kv`, which extracts the references again every step at the cost of one
extra reference pass per step.

## Config files and Python

TOML and `KINO_FLUX2_ADAPTERS` take an ordered `adapters` list. Each record has `path`,
`strength`, `exclude`, `step_strengths`, and `sigma_window`:

```toml
[generate]
adapters = [
  {path = "style.safetensors", strength = 0.8, exclude = ["modulation"]},
  {path = "detail.safetensors"},
]
```

A step list replaces its record's strength and window, so giving both is refused. Edit one record
with `--set 'generate.adapters[1].strength=0.25'`, replace it whole with
`--set 'generate.adapters[1]={path="detail.safetensors"}'`, or clear the list with
`--set 'generate.adapters=[]'`. A strength you left out stays out of the saved configuration.

Python requests take `Flux2Request(adapters=(Flux2AdapterConfig(...), ...))` from
`kinomlx.models.flux2`.
