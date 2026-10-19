# Generating images

This page covers text-to-image choices for Klein: the recipe, size, prompts, guidance, and
schedules. Every argument it mentions is in [the catalog](arguments.md).

- [Recipes](#recipes)
- [Size](#size)
- [Several prompts and images](#several-prompts-and-images)
- [Guidance branches](#guidance-branches)
- [Set the schedule yourself](#set-the-schedule-yourself)
- [Noise](#noise)
- [Output](#output)
- [Config files](#config-files)

## Recipes

The default `distilled` variant takes four steps with no external guidance:

```bash
kinomlx --model flux2 --prompt "A ceramic vase beside a window." --seed 42
```

`base` takes 50 steps with guidance 4, and accepts a negative prompt:

```bash
kinomlx --model flux2 --variant base --prompt "A painted landscape." \
  --negative-prompt "blurry" --guidance 4
```

`--steps` and `--guidance` override the recipe. Guidance applies to base only, and must be at
least 1; distilled and KV refuse anything but 1. A short base recipe needs a
[Turbo LoRA](adapters.md#turbo-loras-for-base): `--steps 4` alone does not make base a four-step
model.

## Size

`--width` and `--height` default to 1024 and must be multiples of 16, at least 64. With one
reference image and no size given, the output follows the reference's shape; see
[output size](editing.md#output-size).

## Several prompts and images

Repeat `--prompt` to run several prompts in order. `--negative-prompt` takes one value for every
prompt, or one per prompt. `--images-per-prompt N` makes N images of each prompt, one after
another, with consecutive seeds; see [seeds](../USAGE.md#seeds).

A prompt longer than 512 tokens after templating is cut off, with a notice.

`--prompt` and `--prompt-file` cannot both come from one config file or command line.

## Guidance branches

With guidance, each base step runs a positive and a negative branch. `--cfg-mode` chooses how:

| `--cfg-mode` | Behavior |
| --- | --- |
| `sequential` (default) | One branch after the other, using less memory |
| `batched` | Both branches together: more memory, and slightly different rounding |

It never batches separate images together.

## Set the schedule yourself

Klein shifts its noise levels by the size of the output, so one step count gives different levels
at different sizes. The sampling section of the log shows the resolved schedule, and the run
record keeps it per image. `--sigmas` gives the levels directly:

```bash
kinomlx --model flux2 --prompt "A quiet coastal sunrise, cinematic light" \
  --sigmas "1,0.85,0.6,0.3,0"
```

An explicit list ignores the output size and fixes the step count: five values run four steps.
Values must be finite, between 0 and 1, and strictly decreasing. A denoise restart may change the
schedule. See [Denoise schedules](../SCHEDULES.md).

## Noise

`--noise-backend torch-mps` reproduces PyTorch's MPS noise from the same seed. A tool using a
PyTorch CPU generator still produces different noise from that seed. To compare implementations
exactly, feed both the same explicit noise and conditioning tensors through the
[Python API](api.md#lower-level-generation); equal seeds do not imply equal inputs.

## Output

Images are PNG or TIFF. With an exact `--output`, existing files are protected unless you add
`--force`. A run's inputs, such as source images, adapters, and saved conditioning, are never
replaced, even through a symbolic or hard link.

The whole bundle of images and tensor sidecars is staged first, so an ordinary failure while
publishing restores the previous files. See [output files](../USAGE.md#output-files) for names.

## Config files

Any run can be a TOML file; see [Configuration](../CONFIG.md). This one runs base with a Turbo
LoRA and a style adapter:

```toml
model = "flux2"

[model_settings]
variant = "base"

[generate]
prompt = "A painted landscape."
negative_prompt = "blurry"
guidance = 4.0
steps = 4
seed = 42

[[generate.adapters]]
path = "klein_turbo_lora.safetensors"
strength = 1.0

[[generate.adapters]]
path = "style.safetensors"
strength = 0.8

[output]
directory = "outputs"
prefix = "painted-landscape"
save_all_sidecars = true
```

This one is a KV edit:

```toml
model = "flux2"

[model_settings]
variant = "kv"

[generate]
prompt = "Turn the ceramic vase into blue glass, preserving the window and light."
seed = 42
source_images = ["reference.jpg"]
match_source_aspect = "long"
height = 1024

[output]
directory = "outputs"
prefix = "vase-edit"
save_all_sidecars = true
```

```bash
kinomlx --config painted-landscape.toml
kinomlx --config vase-edit.toml --set generate.seed=7
kinomlx --model flux2 --variant kv --print-config --only-non-defaults
```

`kinomlx config init --model flux2` writes a template with every field and its default.
