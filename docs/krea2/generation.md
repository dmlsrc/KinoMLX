# Generating images

This page covers text-to-image choices: the recipe, size, prompts, schedules, and the optional 2x
decoder. Every argument it mentions is in [the catalog](arguments.md).

- [Recipes](#recipes)
- [Size](#size)
- [Prompts and batches](#prompts-and-batches)
- [Prompt length](#prompt-length)
- [System instruction](#system-instruction)
- [Set the schedule yourself](#set-the-schedule-yourself)
- [Decode at twice the size](#decode-at-twice-the-size)
- [Config files](#config-files)

## Recipes

Turbo, the default, takes eight steps with no guidance and a fixed scheduler shift of 1.15:

```bash
kinomlx --model krea2 \
  --prompt "A quiet coastal sunrise, cinematic light" \
  --width 1024 --height 1024 --seed 42
```

Raw takes 52 steps with guidance 3.5 and a shift that follows the resolution. Its negative branch
uses an empty prompt unless you give one:

```bash
kinomlx --model krea2 --variant raw \
  --prompt "A quiet coastal sunrise, cinematic light" --negative-prompt "low contrast" \
  --width 1024 --height 1024 --seed 42
```

`--steps`, `--guidance`, and `--mu` override the recipe. Krea's guidance is an offset,
`conditional + guidance * (conditional - unconditional)`, so guidance 2 equals conventional CFG 3.

Guidance 0 skips the negative branch entirely, halving the work per step. A nonempty negative
prompt is then refused, since nothing would read it.

`--noise-backend torch-mps` reproduces PyTorch's MPS noise from the same seed, to compare against
tools built on PyTorch. It changes the seeded sample, not the model.

## Size

`--width` and `--height` default to 1024. Generation rounds them up to a multiple of 16 and
crops the result back to exactly the size you asked for.

## Prompts and batches

Repeat `--prompt` to run several prompts in order. One `--negative-prompt` applies to all of
them; to give each its own, repeat it once per prompt. `--images-per-prompt N` makes N images of
each prompt with consecutive seeds; see [seeds](../USAGE.md#seeds).

Each image runs through the denoiser on its own, which needs less memory than a batch and is no
slower.

## Prompt length

The text encoder holds 507 tokens of prompt, roughly a few hundred words. Longer prompts are cut
off, even though the whole file is read, so put the important instructions first. Prompts over
65,536 characters are refused before tokenizing.

## System instruction

`--system-prompt` replaces the instruction Krea gives its Qwen text encoder before your prompt:

```bash
kinomlx --model krea2 --system-prompt "Describe the visible objects and lighting." \
  --prompt "A ceramic cup on a walnut table"
```

Omit it to keep Krea's training instruction; `--system-prompt ""` sends an empty one, and a custom
instruction keeps its whitespace. Treat custom instructions as experiments: the model was trained
with its own.

An instruction may be up to 65,536 characters, and with the prompt it must fit the text encoder's
262,144-token context. A longer instruction makes text encoding slower but does not shorten the
prompt's budget. In TOML it is `[generate].system_prompt`, and in Python
`Krea2Request(system_prompt=...)`.

The option applies to text-to-image only. Editing has its own `--grounding-system-prompt`; see
[editing](editing.md#describe-the-source-differently). Saved text conditioning records the
instruction, so reusing it requires the same one.

## Set the schedule yourself

Both variants derive their noise levels by shifting a uniform 1-to-0 ramp. `--sigmas` gives the
levels directly:

```bash
kinomlx --model krea2 --prompt "A quiet coastal sunrise, cinematic light" \
  --sigmas "1,0.9567,0.9045,0.8403,0.7595,0.6547,0.5128,0.3109,0"
```

The list holds one more value than the steps it drives, so nine values run eight steps whatever
`--steps` says, and `--mu` is then unused. Values must be finite, between 0 and 1, and strictly
decreasing; a last value above zero stops early and leaves some noise.

Every run reports its schedule in the sampling section and records it as `denoise.sigmas`. See
[Denoise schedules](../SCHEDULES.md).

## Decode at twice the size

`--vae-decoder upscale2x` decodes with a community decoder that outputs twice the width and
height, so a 1024x1024 generation saves a 2048x2048 image. It reads Krea's latents directly,
because Wan 2.1 and Krea share one latent format. Download it first; see
[optional files](downloads.md#optional-files).

```bash
kinomlx --model krea2 --vae-decoder upscale2x \
  --prompt "A quiet coastal sunrise, cinematic light" --width 1024 --height 1024
```

Treat it like an aesthetic adapter. It was trained for perceptual quality on real photographs and
can improve skin, hair, and texture, but it can also shift color, oversharpen, or invent texture.
Its model card warns that anime, line art, and text may suffer. The stock `qwen` decoder stays the
default.

Passing the file as `--vae-path` selects it too, without `--vae-decoder`. Decoder tiling accounts
for the larger output, and custom tile sizes still describe the image before doubling.

## Config files

Any run can be a TOML file; see [Configuration](../CONFIG.md). This one is a Turbo run with the
four-step adapter and a style adapter:

```toml
model = "krea2"

[model_settings]
variant = "turbo"

[generate]
prompt = "A painted landscape."
width = 1024
height = 1024
seed = 42
turbo_4step = true

[[generate.adapters]]
path = "style.safetensors"
strength = 0.8

[output]
directory = "outputs"
prefix = "painted-landscape"
save_all_sidecars = true
```

```bash
kinomlx --config painted-landscape.toml
kinomlx --config painted-landscape.toml --set generate.seed=7
```

`kinomlx config init --model krea2` writes a template with every field and its default.
