"""Generate a Klein still through the typed API and transactional output sink."""

import argparse
from pathlib import Path

from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.models.flux2 import (
    Flux2OutputSink,
    Flux2Request,
    Flux2Runner,
    Flux2Settings,
    plan_output,
)
from kinomlx.settings import Settings


def generate(prompt: str, output: Path, *, variant: str = "distilled", seed: int = 42) -> None:
    """Reserve output before preparing or loading any model."""
    plan = plan_output(
        ImageOutputConfig(path=output, save_final_latents=True),
        variant=variant,
        seed=seed,
        image_count=1,
    )
    request = Flux2Request(prompt=prompt, seed=seed, width=512, height=512)
    recipe = request.recipe(variant)
    with plan.reserve() as reservation:
        runner = Flux2Runner(
            Flux2Settings(variant=variant),
            infrastructure=Settings.from_env(),
        )
        result = runner.generate(request)
        Flux2OutputSink(plan).write(list(result.images), recipe, reservation=reservation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "distilled", "kv"), default="distilled")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.prompt, args.output, variant=args.variant, seed=args.seed)


if __name__ == "__main__":
    main()
