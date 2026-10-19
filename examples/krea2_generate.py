"""Generate Raw or Turbo stills through the public Krea 2 API.

Download the model components described in docs/krea2/downloads.md before running this
example. Resource preparation discovers local sources and captures execution
policy. The ordinary variant defaults supply the step count and guidance. Output
paths must be unused PNG or TIFF files.
"""

# Runners apply the resource's memory policy before each invocation. Set
# mlx_cache_limit_gb and mlx_wired_limit_gb on infrastructure Settings.

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from kinomlx import Settings
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.models.krea2 import (
    Krea2OutputSink,
    Krea2Request,
    Krea2Runner,
    Krea2Settings,
    plan_krea2_output,
)
from kinomlx.reporting import Reporter


def generate_images(
    request: Krea2Request,
    output: Path,
    *,
    model_settings: Krea2Settings | None = None,
    infrastructure: Settings | None = None,
    reporter: Reporter | None = None,
) -> tuple[Path, ...]:
    """Prepare local resources and publish images before closing their owner.

    Hosts can pass a Reporter for progress, a batch request for multiple images,
    or request.text_conditioning to reuse a saved text station. The output sink
    owns display conversion and transactional image publication.
    """
    runner = Krea2Runner(model_settings, infrastructure=infrastructure, reporter=reporter)
    plan = plan_krea2_output(
        request,
        runner.resources.variant,
        ImageOutputConfig(path=output),
    )
    # Reserve every target before inference; existing files are never replaced.
    with plan.reserve() as reservation:
        with runner.generate(request) as result:
            artifacts = Krea2OutputSink(plan).write(result, reservation=reservation)
        reservation.complete()
    return artifacts.images


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public recipe with variant defaults and an explicit output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("raw", "turbo"))
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args(argv)
    settings = Krea2Settings.from_env()
    if options.variant is not None:
        settings = replace(settings, variant=options.variant)
    if options.source_path is not None:
        settings = replace(settings, source_path=options.source_path)
    generate_images(
        Krea2Request(
            options.prompt,
            width=options.width,
            height=options.height,
            seed=options.seed,
        ),
        options.output,
        model_settings=settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
