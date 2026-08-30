#!/usr/bin/env python3
"""Visual demo for ``kinomlx.ui`` - logging levels + stacked progress bars.

Run a scenario to eyeball that the rich-backed UI looks right. Unit tests pin the
formulas and config, but they can't catch "ugly" or "messages clobber the bars" -
that's what this script is for.

Usage::

    python scripts/demo_ui.py                          # list scenarios
    python scripts/demo_ui.py --scenario all           # run every scenario
    python scripts/demo_ui.py --scenario distilled     # one scenario
    python scripts/demo_ui.py --scenario messages -v   # show DEBUG output too
    python scripts/demo_ui.py --scenario all --log-file run.log   # tee a sidecar

Scenarios are shaped like KinoMLX's real workloads, so the demo doubles as a
sanity check after touching ``kinomlx.ui`` internals (theme, levels, columns,
throttle, completion summaries).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable

from kinomlx.ui import configure_logging, make_progress, track_phase

# Two subsystem loggers, so the demo shows the per-subsystem name in a sidecar
# file and that levels are filterable per subsystem (logger name = subsystem).
_log = logging.getLogger("kinomlx.demo")
_loader = logging.getLogger("kinomlx.demo.loader")

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def demo_messages() -> None:
    """info / debug / warning / error rendering - the output a real run produces."""
    _loader.info("loading transformer weights from cache")
    _loader.debug("dequantized fp8 block 12 -> bf16 (hidden at INFO; shows with -v)")
    _log.info("48 transformer blocks resident; fast mode ON")
    _log.warning("LoRA shape mismatch for transformer.block_0.attn.to_q.weight; skipping target")
    _log.error("failed to allocate 8 GiB; consider --transformer-resident-blocks 4")


def demo_single_bar() -> None:
    """One slow bar - confirm columns line up, pace formats ``s/frame``, summary logs."""
    _log.info("synthesizing 5 latent frames")
    with make_progress() as prog, track_phase(prog, "VAE decode", total=5, unit="frame") as task:
        for _ in range(5):
            time.sleep(1.2)
            prog.advance(task)


def demo_distilled() -> None:
    """Two-stage distilled denoise - stage 1 (8 steps), status between, stage 2 (3 steps).

    Mirrors KinoMLX's actual shape: a longer stage finishes (and logs its
    completion summary), two status messages land above the bar, then the second
    stage joins the table.
    """
    _log.info("starting distilled two-stage denoise (8 + 3 steps)")
    with make_progress() as prog:
        with track_phase(prog, "Stage 1 denoise", total=8, unit="step") as s1:
            for _ in range(8):
                time.sleep(1.0)
                prog.advance(s1)
        _log.info("stage 1 complete; upsampling latent 2x with spatial upscaler")
        _log.info("stage 2: 3 steps at 192x192")
        with track_phase(prog, "Stage 2 denoise", total=3, unit="step") as s2:
            for _ in range(3):
                time.sleep(1.0)
                prog.advance(s2)
    _log.info("done")


def demo_stacked() -> None:
    """Two bars ticking together (VAE chunks slow, VSR frames fast).

    Confirms the inner table aligns columns across rows and the pace column
    gracefully handles two magnitudes (``X.X s/chunk`` next to ``XX.X frame/s``).
    Kept on the raw ``add_task`` API since the two bars interleave.
    """
    _log.info("encoding a 4-chunk video through VAE -> VSR")
    chunks = 4
    frames_per_chunk = 20
    with make_progress() as prog:
        vae = prog.add_task("VAE chunks", total=chunks, unit="chunk")
        vsr = prog.add_task("VSR frames", total=chunks * frames_per_chunk, unit="frame")
        for _ in range(chunks):
            time.sleep(1.5)
            prog.advance(vae)
            for _ in range(frames_per_chunk):
                time.sleep(0.08)
                prog.advance(vsr)


def demo_throttle() -> None:
    """1000 ticks in ~2s - verify the 1 Hz refresh throttle does its job.

    macOS hardware-accelerates the terminal; each redraw costs Terminal +
    WindowServer GPU. At the default ``refresh_per_second=1.0`` this loop should
    redraw a handful of times, not 1000 - visible as the percentage jumping in
    big steps rather than smoothly counting up.
    """
    _log.info("ticking 1000 frames in ~2s (throttled to 1 redraw/sec)")
    with make_progress() as prog, track_phase(prog, "hot loop", total=1000, unit="frame") as task:
        for _ in range(1000):
            time.sleep(0.002)
            prog.advance(task)
    _log.info("done - percentage should have advanced in big steps, not smoothly")


SCENARIOS: dict[str, Callable[[], None]] = {
    "messages": demo_messages,
    "single": demo_single_bar,
    "stacked": demo_stacked,
    "distilled": demo_distilled,
    "throttle": demo_throttle,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def list_scenarios() -> None:
    width = max(len(name) for name in SCENARIOS)
    rows = [
        f"  {name:<{width}}  {(fn.__doc__ or '').strip().splitlines()[0]}"
        for name, fn in SCENARIOS.items()
    ]
    _log.info(
        "Available --scenario values:\n\n%s\n\n"
        "Or --scenario all to run every scenario back-to-back.",
        "\n".join(rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="scenario name (omit or pass 'list' for the catalog; 'all' runs each in turn)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for DEBUG output")
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    parser.add_argument(
        "--log-file", default=None, help="also tee a DEBUG sidecar log to this path"
    )
    parser.add_argument("--show-date", action="store_true", help="prefix the date on each line")
    args = parser.parse_args()

    configure_logging(
        verbosity=args.verbose,
        quiet=args.quiet,
        show_date=args.show_date,
        log_file=args.log_file,
    )

    if args.scenario is None or args.scenario == "list":
        list_scenarios()
        return 0
    if args.scenario == "all":
        for name, fn in SCENARIOS.items():
            _log.info("--- %s ---", name)
            fn()
        return 0
    if args.scenario in SCENARIOS:
        SCENARIOS[args.scenario]()
        return 0

    _log.error("Unknown scenario: %r", args.scenario)
    list_scenarios()
    return 1


if __name__ == "__main__":
    sys.exit(main())
