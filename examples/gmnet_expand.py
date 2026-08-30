"""Public GMNet recipe plus transactional still-output composition."""

from __future__ import annotations

from pathlib import Path

from kinomlx import (
    GMNetArtifactSet,
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    GMNetSettings,
    Settings,
    expand_gmnet,
    plan_gmnet_output,
    prepare_gmnet_resources,
)


def expand_still(
    source: Path,
    *,
    output: Path | None = None,
    output_directory: Path = Path("outputs"),
) -> GMNetArtifactSet:
    """Expand one SDR still and publish its selected HDR artifacts."""
    infrastructure = Settings.from_env()
    resources = prepare_gmnet_resources(
        GMNetSettings.from_env(),
        infrastructure=infrastructure,
    )
    request = GMNetRequest(image=source)
    plan = plan_gmnet_output(
        request,
        GMNetOutputConfig(path=output, directory=output_directory),
    )
    runner = GMNetRunner(resources=resources)
    with plan.reserve() as reservation:
        result = runner.run(expand_gmnet, request)
        return GMNetOutputSink(plan).write(result, reservation=reservation)


__all__ = ["expand_still"]
