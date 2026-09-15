"""Public GMNet recipe plus transactional still-output composition."""

from pathlib import Path

from kinomlx import Settings
from kinomlx.models.gmnet import (
    GMNetArtifactSet,
    GMNetOutputConfig,
    GMNetOutputSink,
    GMNetRequest,
    GMNetRunner,
    GMNetSettings,
    expand_gmnet,
    plan_gmnet_output,
    prepare_resources,
)


def expand_still(
    source: Path,
    *,
    output: Path | None = None,
    output_directory: Path = Path("outputs"),
) -> GMNetArtifactSet:
    """Expand one SDR still and publish its selected HDR artifacts."""
    # Both MLX limits travel with resources. The runner applies them once before
    # model work, including when these resources are reused for another call.
    infrastructure = Settings.from_env()
    resources = prepare_resources(
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
