"""Public stateless GMNet SDR-to-HDR recipe."""

from __future__ import annotations

from kinomlx.reporting import NullReporter, Reporter

from .components import GMNetComponents, NativeGMNetComponents
from .expand import ExpansionResult, expand_image
from .resources import GMNetResources
from .types import GMNetRequest


def _validate_request(request: GMNetRequest) -> None:
    source = request.image
    if not source.is_file():
        raise FileNotFoundError(f"no such image: {source}")
    if source.suffix.lower() == ".exr":
        raise ValueError(
            f"expansion input must be a display-referred SDR image; {source} is an EXR"
        )


def expand_gmnet(
    request: GMNetRequest,
    resources: GMNetResources,
    *,
    components: GMNetComponents | None = None,
    reporter: Reporter | None = None,
) -> ExpansionResult:
    """Expand one SDR still using bounded model ownership and native ImageIO."""
    _validate_request(request)
    sink = reporter if reporter is not None else NullReporter()
    provider = components if components is not None else NativeGMNetComponents(reporter=sink)

    from kinomlx.io.image import load_image

    load_phase = "load SDR image"
    sink.phase_start(load_phase, total=1, unit="image")
    try:
        image = load_image(request.image)
        sink.phase_advance(load_phase)
    finally:
        sink.phase_end(load_phase)

    inference_phase = "GMNet expansion"
    sink.phase_start(inference_phase, total=1, unit="image")
    try:
        with provider.generator(resources) as model:
            result = expand_image(model, image, resources.spec)
        sink.phase_advance(inference_phase)
        return result
    finally:
        sink.phase_end(inference_phase)


__all__ = ["expand_gmnet"]
