"""Model catalog: every selectable model's name, display text, and default."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One selectable model: its name, display text, and whether it is the default."""

    name: str
    display_name: str
    help: str
    default: bool = False


MODEL_SPECS = {
    "flux2": ModelSpec(
        name="flux2",
        display_name="FLUX.2 Klein",
        help="generate Klein base, distilled, and KV still images",
    ),
    "gmnet": ModelSpec(
        name="gmnet",
        display_name="GMNet",
        help="expand one SDR still to HDR (EXR / PQ HEIC) with GMNet",
    ),
    "krea2": ModelSpec(
        name="krea2",
        display_name="Krea 2",
        help="generate Krea 2 Raw or Turbo still images",
    ),
    "ltx2": ModelSpec(
        name="ltx2",
        display_name="LTX-2",
        help="generate distilled LTX-2.3/LTX-2.5 video",
        default=True,
    ),
}

_DEFAULT_MODELS = tuple(name for name, spec in MODEL_SPECS.items() if spec.default)
if len(_DEFAULT_MODELS) != 1:
    raise RuntimeError("the model catalog must declare exactly one default model")
DEFAULT_MODEL = _DEFAULT_MODELS[0]


def model_choices() -> tuple[str, ...]:
    """Every selectable model-owned inference command."""
    return tuple(sorted(MODEL_SPECS))
