"""GMNet - learned gain-map prediction for SDR-to-HDR inverse tone mapping.

An independent MLX reimplementation of the GMNet architecture from
"Learning Gain Map for Inverse Tone Mapping" (Liao et al., ICLR 2025).
The network predicts a normalized gain map plus a Qmax scalar over a
display-referred SDR still; the package reconstructs scene-linear HDR
from them and hands the result to the native EXR / PQ-HEIC terminals.

Self-contained per the multi-model convention: no imports from other
model packages. Selected from the main CLI as ``kinomlx --model gmnet``.
"""

from .catalog import (
    GMNetVariant,
    GMNetVariantSpec,
    resolve_variant_weights,
    variant_spec,
)
from .components import GMNetComponents, GMNetComponentsProvider
from .expand import (
    ExpansionResult,
    expand_image,
    reconstruct_linear_hdr,
    write_gain_map_sidecar,
)
from .net import GMNet, load_gmnet_weights
from .output import (
    GMNetArtifactSet,
    GMNetOutputError,
    GMNetOutputPlan,
    GMNetOutputReservation,
    GMNetOutputSink,
    plan_gmnet_output,
    write_gmnet_output,
)
from .pipelines import expand_gmnet
from .resources import GMNetResources, prepare_resources
from .runner import GMNetError, GMNetRunner
from .settings import GMNetSettings
from .types import GMNetOutputConfig, GMNetRequest

__all__ = [
    "ExpansionResult",
    "GMNet",
    "GMNetArtifactSet",
    "GMNetComponents",
    "GMNetComponentsProvider",
    "GMNetError",
    "GMNetOutputConfig",
    "GMNetOutputError",
    "GMNetOutputPlan",
    "GMNetOutputReservation",
    "GMNetOutputSink",
    "GMNetRequest",
    "GMNetResources",
    "GMNetRunner",
    "GMNetSettings",
    "GMNetVariant",
    "GMNetVariantSpec",
    "expand_gmnet",
    "expand_image",
    "load_gmnet_weights",
    "plan_gmnet_output",
    "prepare_resources",
    "reconstruct_linear_hdr",
    "resolve_variant_weights",
    "variant_spec",
    "write_gain_map_sidecar",
    "write_gmnet_output",
]
