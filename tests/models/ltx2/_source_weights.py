"""Small source-loading fixtures shared by binding and real checkpoint tests."""

from pathlib import Path
from types import SimpleNamespace

from kinomlx.io.safetensors import load_weights
from kinomlx.models.ltx2.metadata import resolve_transformer_bindings
from kinomlx.models.ltx2.preparation import prepare_transformer_source
from kinomlx.models.ltx2.resources import ComponentKind, LTX2Resources, prepare_resources
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.source_loading import family_source_weights
from kinomlx.settings import Settings

NO_LTX23_CHECKPOINT = (
    "no LTX-2.3 checkpoint: set KINO_LTX2_WEIGHTS_PATH or cache Lightricks/LTX-2.3"
)


def ltx2_resources(settings: LTX2Settings | None = None) -> LTX2Resources | None:
    """The resources the product prepares for ``settings``, or None when it would refuse.

    The product reads its selection from the environment (KINO_LTX2_WEIGHTS_PATH and
    the other KINO_LTX2 variables) and discovers whatever is unset in the local
    Hugging Face snapshots. A test that asks the same way runs wherever the product
    would run and skips wherever the product would refuse.
    """
    try:
        return prepare_resources(
            LTX2Settings.from_env() if settings is None else settings,
            infrastructure=Settings.from_env(),
        )
    except OSError, ValueError:
        return None


def ltx23_checkpoint() -> Path | None:
    """The monolithic LTX-2.3 checkpoint the product selects, or None when it selects none."""
    resources = ltx2_resources()
    if resources is None:
        return None
    try:
        path = resources.weights_path
    except LookupError:
        return None
    return path if path.is_file() else None


def family_weights(path, family):
    resources = SimpleNamespace(require=lambda _kind: SimpleNamespace(source_path=path))
    return family_source_weights(resources, ComponentKind(family))


def transformer_weights(path, *, config=None, **options):
    bindings = (
        None
        if config is None
        else {
            b.source_key: b.target_key
            for b in resolve_transformer_bindings(
                path, config, include_audio=options.get("include_audio", True)
            )
        }
    )
    defaults = {
        "include_audio": True,
        "video_ff_layout_specs": (),
        "video_ff_layout_layers": (),
        "video_attn_layout_specs": (),
        "video_attn_layout_layers": (),
    }
    defaults.update(options)
    preparation = prepare_transformer_source(path, source_bindings=bindings, **defaults)
    raw = load_weights(path)
    return {
        key: value
        for item in preparation.sources
        if item.convert
        for key, value in preparation.convert(item, raw[item.source_key]).tensors
    }
