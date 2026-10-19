"""Immutable, header-validated Klein resources with explicit source accounting."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from kinomlx._json import JsonObject
from kinomlx.huggingface import resolve_cached_snapshot
from kinomlx.image_restart_config import IMAGE_STAGES, ImageStage
from kinomlx.io.identity import payload_digest
from kinomlx.io.safetensors import inspect_safetensors
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.modeling.artifacts import (
    TokenizerSource,
    WeightSource,
    inspect_tokenizer_source,
    inspect_weight_source,
)
from kinomlx.modeling.dtypes import DTypePolicy, FloatDType
from kinomlx.settings import Settings
from kinomlx.tokenizers.byte_bpe import QWEN_TOKENIZER_TABLE_POLICY, ByteBPETables
from kinomlx.tokenizers.loading import load_tokenizer_tables

from .catalog import Flux2Profile, Flux2Variant, profile
from .layout import weight_layout
from .metadata import SCHEDULER, TEXT_TAPS, validate_config, validate_model_index
from .settings import Flux2Settings
from .shapes import text_shapes, transformer_shapes, vae_shapes

KINDS = ("tokenizer", "text_encoder", "transformer", "vae")
# The prepared components each stage needs. A restart reopens those of the
# stages it runs; a normal run reopens every kind.
STAGE_COMPONENTS: Mapping[str, tuple[str, ...]] = frozendict(
    {
        ImageStage.TEXT: ("tokenizer", "text_encoder"),
        ImageStage.DENOISE: ("transformer",),
        ImageStage.DECODE: ("vae",),
    }
)


def restart_components(stage: str) -> frozenset[str]:
    """The prepared component kinds a restart at ``stage`` reopens."""
    running = IMAGE_STAGES[IMAGE_STAGES.index(stage) :]
    return frozenset(*STAGE_COMPONENTS[stage] for stage in running)


@dataclass(frozen=True)
class Flux2Binding:
    source_file: Path
    source_key: str
    shape: tuple[int, ...]
    use: str


@dataclass(frozen=True)
class Flux2Component:
    """Validated source tensors and their lazy preparation policy."""

    kind: str
    source: WeightSource | TokenizerSource
    graph_identity: str
    identity: str
    bindings: tuple[Flux2Binding, ...] = ()
    dtype_policy: DTypePolicy = FloatDType.BFLOAT16

    @cached_property
    def tokenizer_tables(self) -> ByteBPETables:
        """Derive once per resource lifetime, without retaining model arrays."""
        if not isinstance(self.source, TokenizerSource):
            raise ValueError("Klein tokenizer requires a tokenizer source")
        tables = load_tokenizer_tables(self.source.tokenizer_json, self.source.tokenizer_config)
        if self.source.maximum_token_id >= 151936 or tables.pad_token_id != 151643:
            raise ValueError("Klein tokenizer vocabulary/padding contradicts Qwen3 embeddings")
        return tables

    @property
    def active_bindings(self) -> tuple[Flux2Binding, ...]:
        return tuple(b for b in self.bindings if b.use == "runtime")


@dataclass(frozen=True)
class Flux2Resources:
    profile: Flux2Profile
    source_root: Path | None
    revision: str
    recipe_identity: str
    components: tuple[Flux2Component, ...]
    check_adapter_weights: bool = True
    memory_policy: MLXMemoryPolicy = MLXMemoryPolicy()

    def component(self, kind: str) -> Flux2Component:
        return next(c for c in self.components if c.kind == kind)

    def describe(self) -> JsonObject:
        return {
            "model": "flux2",
            "variant": self.profile.variant.value,
            "revision": self.revision,
            "recipe_identity": self.recipe_identity,
            "steps": self.profile.steps,
            "guidance": self.profile.guidance,
            "reference_kv": self.profile.kv,
            "check_adapter_weights": self.check_adapter_weights,
            "components": [_describe_component(c) for c in self.components],
        }


def _describe_component(component: Flux2Component) -> JsonObject:
    entry: JsonObject = {
        "kind": component.kind,
        "dtype": component.dtype_policy,
        "source": str(component.source.source_path),
        "source_identity": component.source.source_fingerprint,
        "graph_identity": component.graph_identity,
        "identity": component.identity,
        "source_tensors": len(component.bindings),
        "runtime_tensors": len(component.active_bindings),
        "excluded_tensors": len(component.bindings) - len(component.active_bindings),
    }
    return entry


def _validate_weights(kind: str, source: WeightSource) -> tuple[Flux2Binding, ...]:
    expected = {"text_encoder": text_shapes, "transformer": transformer_shapes, "vae": vae_shapes}[
        kind
    ]()
    actual = source.tensor_shapes
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(k for k in actual.keys() & expected.keys() if actual[k] != expected[k])
        raise ValueError(
            f"Klein {kind} tensor contract: missing={missing[:3]}, extra={extra[:3]}, wrong_shape={wrong[:3]}"
        )
    bindings = []
    for file in source.tensor_files:
        for tensor in inspect_safetensors(file.path).tensors:
            expected_dtype = "I64" if tensor.name == "bn.num_batches_tracked" else "BF16"
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Klein {kind} {tensor.name} requires {expected_dtype}, got {tensor.dtype}"
                )
            use = "runtime"
            if kind == "text_encoder":
                if tensor.name in {"lm_head.weight", "model.norm.weight"}:
                    use = "unused-language-output"
                elif (
                    tensor.name.startswith("model.layers.")
                    and int(tensor.name.split(".")[2]) >= TEXT_TAPS[-1]
                ):
                    use = "after-final-hidden-tap"
            if tensor.name == "bn.num_batches_tracked":
                use = "training-counter"
            bindings.append(Flux2Binding(file.path, tensor.name, tensor.shape, use))
    return tuple(bindings)


def weight_component(kind: str, source: WeightSource, *, dtype: FloatDType) -> Flux2Component:
    """Validate source provenance and preparation policy independently of persistence."""
    graph = validate_config(kind, source.config_path)
    bindings = _validate_weights(kind, source)
    identity = payload_digest(
        {
            "kind": kind,
            "source": source.source_fingerprint,
            "graph": graph,
            "layout": weight_layout(kind, dtype),
            "text_taps": list(TEXT_TAPS) if kind == "text_encoder" else [],
        }
    )
    return Flux2Component(kind, source, graph, identity, bindings, dtype)


def _component_dtype(kind: str, settings: Flux2Settings) -> FloatDType:
    """Select the dtype one component binds at; only two are user-selectable."""
    if kind == "transformer":
        return settings.transformer_dtype
    if kind == "vae":
        return settings.vae_dtype
    return FloatDType.BFLOAT16


def prepare_resources(settings: Flux2Settings, *, infrastructure: Settings) -> Flux2Resources:
    """Validate the source components without loading tensor payloads."""
    settings.validate()
    infrastructure.validate()
    selected = profile(settings.variant)
    overrides = settings.component_overrides
    root = settings.source_path.expanduser().absolute() if settings.source_path else None
    revision = "local-components"
    # Inspect explicit overrides first; a missing baseline must not hide an invalid override.
    sources: dict[str, WeightSource | TokenizerSource] = {}
    tokenizers: dict[str, Flux2Component] = {}
    for kind, path in overrides.items():
        sources[kind] = inspect_component_source(kind, path)
        if isinstance(sources[kind], TokenizerSource):
            tokenizer = sources[kind]
            assert isinstance(tokenizer, TokenizerSource)
            tokenizers[kind] = tokenizer_component(tokenizer)
        if isinstance(sources[kind], WeightSource):
            source = sources[kind]
            assert isinstance(source, WeightSource)
            validate_config(kind, source.config_path)
            _validate_weights(kind, source)
    if root is None and set(overrides) != set(KINDS):
        snapshot = resolve_cached_snapshot(infrastructure.hf_home, selected.repo_id)
        root, revision = snapshot.path, snapshot.revision
    elif root is not None:
        revision = root.name
    if root is not None:
        validate_model_index(
            root / "model_index.json", is_distilled=selected.variant is not Flux2Variant.BASE
        )
        validate_config("scheduler", root / "scheduler/scheduler_config.json")
    components = []
    for kind in KINDS:
        dtype = _component_dtype(kind, settings)
        if kind not in sources:
            assert root is not None
            sources[kind] = inspect_component_source(kind, root / kind)
        source = sources[kind]
        if isinstance(source, TokenizerSource):
            components.append(
                tokenizers[kind] if kind in tokenizers else tokenizer_component(source)
            )
        else:
            components.append(weight_component(kind, source, dtype=dtype))
    recipe = payload_digest(
        {
            "variant": selected.variant.value,
            "scheduler": SCHEDULER,
            "transformer": components[2].identity,
        }
    )
    return Flux2Resources(
        selected,
        root,
        revision,
        recipe,
        tuple(components),
        check_adapter_weights=infrastructure.check_adapter_weights,
        memory_policy=MLXMemoryPolicy.from_settings(infrastructure),
    )


def tokenizer_component(source: TokenizerSource) -> Flux2Component:
    """Validate tokenizer semantics once and identify the source plus derivation policy."""
    graph = QWEN_TOKENIZER_TABLE_POLICY
    identity = payload_digest(
        {"kind": "tokenizer", "graph": graph, "source": source.source_fingerprint}
    )
    component = Flux2Component("tokenizer", source, graph, identity)
    _ = component.tokenizer_tables
    return component


def inspect_component_source(kind: str, path: Path) -> WeightSource | TokenizerSource:
    if kind == "tokenizer":
        return inspect_tokenizer_source(path)
    return inspect_weight_source(
        path, filenames=("model.safetensors", "diffusion_pytorch_model.safetensors")
    )
