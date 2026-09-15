"""Immutable Krea 2 source, capability, and component records."""

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from kinomlx.io.identity import object_digest, payload_digest
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.tokenizers.byte_bpe import ByteBPETables
from kinomlx.tokenizers.loading import load_tokenizer_tables

from .catalog import Krea2VAEDecoder, Krea2Variant
from .metadata import (
    Krea2ModelIndex,
    Krea2SchedulerConfig,
    Krea2TextConfig,
    Krea2TransformerConfig,
    Krea2VAEConfig,
)
from .settings import OPTIMIZED_DENOISE_MODE


class Krea2ComponentKind(StrEnum):
    """Logical resources whose providers remain replaceable."""

    TOKENIZER = "tokenizer"
    TEXT_ENCODER = "text_encoder"
    TRANSFORMER = "transformer"
    VAE = "vae"


@dataclass(frozen=True)
class Krea2TensorBinding:
    """One source tensor routed to one logical weight target."""

    source_file: Path
    source_key: str
    target_key: str
    reshape_rows: int | None = None
    transform: str | None = None


@dataclass(frozen=True)
class Krea2ComponentResource:
    """One immutable logical component source and weight bindings."""

    kind: Krea2ComponentKind
    source_path: Path
    config_path: Path | None
    layout: str
    source_fingerprint: str
    graph_fingerprint: str
    dtype_policy: str
    tensor_count: int
    bindings: tuple[Krea2TensorBinding, ...] = ()
    tokenizer_json: Path | None = None
    tokenizer_config: Path | None = None

    @cached_property
    def tokenizer_tables(self) -> ByteBPETables:
        """Derive tokenizer tables once within this resource lifetime."""
        if self.tokenizer_json is None or self.tokenizer_config is None:
            raise ValueError("Krea tokenizer requires original JSON assets")
        return load_tokenizer_tables(self.tokenizer_json, self.tokenizer_config)

    @cached_property
    def tokenizer_identity(self) -> str:
        """Compute the saved-conditioning identity once, reusing the derived tables."""
        # The existing sidecar identity is semantic and path-independent.
        return "object:" + object_digest(
            {
                "schema_version": 2,
                "model": "krea2",
                "component": "tokenizer",
                "component_fingerprint": payload_digest(self.tokenizer_tables.to_payload()),
                "graph_fingerprint": self.graph_fingerprint,
                "dtype_policy": self.dtype_policy,
            }
        )

    @property
    def consumed_tensor_count(self) -> int:
        return len(self.bindings)


@dataclass(frozen=True)
class Krea2Capabilities:
    """Checkpoint-derived recipe and geometry facts for later fail-fast use."""

    variant: Krea2Variant
    is_distilled: bool
    recipe_families: tuple[str, ...]
    negative_prompt: bool
    latent_channels: int
    spatial_compression: int
    transformer_patch_size: int
    selected_text_layers: tuple[int, ...]


@dataclass(frozen=True)
class Krea2Resources:
    """Recipe-neutral Krea 2 resources with no live model arrays."""

    variant: Krea2Variant
    revision: str
    source_root: Path | None
    recipe_identity: str
    model_index: Krea2ModelIndex
    scheduler_config: Krea2SchedulerConfig
    transformer_config: Krea2TransformerConfig
    text_config: Krea2TextConfig
    vae_config: Krea2VAEConfig
    vae_decoder: Krea2VAEDecoder
    capabilities: Krea2Capabilities
    tokenizer: Krea2ComponentResource
    text_encoder: Krea2ComponentResource
    transformer: Krea2ComponentResource
    vae: Krea2ComponentResource
    check_adapter_weights: bool = True
    reference_denoise: bool = False
    hf_home: Path | None = None
    memory_policy: MLXMemoryPolicy = MLXMemoryPolicy()

    @property
    def components(self) -> tuple[Krea2ComponentResource, ...]:
        return (self.tokenizer, self.text_encoder, self.transformer, self.vae)

    def to_dict(self) -> dict[str, object]:
        """Return the compact CLI preparation receipt."""
        return {
            "variant": self.variant.value,
            "revision": self.revision,
            "source_root": None if self.source_root is None else str(self.source_root),
            "recipe_identity": self.recipe_identity,
            "denoise_mode": "reference" if self.reference_denoise else OPTIMIZED_DENOISE_MODE,
            "vae_decoder": {
                "name": self.vae_decoder.value,
                "output_scale": self.vae_decoder.output_scale,
            },
            "is_distilled": self.capabilities.is_distilled,
            "components": {
                component.kind.value: {
                    "source": str(component.source_path),
                    "source_fingerprint": component.source_fingerprint,
                    "layout": component.layout,
                    "source_tensors": component.tensor_count,
                    "consumed_tensors": component.consumed_tensor_count,
                }
                for component in self.components
            },
        }


__all__ = [
    "Krea2Capabilities",
    "Krea2ComponentKind",
    "Krea2ComponentResource",
    "Krea2Resources",
    "Krea2TensorBinding",
]
