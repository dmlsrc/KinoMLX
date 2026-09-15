"""Klein run-record selection and saved-station resource admission."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx._json import JsonObject, load_json_object
from kinomlx.config.persistence import read_configuration, rebase_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stations import station_output_defaults
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record
from kinomlx.image_restart_config import ImageRestartArtifactFields
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import inspect_safetensors, load_weights_with_metadata
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.modeling.artifacts import WeightSource
from kinomlx.restart_selection import (
    CANCELED_RUN_STATUS,
    PLANNED_IMAGE_COUNT,
    RESTARTABLE_RUN_STATUSES,
    fingerprint_key,
    resolve_restart_index,
    selected_artifact,
    source_image_count,
    unpublished_image,
)

from .catalog import Flux2Profile, profile
from .output_policies import OUTPUT_POLICIES
from .resources import Flux2Component, inspect_component_source, weight_component
from .types import Flux2RestartConfig


@config_record
@dataclass(frozen=True)
class Flux2Restart(ImageRestartArtifactFields):
    """Public selection of a saved conditioning or final-latent station."""


@dataclass(frozen=True)
class PreparedResources:
    """Only the validated components needed after a saved station."""

    profile: Flux2Profile
    components: tuple[Flux2Component, ...]
    check_adapter_weights: bool = True
    memory_policy: MLXMemoryPolicy = MLXMemoryPolicy()

    def component(self, kind: str) -> Flux2Component:
        return next(component for component in self.components if component.kind == kind)


@dataclass(frozen=True)
class RestartManifest:
    selection: Flux2RestartConfig
    payload: JsonObject

    @classmethod
    def load(cls, selection: Flux2RestartConfig) -> RestartManifest:
        validate_record(selection, location="[restart]")
        payload = load_json_object(selection.run)
        read_configuration(payload, model="flux2", source=str(selection.run))
        if payload.get("model") != "flux2" or payload.get("status") not in RESTARTABLE_RUN_STATUSES:
            raise ValueError("restart requires a completed or canceled KinoMLX Klein run record")
        if not isinstance(payload.get("generation"), dict):
            raise ValueError("Klein restart lacks generation metadata")
        manifest = cls(selection, payload)
        if payload.get("status") == CANCELED_RUN_STATUS:
            manifest.require_published_image()
        return manifest

    def require_published_image(self) -> None:
        """Refuse an image a canceled run stopped before publishing.

        A run publishes its images in order, so the artifacts it recorded are
        exactly the ones it finished; anything past them was planned and never
        written. A run that published nothing is refused the same way.
        """
        name = "final_latents" if self.selection.phase == "decode" else "text_conditioning"
        outputs = self.payload.get("outputs")
        entries = outputs.get(name) if isinstance(outputs, dict) else None
        published = len(entries) if isinstance(entries, list) else 0
        if self.source_index >= published:
            raise ValueError(unpublished_image(self.source_index, self.source_image_count))

    def base_config(self) -> PreparedConfig:
        intent = read_configuration(self.payload, model="flux2", source=str(self.selection.run))
        config = dict(intent.values)
        config.pop("restart", None)
        config["operation"] = {"dry_run": False}
        # A replay is a new output bundle beside its source run; it never
        # inherits replacement authority or the source's own name.
        output = dict(cast(dict[str, object], config.get("output", {})))
        source_stem = self.selection.run.stem.removesuffix("_run")
        output.update(
            path=None,
            directory=self.selection.run.parent,
            prefix=(
                f"{source_stem}_final_decode"
                if self.selection.phase == "decode"
                else f"{source_stem}_denoise"
            ),
            force=False,
        )
        config["output"] = station_output_defaults(
            output, OUTPUT_POLICIES, station=self.selection.phase
        )
        generate = dict(cast(dict[str, object], config.get("generate", {})))
        generate.update(self._selected_image())
        config["generate"] = generate
        return rebase_configuration(intent, config)

    def _selected_image(self) -> dict[str, object]:
        """Narrow a multi-image source run to the one image this restart resumes.

        The recorded seed list would otherwise re-expand the inherited prompt
        batch, so prompt and seed collapse to the selected row.
        """
        index = self.source_index
        generation = self.payload.get("generation")
        images = generation.get("images") if isinstance(generation, dict) else None
        entry = images[index] if isinstance(images, list) and index < len(images) else None
        if not isinstance(entry, dict):
            raise ValueError(
                f"Klein run records no image {index + 1}; restart from a run whose "
                "generation metadata lists every image"
            )
        selected: dict[str, object] = {"images_per_prompt": 1, "seeds": None}
        seed = entry.get("seed")
        prompt = entry.get("prompt")
        negative = entry.get("negative_prompt")
        if type(seed) is not int or not isinstance(prompt, str) or not isinstance(negative, str):
            raise ValueError(f"Klein image {index + 1} requires recorded seed and prompt rows")
        selected.update(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative,
            prompt_unconditional=prompt == "",
        )
        return selected

    @property
    def source_image_count(self) -> int:
        outputs = self.payload.get("outputs")
        return source_image_count(
            outputs if isinstance(outputs, dict) else {},
            planned=self.payload.get(PLANNED_IMAGE_COUNT),
        )

    @property
    def source_index(self) -> int:
        """Zero-based output index this restart resumes."""
        return resolve_restart_index(self.selection.index, self.source_image_count)

    def artifact(self, name: str, replacement: Path | None = None) -> Path:
        if replacement is not None:
            return replacement.expanduser().absolute()
        outputs = self.payload.get("outputs")
        fingerprints = self.payload.get("output_fingerprints")
        count = self.source_image_count
        index = self.source_index
        value = (
            selected_artifact(outputs.get(name), index=index, label=f"outputs.{name}")
            if isinstance(outputs, dict)
            else None
        )
        digest = (
            fingerprints.get(fingerprint_key(name, index=index, count=count))
            if isinstance(fingerprints, dict)
            else None
        )
        if not isinstance(value, str) or not isinstance(digest, str):
            raise ValueError(f"Klein run has no fingerprinted {name} station")
        path = Path(value)
        if not path.is_absolute():
            path = self.selection.run.parent / path
        if file_sha256(path) != digest:
            raise ValueError(f"Klein saved {name} fingerprint mismatch")
        return path

    def resources(
        self,
        variant: str,
        *,
        phase: str,
        check_adapter_weights: bool = True,
    ) -> PreparedResources:
        """Reopen and validate the recorded sources needed by the saved station."""
        generation = cast(JsonObject, self.payload["generation"])
        resource = generation.get("resources")
        if not isinstance(resource, dict) or resource.get("variant") != variant:
            raise ValueError("Klein restart cannot change checkpoint variant")
        entries = resource.get("components")
        if not isinstance(entries, list):
            raise ValueError("Klein run lacks prepared component receipts")
        required = {"vae"} if phase == "decode" else {"transformer", "vae"}
        components: list[Flux2Component] = []
        for item in entries:
            if not isinstance(item, dict) or item.get("kind") not in required:
                continue
            kind = cast(str, item["kind"])
            source_path, dtype = item.get("source"), item.get("dtype")
            if (
                not isinstance(source_path, str)
                or dtype not in {"float16", "bfloat16"}
                or (kind == "vae" and dtype != "bfloat16")
            ):
                raise ValueError("Invalid Klein component receipt")
            source = inspect_component_source(kind, Path(source_path))
            assert isinstance(source, WeightSource)
            component = weight_component(
                kind,
                source,
                dtype=cast(str, dtype),
            )
            if any(
                item.get(key) != expected
                for key, expected in (
                    ("source_identity", component.source.source_fingerprint),
                    ("graph_identity", component.graph_identity),
                    ("identity", component.identity),
                )
            ):
                raise ValueError(f"Klein saved {kind} source identity mismatch")
            components.append(component)
        if {component.kind for component in components} != required or len(components) != len(
            required
        ):
            raise ValueError("Klein restart lacks required prepared components")
        return PreparedResources(profile(variant), tuple(components), check_adapter_weights)


def load_final_latents(path: Path) -> tuple[mx.array, int, int]:
    """Read a model-tagged final station; numerical validation happens before VAE load."""
    inventory = inspect_safetensors(path)
    metadata = dict(inventory.metadata)
    if (
        metadata.get("schema") != "kinomlx.flux2.tensor.v1"
        or metadata.get("artifact") != "final_latents"
        or inventory.tensor_names != ("final_latents",)
    ):
        raise ValueError("Expected a KinoMLX Klein final-latent artifact")
    try:
        width, height = int(metadata["width"]), int(metadata["height"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Klein final latents lack an output grid") from exc
    tensors, _ = load_weights_with_metadata(path)
    return tensors["final_latents"], width, height
