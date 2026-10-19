"""Evaluated Klein stage data; reference KV is deliberately never serialized."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import inspect_safetensors, load_weights_with_metadata, save_weights

from .image_processing import ReferenceGeometry, reference_geometry, spatial_ids
from .sampling import SamplingRecipe

TEXT_CONDITIONING_SCHEMA_VERSION = "1"
TEXT_CONDITIONING_ARTIFACT = "flux2-text-conditioning"


@dataclass(frozen=True)
class Flux2Conditioning:
    """Complete batch-one text/reference boundary before transformer loading."""

    text: mx.array
    text_ids: mx.array
    negative: mx.array | None = None
    references: mx.array | None = None
    reference_ids: mx.array | None = None
    geometries: tuple[ReferenceGeometry, ...] = ()
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024

    def tensors(self) -> dict[str, mx.array]:
        return {
            key: value
            for key, value in (
                ("text", self.text),
                ("text_ids", self.text_ids),
                ("negative", self.negative),
                ("references", self.references),
                ("reference_ids", self.reference_ids),
            )
            if value is not None
        }

    def validate(self, recipe: SamplingRecipe) -> None:
        """Validate a full stage before any heavyweight model is constructed."""
        _, guidance = recipe.resolved()
        if min(self.width, self.height) < 64 or self.width % 16 or self.height % 16:
            raise ValueError("Klein conditioning dimensions require multiples of 16 >=64")
        if self.text.shape != (1, 512, 12288) or self.text_ids.shape != (1, 512, 4):
            raise ValueError("Klein conditioning requires 512 text tokens and THWL positions")
        if (self.negative is not None) != (guidance > 1):
            raise ValueError("Saved negative conditioning must match active base CFG")
        if self.negative is not None and self.negative.shape != self.text.shape:
            raise ValueError("Negative conditioning must match positive text shape")
        positions = mx.arange(512, dtype=mx.float32)
        zeros = mx.zeros_like(positions)
        expected_text_ids = mx.stack([zeros, zeros, zeros, positions], axis=-1)[None]
        if not bool(mx.all(self.text_ids == expected_text_ids).item()):
            raise ValueError(
                "Klein saved text positions must match the complete 512-token sequence"
            )
        for geometry in self.geometries:
            if geometry.policy not in {"fit", "crop", "native"}:
                raise ValueError("Unknown saved reference geometry policy")
            expected_geometry = reference_geometry(
                geometry.source_size,
                (self.width, self.height),
                reference_count=len(self.geometries),
                reference_fit=None if geometry.policy == "native" else geometry.policy,
            )
            if geometry != expected_geometry:
                raise ValueError("Saved reference geometry does not match the output grid")
        if self.text.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise ValueError("Klein conditioning requires floating-point text")
        if (self.references is not None) != bool(self.geometries) or (
            self.reference_ids is not None
        ) != bool(self.geometries):
            raise ValueError(
                "Edit replay requires ordered reference latents, positions, and geometry"
            )
        if self.references is not None:
            count = sum(geometry.token_count for geometry in self.geometries)
            if self.references.shape != (1, count, 128):
                raise ValueError("Reference latents do not match the saved geometry")
            expected = []
            for index, geometry in enumerate(self.geometries):
                width, height = geometry.reference_size
                if min(width, height) < 16 or width % 16 or height % 16:
                    raise ValueError("Reference geometry requires positive aligned dimensions")
                expected.append(
                    spatial_ids(
                        height // 16,
                        width // 16,
                        reference_index=index,
                        offset_hw=geometry.offset_hw,
                    )
                )
            assert self.reference_ids is not None
            if self.reference_ids.shape != (1, count, 4) or not bool(
                mx.all(self.reference_ids == mx.concatenate(expected, axis=1)).item()
            ):
                raise ValueError("Reference positions do not match the ordered geometry")
        for name, value in self.tensors().items():
            if name in {"negative", "references"} and value.dtype != self.text.dtype:
                raise ValueError("Klein saved text and reference latents require one dtype")
            if not bool(mx.all(mx.isfinite(value)).item()):
                raise ValueError(f"Klein saved {name} must be finite")

    def astype(self, dtype: mx.Dtype) -> Flux2Conditioning:
        return Flux2Conditioning(
            self.text.astype(dtype),
            self.text_ids,
            self.negative.astype(dtype) if self.negative is not None else None,
            self.references.astype(dtype) if self.references is not None else None,
            self.reference_ids,
            self.geometries,
            self.prompt,
            self.negative_prompt,
            self.width,
            self.height,
        )


def save_conditioning(path: Path, conditioning: Flux2Conditioning, recipe: SamplingRecipe) -> None:
    conditioning.validate(recipe)
    save_weights(
        path,
        conditioning.tensors(),
        {
            "schema_version": TEXT_CONDITIONING_SCHEMA_VERSION,
            "artifact": TEXT_CONDITIONING_ARTIFACT,
            "context": json.dumps(
                {
                    "prompt": conditioning.prompt,
                    "negative_prompt": conditioning.negative_prompt,
                    "width": conditioning.width,
                    "height": conditioning.height,
                    "geometries": [value.to_metadata() for value in conditioning.geometries],
                }
            ),
        },
    )


def load_conditioning(path: Path, recipe: SamplingRecipe) -> Flux2Conditioning:
    inventory = inspect_safetensors(path)
    metadata = dict(inventory.metadata)
    if metadata.get("schema_version") != TEXT_CONDITIONING_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported Klein text-conditioning schema {metadata.get('schema_version')!r}"
        )
    if metadata.get("artifact") != TEXT_CONDITIONING_ARTIFACT:
        raise ValueError(f"{path}: expected a Klein text-conditioning artifact")
    names = set(inventory.tensor_names)
    if not {"text", "text_ids"} <= names or names - {
        "text",
        "text_ids",
        "negative",
        "references",
        "reference_ids",
    }:
        raise ValueError("Unexpected Klein conditioning tensor inventory")
    try:
        context = json.loads(metadata["context"])
        geometries = tuple(
            ReferenceGeometry(
                tuple(item["source_size"]),
                tuple(item["resized_size"]),
                tuple(item["crop_box"]),
                tuple(item["reference_size"]),
                tuple(item["offset_hw"]),
                item["policy"],
            )
            for item in context["geometries"]
        )
        if any(not isinstance(context[key], str) for key in ("prompt", "negative_prompt")):
            raise ValueError("Invalid prompt context")
        if any(type(context[key]) is not int for key in ("width", "height")):
            raise ValueError("Invalid grid context")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid Klein conditioning context") from exc
    expected_shapes = {"text": (1, 512, 12288), "text_ids": (1, 512, 4)}
    if "negative" in names:
        expected_shapes["negative"] = (1, 512, 12288)
    if geometries:
        count = sum(item.token_count for item in geometries)
        expected_shapes.update(references=(1, count, 128), reference_ids=(1, count, 4))
    if {item.name: item.shape for item in inventory.tensors} != expected_shapes:
        raise ValueError("Klein conditioning header does not match the complete stage contract")
    tensors, _ = load_weights_with_metadata(path)
    result = Flux2Conditioning(
        tensors["text"],
        tensors["text_ids"],
        tensors.get("negative"),
        tensors.get("references"),
        tensors.get("reference_ids"),
        geometries,
        context["prompt"],
        context["negative_prompt"],
        context["width"],
        context["height"],
    )
    result.validate(recipe)
    return result
