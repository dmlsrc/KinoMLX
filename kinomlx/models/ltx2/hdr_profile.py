"""Resolve explicit HDR producer facts without checkpoint allowlists."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kinomlx.io.safetensors import read_metadata
from kinomlx.media.signals import ColorTransfer

from .resources import LTX2Resources
from .types import DistilledRequest, LoRASelection

if TYPE_CHECKING:
    from .cache import LoRAAdapterReceipt


@dataclass(frozen=True)
class HDRRecipeFacts:
    """Facts that authorize one model working-transfer interpretation."""

    producer: str
    working_transfer: ColorTransfer
    semantic_anchor: str
    adapter_path: Path | None = None
    reference_downscale_factor: int | None = None
    stage_1_strength: float | None = None
    stage_2_strength: float | None = None

    def to_metadata(self) -> dict[str, object]:
        """Return the narrow receipt without copying arbitrary adapter metadata."""
        return {
            "producer": self.producer,
            "working_transfer": self.working_transfer.value,
            "semantic_anchor": self.semantic_anchor,
            "adapter_path": None if self.adapter_path is None else str(self.adapter_path),
            "reference_downscale_factor": self.reference_downscale_factor,
            "stage_1_strength": self.stage_1_strength,
            "stage_2_strength": self.stage_2_strength,
        }


def _effective_stage_strength(selection: LoRASelection, stage: int) -> float:
    if stage == 1:
        override = selection.stage_1_strength
    elif stage == 2:
        override = selection.stage_2_strength
    else:
        raise ValueError(f"LoRA stage must be 1 or 2, got {stage}")
    return selection.strength if override is None else override


def _resolve_ltx23_hdr_adapter(request: DistilledRequest) -> HDRRecipeFacts:
    matches: list[tuple[LoRASelection, int]] = []
    for selection in request.resolved_loras():
        metadata = read_metadata(selection.path)
        transform = metadata.get("hdr_transform", "").strip().lower()
        if not transform:
            continue
        if transform != "logc3":
            raise ValueError(
                f"HDR adapter {selection.path} declares unsupported hdr_transform={transform!r}"
            )
        raw_factor = metadata.get("reference_downscale_factor", "1").strip()
        try:
            factor = int(raw_factor)
        except ValueError as exc:
            raise ValueError(
                f"HDR adapter {selection.path} has invalid reference_downscale_factor={raw_factor!r}"
            ) from exc
        matches.append((selection, factor))

    if not matches:
        raise ValueError("LTX-2.3 HDR requires a LoRA that explicitly declares hdr_transform=logc3")
    if len(matches) != 1:
        raise ValueError("LTX-2.3 HDR requires exactly one explicitly declared HDR adapter")
    selection, factor = matches[0]
    if factor != 1:
        raise ValueError("the supported LTX-2.3 HDR profile requires reference_downscale_factor=1")
    if selection.exclude:
        raise ValueError("the HDR adapter cannot use LoRA target exclusions")
    stage_1 = _effective_stage_strength(selection, 1)
    stage_2 = _effective_stage_strength(selection, 2)
    if stage_1 <= 0.0 or stage_2 <= 0.0:
        raise ValueError("the HDR adapter must have positive strength in both denoising stages")
    if request.generate_audio:
        raise ValueError("the supported LTX-2.3 HDR IC-LoRA recipe is video-only")
    if request.generated_keyframes:
        raise ValueError("the LTX-2.3 HDR IC-LoRA recipe does not support generated keyframes")
    if request.image is not None:
        raise ValueError(
            "LTX-2.3 HDR reference conditioning uses hdr_reference, not image/keyframe conditioning"
        )
    if request.hdr_reference is None:
        raise ValueError(
            "the LTX-2.3 HDR IC-LoRA is a video-to-video SDR-to-HDR converter and "
            "requires an SDR hdr_reference video; without a reference track it runs "
            "off-distribution, so text-only HDR is not supported"
        )
    return HDRRecipeFacts(
        producer="ltx23-hdr-ic-lora",
        working_transfer=ColorTransfer.LOGC3,
        semantic_anchor="hdr-adapter",
        adapter_path=selection.path,
        reference_downscale_factor=factor,
        stage_1_strength=stage_1,
        stage_2_strength=stage_2,
    )


def resolve_hdr_recipe(
    request: DistilledRequest,
    resources: LTX2Resources,
) -> HDRRecipeFacts | None:
    """Resolve output HDR semantics from generation plus explicit recipe facts."""
    if request.hdr is None:
        return None
    generation = resources.capabilities.model_generation
    if generation == "2.5" and resources.capabilities.native_hdr:
        if request.image is None or request.image.path.suffix.lower() != ".exr":
            raise ValueError(
                "LTX-2.5 native HDR is image-to-video from an HDR EXR condition, "
                "which establishes the ACEScct working signal; text-only and "
                "SDR-only generation remain SDR"
            )
        return HDRRecipeFacts(
            producer="ltx25-native",
            working_transfer=ColorTransfer.ACESCCT,
            semantic_anchor="exr-condition",
        )
    if generation == "2.3":
        return _resolve_ltx23_hdr_adapter(request)
    raise ValueError(f"LTX-{generation} does not provide a supported HDR recipe")


def validate_hdr_adapter_placement(
    recipe: HDRRecipeFacts | None,
    receipts: Sequence[LoRAAdapterReceipt],
    *,
    stage: int,
) -> None:
    """Require the selected HDR adapter to fit completely before denoising."""
    if recipe is None or recipe.adapter_path is None:
        return
    target = recipe.adapter_path.resolve()
    matches = tuple(receipt for receipt in receipts if receipt.path.resolve() == target)
    if len(matches) != 1:
        raise ValueError(
            f"LTX-2.3 HDR stage {stage} expected one placement receipt for "
            f"{recipe.adapter_path}, got {len(matches)}"
        )
    receipt = matches[0]
    if receipt.complete_targets <= 0 or receipt.placed_targets != receipt.complete_targets:
        raise ValueError(
            f"LTX-2.3 HDR stage {stage} adapter is not structurally complete: "
            f"placed {receipt.placed_targets}/{receipt.complete_targets} targets"
        )


__all__ = [
    "HDRRecipeFacts",
    "resolve_hdr_recipe",
    "validate_hdr_adapter_placement",
]
