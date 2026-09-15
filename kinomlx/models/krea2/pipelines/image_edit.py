"""Reference-conditioned image generation using the normal Krea lifecycle."""

import gc
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import mlx.core as mx

from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.adapters.validation import AdapterWeightCheck, pending_weight_check
from kinomlx.components import ComponentLease
from kinomlx.config_records import derive_config
from kinomlx.io.image import load_image
from kinomlx.io.resampling import fit_reference, reference_fit_geometry
from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.settings import Settings

from ..adapters.edit_profile import resolve_edit_adapters
from ..adapters.loading import Krea2AdapterPlan
from ..adapters.turbo_profile import resolve_turbo_adapters
from ..components import (
    Krea2Components,
    Krea2ImageDecoderPort,
    Krea2TransformerPort,
    NativeKrea2Components,
)
from ..edit_validation import validate_edit_request
from ..grounding import GroundedKrea2TextConditioner
from ..image_processing import reference_boost_selection
from ..metadata import parse_vae_config
from ..reference_regions import parse_reference_region, reference_region_selection
from ..resource_types import Krea2Resources
from ..sampling import Krea2Checkpoint, pack_latents
from ..sizing import resolve_source_sizing
from ..text_conditioning import Krea2TextConditioner
from ..transformer import Krea2BoundaryObserver, Krea2TransformerOutput
from ..transformer.model import Krea2PreparedConditioning, Krea2Transformer
from ..transformer.reference import ReferenceAttention
from ..types import Krea2Request
from ..vae import Krea2ImageEncoder, load_krea2_encoder_weights
from .text_to_image import (
    Krea2GenerationOutput,
    generate_krea2,
    resolve_request_recipe,
)


@dataclass(frozen=True)
class Krea2ReferenceTokens:
    """Fixed image tokens and their spatial grid, independent of any adapter."""

    tokens: mx.array
    grid: tuple[int, int]
    strength: float = 1.0
    boost_selection: mx.array | None = None


@dataclass(frozen=True)
class _ReferenceBranch:
    source_hidden: mx.array
    source_positions: mx.array
    source_mask: mx.array | None
    hidden: mx.array
    positions: mx.array
    tokens: mx.array
    bias: ReferenceAttention | None
    conditioning: Krea2PreparedConditioning | None


class ReferenceConditionedTransformer:
    """Target-only predictions with fixed products retained for at most two CFG branches.

    The wrapper belongs to one generation lease. Conditioning arrays and model
    weights are fixed for that lease; the sampler changes only target tokens/time.
    """

    def __init__(
        self,
        model: Krea2TransformerPort,
        references: tuple[Krea2ReferenceTokens, ...],
        target_grid: tuple[int, int],
        *,
        centered: bool,
    ) -> None:
        self.model = model
        self.references = references
        self.target_grid = target_grid
        self.centered = centered
        self._branches: list[_ReferenceBranch] = []

    @property
    def adapter_weight_check(self) -> AdapterWeightCheck | None:
        return pending_weight_check(self.model)

    @property
    def adapter_receipts(self) -> tuple[AdapterReceipt, ...]:
        return self.model.adapter_receipts

    def _prepare(
        self, hidden: mx.array, positions: mx.array, mask: mx.array | None, dtype: mx.Dtype
    ) -> _ReferenceBranch:
        original_hidden, original_positions = hidden, positions
        if mask is not None:
            text_length = hidden.shape[1]
            valid = cast(list[bool], mask[0, :text_length].tolist())
            take = mx.array([i for i, value in enumerate(valid) if value], dtype=mx.int32)
            hidden = hidden[:, take]
            positions = mx.concatenate([positions[:, take], positions[:, text_length:]], axis=1)
        text_length = hidden.shape[1]
        reference_length = sum(ref.tokens.shape[1] for ref in self.references)
        target_length = positions.shape[1] - text_length
        logging.getLogger(__name__).info(
            "Edit sequence: text %d + reference %d + target %d = %d tokens",
            text_length,
            reference_length,
            target_length,
            text_length + reference_length + target_length,
        )
        ref_positions: list[mx.array] = []
        ref_tokens: list[mx.array] = []
        boosted = any(reference.strength != 1 for reference in self.references)
        biases: list[mx.array] = [mx.zeros((text_length,))] if boosted else []
        for frame, reference in enumerate(self.references, start=1):
            height, width = reference.grid
            rows = mx.repeat(mx.arange(height, dtype=mx.float32), width)
            cols = mx.tile(mx.arange(width, dtype=mx.float32), height)
            if self.centered:
                rows = rows + max(0, (self.target_grid[0] - height) / 2)
                cols = cols + max(0, (self.target_grid[1] - width) / 2)
            ref_positions.append(mx.stack([mx.full(rows.shape, frame), rows, cols], axis=-1)[None])
            ref_tokens.append(reference.tokens.astype(dtype))
            if boosted:
                values = mx.full((height * width,), math.log(max(reference.strength, 1e-4)))
                if reference.boost_selection is not None:
                    if reference.boost_selection.shape != (height * width,):
                        raise ValueError("reference boost selection must match its token grid")
                    values = mx.where(reference.boost_selection, values, 0)
                biases.append(values)
        tokens = mx.concatenate(ref_tokens, axis=1)
        combined_positions = mx.concatenate(
            [positions[:, :text_length], *ref_positions, positions[:, text_length:]], axis=1
        )
        bias = None
        if boosted:
            biases.append(mx.zeros((positions.shape[1] - text_length,)))
            bias = ReferenceAttention(text_length + tokens.shape[1], mx.concatenate(biases))
        conditioning = None
        if isinstance(self.model, Krea2Transformer) and not self.model.reference_denoise:
            conditioning = self.model.prepare_conditioning(hidden, combined_positions)
        branch = _ReferenceBranch(
            original_hidden,
            original_positions,
            mask,
            hidden,
            combined_positions,
            tokens,
            bias,
            conditioning,
        )
        if len(self._branches) == 2:
            self._branches.pop(0)
        self._branches.append(branch)
        return branch

    def __call__(
        self,
        image_tokens: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None,
        observer: Krea2BoundaryObserver | None = None,
        reference_attention: ReferenceAttention | None = None,  # noqa: ARG002
    ) -> Krea2TransformerOutput:
        branch = next(
            (
                entry
                for entry in self._branches
                if entry.source_hidden is encoder_hidden_states
                and entry.source_positions is position_ids
                and entry.source_mask is attention_mask
                and entry.tokens.dtype == image_tokens.dtype
            ),
            None,
        )
        if branch is None:
            branch = self._prepare(
                encoder_hidden_states, position_ids, attention_mask, image_tokens.dtype
            )
        combined = mx.concatenate([branch.tokens, image_tokens], axis=1)
        if branch.conditioning is not None:
            assert isinstance(self.model, Krea2Transformer)
            output = self.model.forward_prepared(
                combined,
                conditioning=branch.conditioning,
                timestep=timestep,
                observer=observer,
                reference_attention=branch.bias,
            )
        elif branch.bias is None:
            output = self.model(
                combined,
                encoder_hidden_states=branch.hidden,
                timestep=timestep,
                position_ids=branch.positions,
                attention_mask=None,
                observer=observer,
            )
        else:
            output = self.model(
                combined,
                encoder_hidden_states=branch.hidden,
                timestep=timestep,
                position_ids=branch.positions,
                attention_mask=None,
                observer=observer,
                reference_attention=branch.bias,
            )
        return replace(output, sample=output.sample[:, branch.tokens.shape[1] :])


@dataclass(frozen=True)
class ReferenceKrea2Components:
    base: Krea2Components
    references: tuple[Krea2ReferenceTokens, ...]
    target_grid: tuple[int, int]
    centered: bool

    def transformer(
        self, resources: Krea2Resources, *, adapter_plan: Krea2AdapterPlan | None = None
    ) -> ComponentLease[Krea2TransformerPort]:
        lease = self.base.transformer(resources, adapter_plan=adapter_plan)
        wrapper = ReferenceConditionedTransformer(
            lease.value, self.references, self.target_grid, centered=self.centered
        )
        return ComponentLease(wrapper, cleanup=lease.close)

    def image_decoder(self, resources: Krea2Resources) -> ComponentLease[Krea2ImageDecoderPort]:
        return self.base.image_decoder(resources)


@memory_managed
def generate_krea2_edit(
    request: Krea2Request,
    resources: Krea2Resources,
    *,
    components: Krea2Components | None = None,
    text_conditioner: Krea2TextConditioner | None = None,
    reporter: Reporter | None = None,
    checkpoint: Krea2Checkpoint | None = None,
    on_image: Callable[[int, Krea2GenerationOutput], None] | None = None,
) -> Krea2GenerationOutput:
    """Encode both reference paths before sampling a fresh target from noise."""
    validate_edit_request(request)
    selections, adapter_resolution = resolve_edit_adapters(
        resources.hf_home if resources.hf_home is not None else Settings.from_env().hf_home,
        edit_path=request.edit_adapter_path,
        edit_strength=request.edit_adapter_strength,
        edit_exclusions=request.edit_adapter_exclusions,
        additional=request.resolved_adapters(),
    )
    # Admit the profile before either reference encoder runs. The shared generation
    # path below performs the actual profile selection and fusion exactly once.
    resolve_turbo_adapters(
        resources.hf_home if request.turbo_4step else None,
        enabled=request.turbo_4step,
        strength=request.turbo_4step_strength,
        exclusions=request.turbo_4step_exclusions,
        additional=selections,
    )
    if resources.source_root is None:
        raise ValueError("editing requires an original Krea source root with the image encoder")
    sink = reporter if reporter is not None else NullReporter()
    pixels = tuple(load_image(path)[None] for path in request.source_images)
    sizing = None
    if request.match_source_aspect is not None:
        request, sizing = resolve_source_sizing(
            request,
            alignment=resources.capabilities.spatial_compression
            * resources.capabilities.transformer_patch_size,
            source_size=(pixels[0].shape[2], pixels[0].shape[1]),
        )
    geometry, _recipe = resolve_request_recipe(resources, request)
    boost_pixels = None
    if request.reference_boost_mask is not None:
        boost_pixels = load_image(request.reference_boost_mask)[None]
        if boost_pixels.shape[1:3] != pixels[-1].shape[1:3]:
            raise ValueError("reference boost mask dimensions must match the last source image")
        top, left, ch, cw, _, _ = reference_fit_geometry(
            pixels[-1].shape[1],
            pixels[-1].shape[2],
            geometry.height,
            geometry.width,
            mode=request.reference_fit,
        )
        boost_pixels = boost_pixels[:, top : top + ch, left : left + cw]

    vae_root = resources.source_root / "vae"
    encoder = Krea2ImageEncoder(
        parse_vae_config(vae_root / "config.json"), compute_dtype=resources.vae.dtype_policy
    )
    references: list[Krea2ReferenceTokens] = []
    try:
        with reported_phase(sink, "load reference VAE"):
            load_krea2_encoder_weights(encoder, vae_root / "diffusion_pytorch_model.safetensors")
        for index, image in enumerate(pixels):
            if checkpoint is not None:
                checkpoint()
            fitted = fit_reference(
                image, geometry.height, geometry.width, mode=request.reference_fit
            )
            latent = encoder(fitted, reporter=sink)
            patch = resources.capabilities.transformer_patch_size
            tokens = pack_latents(latent, patch_size=patch)
            strength = (
                request.reference_strength
                if index == len(pixels) - 1
                else request.scene_reference_strength
            )
            grid = (latent.shape[2] // patch, latent.shape[3] // patch)
            selection = None
            if (
                index == len(pixels) - 1
                and request.reference_boost_mask is not None
                and strength != 1
            ):
                assert boost_pixels is not None
                selection = reference_boost_selection(boost_pixels, grid)
            if index == len(pixels) - 1 and request.reference_boost_regions and strength != 1:
                region_selection = reference_region_selection(
                    tuple(
                        parse_reference_region(region) for region in request.reference_boost_regions
                    ),
                    grid,
                )
                selection = region_selection if selection is None else selection | region_selection
            mx.eval(tokens, *(() if selection is None else (selection,)))
            references.append(Krea2ReferenceTokens(tokens, grid, strength, selection))
    finally:
        del encoder
        gc.collect()
        mx.clear_cache()
    base = components if components is not None else NativeKrea2Components(reporter=sink)
    ports = ReferenceKrea2Components(
        base,
        tuple(references),
        (geometry.latent_height // patch, geometry.latent_width // patch),
        request.reference_fit == "fit",
    )
    conditioner = (
        text_conditioner
        if text_conditioner is not None
        else GroundedKrea2TextConditioner(
            pixels, request.grounding_px, request.grounding_system_prompt
        )
    )
    result = generate_krea2(
        derive_config(
            request,
            source_images=(),
            grounding_system_prompt="",
            reference_boost_mask=None,
            reference_boost_regions=(),
            edit_adapter_path=None,
            edit_adapter_strength=1.0,
            edit_adapter_exclusions="",
            match_source_aspect=None,
            scene_reference_strength=1.0,
        ),
        resources,
        components=ports,
        text_conditioner=conditioner,
        reporter=sink,
        checkpoint=checkpoint,
        on_image=on_image,
        # The resolved edit adapter travels as a runtime port, ahead of the
        # request's own adapters, so the request keeps recording only the
        # adapters the user wrote and a later restart replays exactly those.
        leading_adapters=selections[:1],
    )
    result.metadata["editing"] = {
        "adapter_resolution": adapter_resolution,
        "edit_adapter_strength": request.edit_adapter_strength,
        "edit_adapter_exclusions": list(selections[0].exclude),
        "recipe": "image-grounded-reference-v1",
        "source_images": [str(path) for path in request.source_images],
        "reference_fit": request.reference_fit,
        "output_sizing": None if sizing is None else sizing.to_metadata(),
        "grounding_px": request.grounding_px,
        "reference_boost_regions": list(request.reference_boost_regions),
        "grounding_system_prompt": request.grounding_system_prompt,
        "reference_boost_mask": str(request.reference_boost_mask)
        if request.reference_boost_mask is not None
        else None,
        "reference_grids": [list(ref.grid) for ref in references],
        "reference_strengths": [ref.strength for ref in references],
    }
    return result
