"""Resume Krea 2 from saved text conditioning or final latents."""

import json
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from kinomlx.config_records import derive_config, replace_config
from kinomlx.config_rules import validate_record
from kinomlx.io.safetensors import inspect_safetensors, load_weights
from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import NullReporter, Reporter, reported_phase

from ..components import Krea2Components, NativeKrea2Components
from ..request_rules import admit_recipe_mode
from ..request_sources import resolve_request_sources
from ..resource_types import Krea2Resources
from ..restart import Krea2Restart
from ..sampling import Krea2Checkpoint, Krea2Geometry, resolve_geometry
from ..text_conditioning import Krea2TextConditioner
from ..types import Krea2Request
from .text_to_image import Krea2GenerationOutput, decode_krea2_latents, generate_krea2

_LATENT_DTYPES = frozenset({mx.bfloat16, mx.float16, mx.float32})
_LATENT_STORAGE_DTYPES = frozenset({"BF16", "F16", "F32"})


def load_final_latents(
    path: Path,
    *,
    request: Krea2Request,
    resources: Krea2Resources,
    geometry: Krea2Geometry,
    reporter: Reporter,
) -> tuple[mx.array, dict[str, str]]:
    """Load and structurally validate one KinoMLX Krea final-latent sidecar."""
    expected_shape = (
        request.image_count,
        resources.capabilities.latent_channels,
        geometry.latent_height,
        geometry.latent_width,
    )
    with reported_phase(reporter, "load final latents"):
        inventory = inspect_safetensors(path)
        metadata = dict(inventory.metadata)
        try:
            saved_system_prompt = json.loads(metadata.get("system_prompt", "null"))
        except (ValueError, RecursionError) as exc:
            raise ValueError("Invalid Krea final-latent system prompt metadata") from exc
        if saved_system_prompt != request.system_prompt:
            raise ValueError(
                "Final latents already contain the effect of their system prompt; decode cannot change it"
            )
        descriptors = {tensor.name: tensor for tensor in inventory.tensors}
        descriptor = descriptors.get("final_latents")
        if descriptor is None:
            raise ValueError(f"Krea restart artifact {path} has no final_latents tensor")
        if len(descriptors) != 1:
            extras = ", ".join(sorted(name for name in descriptors if name != "final_latents"))
            raise ValueError(f"Krea restart artifact {path} has unexpected tensors: {extras}")
        if descriptor.dtype not in _LATENT_STORAGE_DTYPES:
            raise ValueError(
                f"Krea final_latents in {path} must use BF16, F16, or F32 storage, "
                f"got {descriptor.dtype}"
            )
        if descriptor.shape != expected_shape:
            raise ValueError(
                f"Krea final latent shape {descriptor.shape} in {path} does not match "
                f"{expected_shape}"
            )

        final_latents = load_weights(path)["final_latents"]
        if final_latents.dtype not in _LATENT_DTYPES:
            raise ValueError(
                f"Krea final_latents in {path} has unsupported loaded dtype {final_latents.dtype}"
            )
        if not bool(mx.all(mx.isfinite(final_latents)).item()):
            raise ValueError(f"Krea final_latents in {path} must contain only finite values")
        mx.eval(final_latents)
    return final_latents, dict(inventory.metadata)


@memory_managed
def restart_krea2(
    request: Krea2Request,
    resources: Krea2Resources,
    *,
    restart: Krea2Restart,
    components: Krea2Components | None = None,
    text_conditioner: Krea2TextConditioner | None = None,
    reporter: Reporter | None = None,
    checkpoint: Krea2Checkpoint | None = None,
    on_image: Callable[[int, Krea2GenerationOutput], None] | None = None,
) -> Krea2GenerationOutput:
    """Resume at the selected Krea station from one saved product."""
    validate_record(request, location="[generate]")
    validate_record(restart, location="[restart]")
    admit_recipe_mode(request, "decode" if restart.phase == "decode" else "text")
    sink = reporter if reporter is not None else NullReporter()
    selected_components = (
        components if components is not None else NativeKrea2Components(reporter=sink)
    )
    if restart.phase == "denoise":
        text_path = restart.text_conditioning
        assert text_path is not None
        if not text_path.is_file():
            raise ValueError(f"restart text-conditioning artifact does not exist: {text_path}")
        replay_request = replace_config(request, text_conditioning=text_path)
        output = generate_krea2(
            replay_request,
            resources,
            components=selected_components,
            text_conditioner=text_conditioner,
            reporter=sink,
            checkpoint=checkpoint,
            text_replay_policy="observe",
            on_image=on_image,
        )
        output.metadata["execution_mode"] = "restart-denoise"
        return output

    # Decoding consumes the saved latent, not a saved text-conditioning path.
    request = resolve_request_sources(derive_config(request, text_conditioning=None))
    del text_conditioner
    latent_path = restart.latents
    assert latent_path is not None
    if not latent_path.is_file():
        raise ValueError(f"restart latent artifact does not exist: {latent_path}")
    geometry = resolve_geometry(
        request,
        spatial_compression=resources.capabilities.spatial_compression,
        patch_size=resources.capabilities.transformer_patch_size,
    )
    final_latents, declared_metadata = load_final_latents(
        latent_path,
        request=request,
        resources=resources,
        geometry=geometry,
        reporter=sink,
    )
    prompts = request.expanded_prompts()
    seeds = request.selected_seeds() * len(request.prompts)
    images, decode_metadata, vae_metadata = decode_krea2_latents(
        request,
        resources,
        final_latents,
        geometry=geometry,
        prompts=prompts,
        seeds=seeds,
        on_image=(
            None
            if on_image is None
            else lambda image: on_image(image.index, Krea2GenerationOutput(images=(image,)))
        ),
        components=selected_components,
        reporter=sink,
        checkpoint=checkpoint,
    )
    negative_prompts = request.expanded_negative_prompts()
    metadata: dict[str, object] = {
        "execution_mode": "restart-decode",
        "variant": resources.variant.value,
        "revision": resources.revision,
        "recipe_identity": resources.recipe_identity,
        "geometry": geometry.to_metadata(),
        "decode": decode_metadata,
        "image_count": request.image_count,
        "prompts": list(prompts),
        "negative_prompts": (None if negative_prompts is None else list(negative_prompts)),
        "system_prompt": request.system_prompt,
        "source_latents": {
            "path": str(latent_path.expanduser().absolute()),
            "metadata": declared_metadata,
        },
        "vae": vae_metadata,
        "warnings": [],
    }
    # Retain the decoded latent so this station can re-emit it, letting the
    # bundle it writes be decode-restarted in turn.
    return Krea2GenerationOutput(images=images, metadata=metadata, final_latents=final_latents)


__all__ = ["load_final_latents", "restart_krea2"]
