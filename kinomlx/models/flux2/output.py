"""Klein artifact layout over the shared transactional image publisher."""

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from kinomlx.config_rules import validate_record
from kinomlx.debug.metadata import selected_execution_sidecar_paths
from kinomlx.image_restart_config import ImageStage
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.image import save_image
from kinomlx.io.output_bundle import (
    BundleReservation,
    RetainedTransactionError,
    protect_sources,
    publish_staged,
)
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.io.output_paths import timestamped_output_base
from kinomlx.io.safetensors import save_weights
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.restart_selection import fingerprint_key

from .generation import Flux2ImageResult
from .output_policies import OUTPUT_POLICIES
from .sampling import SamplingRecipe
from .text_conditioning import save_conditioning


@dataclass(frozen=True)
class Flux2OutputPlan:
    base: Path
    artifacts: tuple[tuple[str, tuple[Path, ...]], ...]
    execution_sidecars: tuple[tuple[str, Path], ...]
    force: bool = False

    @property
    def directory(self) -> Path:
        return self.base.parent

    @property
    def image_count(self) -> int:
        """Report the images this plan set out to write, digest keying included."""
        return len(dict(self.artifacts)["images"])

    def targets(self) -> tuple[Path, ...]:
        return tuple(*paths for _, paths in self.artifacts) + tuple(
            path for _, path in self.execution_sidecars
        )

    def outputs(self) -> dict[str, object]:
        result: dict[str, object] = {
            name: [str(path) for path in paths]
            if name == "images" or len(paths) > 1
            else str(paths[0])
            for name, paths in self.artifacts
        }
        result.update({name: str(path) for name, path in self.execution_sidecars})
        return result

    def reserve(self) -> BundleReservation[Flux2OutputPlan]:
        return BundleReservation(self)


def plan_output(
    config: ImageOutputConfig,
    *,
    variant: str,
    seed: int,
    image_count: int,
    seeds: tuple[int, ...] | None = None,
    sources: tuple[Path, ...] = (),
    base: Path | None = None,
    decode: bool = False,
) -> Flux2OutputPlan:
    validate_record(config, location="[output]")
    if seeds is not None and len(seeds) != image_count:
        raise ValueError("Output seed identities must match the image count")
    if image_count < 1:
        raise ValueError("Image output requires at least one image")
    exact = config.path.expanduser().absolute() if config.path else None
    if exact:
        base = exact.with_suffix("")
    elif base is None:
        base = timestamped_output_base(
            config.directory, config.prefix or f"flux2-{variant}-seed{seed}"
        ).absolute()
    assert base is not None
    suffix = exact.suffix if exact else ".png"
    stems = (
        tuple(
            base.with_name(
                f"{base.name}-{index:03d}"
                + (f"-seed{seeds[index - 1]}" if seeds is not None else "")
            )
            for index in range(1, image_count + 1)
        )
        if image_count > 1
        else (base,)
    )
    artifacts = [("images", tuple(stem.with_name(stem.name + suffix) for stem in stems))]
    for name, tail in (
        ("initial_noise", "initial_noise"),
        ("final_latents", "final_latents"),
        ("text_conditioning", "text"),
    ):
        selected = getattr(config, f"save_{name}")
        if OUTPUT_POLICIES[f"save_{name}"].selected(
            selected,
            save_all=config.save_all_sidecars,
            stage=ImageStage.DECODE if decode else ImageStage.TEXT,
            field=f"save_{name}",
        ):
            artifacts.append(
                (name, tuple(stem.with_name(f"{stem.name}_{tail}.safetensors") for stem in stems))
            )
    plan = Flux2OutputPlan(
        base,
        tuple(artifacts),
        tuple(selected_execution_sidecar_paths(base, config).items()),
        config.force,
    )
    protect_sources(plan.targets(), sources)
    return plan


class Flux2OutputSink:
    def __init__(self, plan: Flux2OutputPlan, *, reporter: Reporter | None = None) -> None:
        self.plan = plan
        self.reporter = reporter or NullReporter()
        self.completed: set[int] = set()
        self.published: set[Path] = set()
        self.fingerprints: dict[str, str] = {}

    def completed_outputs(self) -> dict[str, object]:
        return {
            name: [str(path) for path in paths if path in self.published]
            for name, paths in self.plan.artifacts
        }

    def write_image(
        self,
        index: int,
        image: Flux2ImageResult,
        recipe: SamplingRecipe,
        *,
        reservation: BundleReservation[Flux2OutputPlan],
    ) -> None:
        """Publish one completed seed, independently of later seeds."""
        if index in self.completed:
            raise ValueError("Seed output was already published")
        child = replace(
            self.plan,
            artifacts=tuple((name, (paths[index],)) for name, paths in self.plan.artifacts),
            execution_sidecars=(),
        )
        sink = Flux2OutputSink(child, reporter=self.reporter)
        try:
            sink.write([image], recipe, reservation=reservation, publish_each=True)
            self.completed.add(index)
        finally:
            self.published.update(sink.published)
            count = self.plan.image_count
            self.fingerprints.update(
                {
                    fingerprint_key(key, index=index, count=count): value
                    for key, value in sink.fingerprints.items()
                }
            )

    def write(
        self,
        results: Sequence[Flux2ImageResult],
        recipe: SamplingRecipe,
        *,
        reservation: BundleReservation[Flux2OutputPlan],
        publish_each: bool = False,
    ) -> dict[str, object]:
        if not reservation.active or any(
            target not in reservation.modes for target in self.plan.targets()
        ):
            raise ValueError("Image publication requires its active output reservation")
        if len(results) != len(dict(self.plan.artifacts)["images"]):
            raise ValueError("Image count does not match the output plan")
        transaction = Path(tempfile.mkdtemp(prefix=".flux2-output-", dir=self.plan.directory))
        retain = False
        temporary: dict[Path, Path] = {}
        fingerprints: dict[str, str] = {}
        try:
            with reported_phase(self.reporter, "write outputs"):
                for name, paths in self.plan.artifacts:
                    for index, (result, target) in enumerate(zip(results, paths, strict=True)):
                        if index in self.completed:
                            continue
                        peer = transaction / target.name
                        if name == "images":
                            if not bool(mx.all(mx.isfinite(result.pixels)).item()):
                                raise ValueError("Image output refuses nonfinite RGB")
                            save_image(peer, result.pixels)
                        elif name == "text_conditioning":
                            if result.conditioning is None:
                                raise ValueError("Result has no complete conditioning stage")
                            save_conditioning(peer, result.conditioning, recipe)
                        else:
                            value = (
                                result.latents if name == "final_latents" else result.initial_noise
                            )
                            if value is None or not bool(mx.all(mx.isfinite(value)).item()):
                                raise ValueError(f"Result has no finite {name} stage")
                            tensors = {name: value}
                            if name == "initial_noise":
                                if result.sigmas is None:
                                    raise ValueError(
                                        "Initial noise requires the recorded Euler schedule"
                                    )
                                tensors["schedule"] = result.sigmas
                            save_weights(
                                peer,
                                tensors,
                                {
                                    "schema": "kinomlx.flux2.tensor.v1",
                                    "artifact": name,
                                    "width": str(result.width),
                                    "height": str(result.height),
                                },
                            )
                        if not peer.is_file() or peer.stat().st_size == 0:
                            raise ValueError(f"Artifact writer did not create {peer.name}")
                        temporary[target] = peer
                        key = name if len(paths) == 1 else f"{name}.{index}"
                        fingerprints[key] = file_sha256(peer)
                        if publish_each:
                            publish_staged(temporary, reservation.modes, transaction)
                            self.published.update(temporary)
                            self.fingerprints.update(fingerprints)
                            temporary.clear()
                            fingerprints.clear()
                publish_staged(temporary, reservation.modes, transaction)
                self.published.update(temporary)
                self.fingerprints.update(fingerprints)
            return self.plan.outputs()
        except RetainedTransactionError:
            retain = True
            raise
        finally:
            if not retain:
                shutil.rmtree(transaction)
