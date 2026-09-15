"""Transactional native still-image publication for Krea 2."""

import json
import logging
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Literal

import mlx.core as mx

from kinomlx.config_records import supplied_fields
from kinomlx.config_rules import validate_record
from kinomlx.debug.metadata import selected_execution_sidecar_paths
from kinomlx.io.atomic import atomic_output_path
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.image import save_image
from kinomlx.io.output_bundle import (
    BundleOutputError,
    BundleReservation,
    RetainedTransactionError,
    protect_sources,
    publish_staged,
)
from kinomlx.io.output_config import LOSSLESS_IMAGE_PATH, ImageOutputConfig
from kinomlx.io.output_paths import timestamped_output_base
from kinomlx.io.reservation import PathReservation, PathReservationError
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.restart_selection import SEED_OUTPUTS, fingerprint_key

from .catalog import Krea2Variant
from .output_policies import OUTPUT_POLICIES
from .pipelines.text_to_image import Krea2GenerationOutput, select_conditioning
from .request_sources import resolve_request_sources
from .text_conditioning import text_conditioning_artifact
from .types import Krea2Request
from .vae import Krea2DecodedRGB

_log = logging.getLogger(__name__)


class Krea2StillOutputError(BundleOutputError):
    """A Krea still target cannot be reserved or published safely."""


@dataclass(frozen=True)
class Krea2StillOutputPlan:
    """Resolved single-image target known before heavyweight inference."""

    path: Path
    force: bool = False

    def __post_init__(self) -> None:
        path = self.path.expanduser()
        LOSSLESS_IMAGE_PATH.validate(path)
        object.__setattr__(self, "path", path)

    def reserve(self) -> Krea2StillOutputReservation:
        return Krea2StillOutputReservation(self)


class Krea2StillOutputReservation:
    """Context-owned hidden marker for one Krea still target."""

    def __init__(self, plan: Krea2StillOutputPlan) -> None:
        self.plan = plan
        self._marker: PathReservation | None = None
        self._entered = False

    @property
    def active(self) -> bool:
        return self._marker is not None and self._marker.active

    def __enter__(self) -> Krea2StillOutputReservation:
        if self._entered:
            raise Krea2StillOutputError("Krea still output reservation is already active")
        self._entered = True
        path = self.plan.path
        try:
            status = path.lstat()
        except FileNotFoundError:
            status = None
        if status is not None:
            if stat.S_ISLNK(status.st_mode):
                raise Krea2StillOutputError("Krea still output refuses symbolic links")
            if not stat.S_ISREG(status.st_mode):
                raise Krea2StillOutputError("Krea still output target must be a regular file")
            if not self.plan.force:
                raise Krea2StillOutputError(f"Krea still output already exists: {path}")
        try:
            self._marker = PathReservation.acquire(path)
        except PathReservationError as exc:
            raise Krea2StillOutputError(str(exc)) from exc
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        marker = self._marker
        self._marker = None
        if marker is not None:
            marker.release()


type Krea2StillWriter = Callable[[Path, mx.array], object]
type Krea2OutputCheckpoint = Callable[[], None]


def _display_rgb(pixels: mx.array) -> mx.array:
    return mx.clip(pixels * 0.5 + 0.5, 0.0, 1.0)


def _require_written_image(path: Path) -> None:
    try:
        status = path.stat()
    except FileNotFoundError:
        status = None
    if status is None or not stat.S_ISREG(status.st_mode) or status.st_size == 0:
        raise BundleOutputError(f"image writer did not create a nonempty regular file: {path.name}")


class Krea2StillOutputSink:
    """Apply the display transform and atomically publish one lossless still."""

    def __init__(
        self,
        plan: Krea2StillOutputPlan,
        *,
        writer: Krea2StillWriter = save_image,
    ) -> None:
        self.plan = plan
        self.writer = writer

    def write(
        self,
        decoded: Krea2DecodedRGB,
        *,
        reservation: Krea2StillOutputReservation | None = None,
        checkpoint: Krea2OutputCheckpoint | None = None,
    ) -> Path:
        """Publish one batch-size-one product, rolling back every failure."""
        if decoded.pixels.shape[0] != 1:
            raise ValueError("Krea still output requires a batch of exactly one image")
        mx.eval(decoded.pixels)
        if not mx.all(mx.isfinite(decoded.pixels)).item():
            raise ValueError("Krea still output refuses nonfinite RGB")
        selected = self.plan.reserve() if reservation is None else reservation
        if selected.plan != self.plan:
            raise ValueError("Krea still output reservation belongs to a different plan")

        def publish(active: Krea2StillOutputReservation) -> Path:
            if not active.active:
                raise ValueError("Krea still output reservation is not active")
            if checkpoint is not None:
                checkpoint()
            display_rgb = _display_rgb(decoded.pixels[0])
            suffix = f".tmp{self.plan.path.suffix.lower()}"
            try:
                with atomic_output_path(self.plan.path, temp_suffix=suffix) as temporary:
                    self.writer(temporary, display_rgb)
                    _require_written_image(temporary)
                    if checkpoint is not None:
                        checkpoint()
            except Exception as exc:
                if isinstance(exc, Krea2StillOutputError):
                    raise
                raise Krea2StillOutputError(f"Krea still output failed: {exc}") from exc
            return self.plan.path

        if reservation is not None:
            return publish(selected)
        with selected as active:
            return publish(active)


def publish_krea2_still(
    path: Path | str,
    decoded: Krea2DecodedRGB,
    *,
    force: bool = False,
) -> Path:
    """Convenience wrapper for one transactionally published Krea still."""
    plan = Krea2StillOutputPlan(Path(path), force=force)
    return Krea2StillOutputSink(plan).write(decoded)


SIDECAR_TENSORS = ("initial_noise", "final_latents", "text_conditioning")


@dataclass(frozen=True)
class Krea2ArtifactSet:
    """The primary files published by one successful Krea generation."""

    images: tuple[Path, ...]
    # One path per image, or empty when that sidecar was not selected.
    initial_noise: tuple[Path, ...] = ()
    final_latents: tuple[Path, ...] = ()
    text_conditioning: tuple[Path, ...] = ()

    def single_path(self, name: str) -> Path | None:
        """Return the one selected path of a single-image set, or None."""
        paths: tuple[Path, ...] = getattr(self, name)
        if len(paths) > 1:
            raise ValueError("Select one seed before writing a Krea sidecar")
        return paths[0] if paths else None

    def paths(self) -> tuple[Path, ...]:
        return (
            *self.images,
            *self.initial_noise,
            *self.final_latents,
            *self.text_conditioning,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"images": [str(path) for path in self.images]}
        for name in SIDECAR_TENSORS:
            paths: tuple[Path, ...] = getattr(self, name)
            if paths:
                result[name] = str(paths[0]) if len(paths) == 1 else [str(path) for path in paths]
        return result


@dataclass(frozen=True)
class Krea2OutputPlan:
    """Resolved image and sidecar targets reserved before model loading."""

    base: Path
    artifacts: Krea2ArtifactSet
    execution_sidecars: tuple[tuple[str, Path], ...]
    force: bool

    @property
    def directory(self) -> Path:
        return self.base.parent

    @property
    def image_count(self) -> int:
        """Report the images this plan set out to write, digest keying included."""
        return len(self.artifacts.images)

    def sidecar_paths(self) -> dict[str, Path]:
        return dict(self.execution_sidecars)

    def seed_artifacts(self, index: int) -> Krea2ArtifactSet:
        """Narrow the plan to one image and that image's requested sidecars."""

        def selected(name: str) -> tuple[Path, ...]:
            paths: tuple[Path, ...] = getattr(self.artifacts, name)
            return (paths[index],) if paths else ()

        return Krea2ArtifactSet(
            (self.artifacts.images[index],),
            *(selected(name) for name in SIDECAR_TENSORS),
        )

    def targets(self) -> tuple[Path, ...]:
        return (*self.artifacts.paths(), *(path for _name, path in self.execution_sidecars))

    def reserve(self) -> BundleReservation[Krea2OutputPlan]:
        return BundleReservation[Krea2OutputPlan](self)


def plan_krea2_output(
    request: Krea2Request,
    variant: Krea2Variant | str,
    config: ImageOutputConfig,
    *,
    base: Path | None = None,
    station: Literal["generate", "decode", "denoise"] = "generate",
) -> Krea2OutputPlan:
    """Resolve a bundle, optionally using an app-selected generated run stem."""
    validate_record(config, location="[output]")
    request = resolve_request_sources(request)
    selected_variant = Krea2Variant(variant)
    exact = None if config.path is None else Path(config.path).expanduser()
    if exact is not None:
        base = exact.with_suffix("")
        suffix = exact.suffix.lower()
    else:
        prefix = (
            f"krea2-{selected_variant.value}-seed{request.selected_seeds()[0]}"
            if config.prefix is None
            else config.prefix
        )
        if base is None:
            base = timestamped_output_base(config.directory, prefix)
        suffix = ".png"
    images: tuple[Path, ...]
    if request.image_count == 1:
        images = (exact if exact is not None else base.with_name(f"{base.name}{suffix}"),)
    else:
        digits = max(3, len(str(request.image_count)))
        images = tuple(
            base.parent
            / (
                f"{base.name}-{index:0{digits}d}"
                f"-seed{request.selected_seeds()[(index - 1) % len(request.selected_seeds())]}"
                + suffix
            )
            for index in range(1, request.image_count + 1)
        )
    recipe_mode = "edit" if request.source_images else "text"
    explicit_output = supplied_fields(config) or frozenset()
    save_initial_noise = OUTPUT_POLICIES["save_initial_noise"].selected_for_recipe(
        config.save_initial_noise,
        save_all=config.save_all_sidecars,
        station=station,
        field="save_initial_noise",
        recipe_mode=recipe_mode,
        explicitly_supplied="save_initial_noise" in explicit_output,
    )
    save_final_latents = OUTPUT_POLICIES["save_final_latents"].selected_for_recipe(
        config.save_final_latents,
        save_all=config.save_all_sidecars,
        station=station,
        field="save_final_latents",
        recipe_mode=recipe_mode,
        explicitly_supplied="save_final_latents" in explicit_output,
    )
    save_text_conditioning = OUTPUT_POLICIES["save_text_conditioning"].selected_for_recipe(
        config.save_text_conditioning,
        save_all=config.save_all_sidecars,
        station=station,
        field="save_text_conditioning",
        recipe_mode=recipe_mode,
        explicitly_supplied="save_text_conditioning" in explicit_output,
    )

    def sidecars(enabled: bool, suffix: str) -> tuple[Path, ...]:
        if not enabled:
            return ()
        return tuple(image.with_name(f"{image.stem}_{suffix}.safetensors") for image in images)

    plan = Krea2OutputPlan(
        base=base,
        artifacts=Krea2ArtifactSet(
            images=images,
            initial_noise=sidecars(save_initial_noise, "initial_noise"),
            final_latents=sidecars(save_final_latents, "final_latents"),
            text_conditioning=sidecars(save_text_conditioning, "text"),
        ),
        execution_sidecars=tuple(selected_execution_sidecar_paths(base, config).items()),
        force=config.force,
    )
    sources = request.source_images + (
        () if request.reference_boost_mask is None else (request.reference_boost_mask,)
    )
    protect_sources(plan.targets(), sources)
    return plan


class Krea2OutputSink:
    """Stage and transactionally publish every image and tensor sidecar."""

    def __init__(
        self,
        plan: Krea2OutputPlan,
        *,
        reporter: Reporter | None = None,
        image_writer: Krea2StillWriter = save_image,
    ) -> None:
        self.plan = plan
        self.reporter = reporter if reporter is not None else NullReporter()
        self.image_writer = image_writer
        self.completed: set[int] = set()
        self.published: set[Path] = set()
        self.fingerprints: dict[str, str] = {}
        self.fingerprint_errors: dict[str, str] = {}

    def completed_outputs(self) -> dict[str, object]:
        return {
            "images": [str(path) for path in self.plan.artifacts.images if path in self.published],
            "published_artifacts": [str(path) for path in sorted(self.published)],
            # Each entry names its own image, so a restart of a canceled run
            # selects by image rather than by position in this list.
            SEED_OUTPUTS: [
                {"index": index, **self.plan.seed_artifacts(index).to_dict()}
                for index in sorted(self.completed)
            ],
        }

    def write_image(
        self,
        index: int,
        result: Krea2GenerationOutput,
        *,
        reservation: BundleReservation[Krea2OutputPlan],
    ) -> None:
        """Publish one image and its requested tensor stations before the next seed."""
        if index in self.completed:
            raise ValueError("Seed output was already published")
        child = replace(self.plan, artifacts=self.plan.seed_artifacts(index), execution_sidecars=())
        sink = Krea2OutputSink(child, reporter=self.reporter, image_writer=self.image_writer)
        try:
            sink.write(result, reservation=reservation, publish_each=True)
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
            self.fingerprint_errors.update(
                {
                    fingerprint_key(key, index=index, count=count): value
                    for key, value in sink.fingerprint_errors.items()
                }
            )

    def _fingerprint_temporaries(
        self, temporary: dict[Path, Path]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Hash staged bytes; expose their receipts only after publication."""
        fingerprints: dict[str, str] = {}
        errors: dict[str, str] = {}
        for name, path in (
            ("initial_noise", self.plan.artifacts.single_path("initial_noise")),
            ("final_latents", self.plan.artifacts.single_path("final_latents")),
            ("text_conditioning", self.plan.artifacts.single_path("text_conditioning")),
        ):
            if path is None or path not in temporary:
                continue
            try:
                fingerprints[name] = file_sha256(temporary[path])
            except OSError as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                _log.warning("Could not fingerprint Krea output %s: %s", path, exc)
        return fingerprints, errors

    def _write_temporaries(
        self,
        result: Krea2GenerationOutput,
        transaction: Path,
        checkpoint: Krea2OutputCheckpoint | None,
        publish: Callable[[dict[Path, Path]], None] | None = None,
    ) -> dict[Path, Path]:
        if len(result.images) != len(self.plan.artifacts.images):
            raise BundleOutputError("Krea result image count does not match the output plan")
        temporary: dict[Path, Path] = {}
        total = len(self.plan.artifacts.paths())
        phase = "write outputs"
        self.reporter.phase_start(phase, total=total, unit="artifact")
        try:
            for image, target in zip(
                result.images,
                self.plan.artifacts.images,
                strict=True,
            ):
                if image.index in self.completed:
                    continue
                if checkpoint is not None:
                    checkpoint()
                mx.eval(image.pixels)
                if not mx.all(mx.isfinite(image.pixels)).item():
                    raise BundleOutputError("Krea output refuses nonfinite RGB")
                peer = transaction / target.name
                self.image_writer(peer, _display_rgb(image.pixels))
                _require_written_image(peer)
                temporary[target] = peer
                if publish is not None:
                    publish(temporary)
                self.reporter.phase_advance(phase)
            for tensor_target, tensor, key in (
                (
                    self.plan.artifacts.single_path("initial_noise"),
                    result.initial_noise,
                    "initial_noise",
                ),
                (
                    self.plan.artifacts.single_path("final_latents"),
                    result.final_latents,
                    "final_latents",
                ),
            ):
                if tensor_target is None:
                    continue
                if tensor is None:
                    raise BundleOutputError(f"Krea result has no requested {key} tensor")
                if checkpoint is not None:
                    checkpoint()
                mx.eval(tensor)
                peer = transaction / tensor_target.name
                tensor_metadata = {"schema": "kinomlx.krea2.tensor.v1", "artifact": key}
                provenance = result.metadata.get("text_conditioning")
                if isinstance(provenance, dict) and provenance.get("system_prompt") is not None:
                    tensor_metadata["system_prompt"] = json.dumps(provenance["system_prompt"])
                mx.save_safetensors(
                    str(peer),
                    {key: tensor},
                    metadata=tensor_metadata,
                )
                if not peer.is_file():
                    raise BundleOutputError(f"tensor writer did not create {peer.name}")
                temporary[tensor_target] = peer
                if publish is not None:
                    publish(temporary)
                self.reporter.phase_advance(phase)
            text_target = self.plan.artifacts.single_path("text_conditioning")
            if text_target is not None:
                if result.text_conditioning is None:
                    raise BundleOutputError(
                        "Krea result has no requested text_conditioning tensors"
                    )
                if checkpoint is not None:
                    checkpoint()
                artifact = text_conditioning_artifact(result.text_conditioning)
                tensors = dict(artifact.tensors)
                mx.eval(*tensors.values())
                peer = transaction / text_target.name
                mx.save_safetensors(str(peer), tensors, metadata=dict(artifact.metadata))
                if not peer.is_file():
                    raise BundleOutputError(f"tensor writer did not create {peer.name}")
                temporary[text_target] = peer
                if publish is not None:
                    publish(temporary)
                self.reporter.phase_advance(phase)
            if checkpoint is not None:
                checkpoint()
            return temporary
        finally:
            self.reporter.phase_end(phase)

    def write(
        self,
        result: Krea2GenerationOutput,
        *,
        reservation: BundleReservation[Krea2OutputPlan] | None = None,
        checkpoint: Krea2OutputCheckpoint | None = None,
        publish_each: bool = False,
    ) -> Krea2ArtifactSet:
        """Write then publish the entire selected bundle or leave primaries untouched."""
        if len(self.completed) == len(self.plan.artifacts.images):
            return self.plan.artifacts
        owns_reservation = reservation is None
        selected = self.plan.reserve() if reservation is None else reservation
        if (
            any(target not in selected.modes for target in self.plan.targets())
            and not owns_reservation
        ):
            raise ValueError("output reservation belongs to another Krea output plan")
        if not owns_reservation and not selected.active:
            raise ValueError("passed Krea output reservation is not active")
        transaction: Path | None = None
        retain_transaction = False
        try:
            if owns_reservation:
                selected.__enter__()
            if len(self.plan.artifacts.images) > 1:
                if len(result.images) != len(self.plan.artifacts.images):
                    raise BundleOutputError(
                        "Krea result image count does not match the output plan"
                    )
                for index, image in enumerate(result.images):
                    if index in self.completed:
                        continue
                    if checkpoint is not None:
                        checkpoint()
                    single = Krea2GenerationOutput(
                        images=(image,),
                        metadata=result.metadata,
                        initial_noise=None
                        if result.initial_noise is None
                        else result.initial_noise[index : index + 1],
                        final_latents=None
                        if result.final_latents is None
                        else result.final_latents[index : index + 1],
                        text_conditioning=None
                        if result.text_conditioning is None
                        else select_conditioning(result.text_conditioning, (index,)),
                    )
                    self.write_image(index, single, reservation=selected)
                if owns_reservation:
                    selected.complete()
                return self.plan.artifacts
            transaction = Path(tempfile.mkdtemp(prefix=".krea2-output-", dir=self.plan.directory))

            def publish(artifacts: dict[Path, Path]) -> None:
                fingerprints, errors = self._fingerprint_temporaries(artifacts)
                publish_staged(artifacts, selected.modes, transaction)
                self.published.update(artifacts)
                self.fingerprints.update(fingerprints)
                self.fingerprint_errors.update(errors)
                artifacts.clear()

            temporary = self._write_temporaries(
                result, transaction, checkpoint, publish if publish_each else None
            )
            fingerprints, fingerprint_errors = self._fingerprint_temporaries(temporary)
            publish_staged(temporary, selected.modes, transaction)
            self.published.update(temporary)
            self.fingerprints.update(fingerprints)
            self.fingerprint_errors.update(fingerprint_errors)
            if owns_reservation:
                selected.complete()
            return self.plan.artifacts
        except RetainedTransactionError:
            retain_transaction = True
            raise
        except BundleOutputError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise BundleOutputError(f"cannot write Krea outputs: {exc}") from exc
        finally:
            if owns_reservation:
                selected.__exit__(None, None, None)
            if transaction is not None and not retain_transaction:
                try:
                    shutil.rmtree(transaction)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    _log.warning(
                        "could not remove Krea output transaction %s: %s", transaction, exc
                    )


__all__ = [
    "Krea2ArtifactSet",
    "Krea2OutputPlan",
    "Krea2OutputSink",
    "Krea2StillOutputError",
    "Krea2StillOutputPlan",
    "Krea2StillOutputReservation",
    "Krea2StillOutputSink",
    "plan_krea2_output",
    "publish_krea2_still",
]
