"""Typed LTX-2 restart selection and prior-run artifact resolution."""

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from kinomlx.config import ConfigError
from kinomlx.config.persistence import rebase_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stations import station_output_defaults
from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record
from kinomlx.restart_receipts import input_receipt, read_restart_run

from .artifacts import OUTPUT_POLICIES
from .restart_rules import (
    LATENT_STAGE_CHOICES,
    RESTART_ARTIFACTS,
    RESTART_PHASE_CHOICES,
    RESTART_SELECTION,
    LatentStage,
    RestartPhase,
    resolved_latent_stage,
)

_LOG = logging.getLogger(__name__)


@config_record
@dataclass(frozen=True)
class RestartConfig:
    """User-facing selection of a prior run and its first repeated station."""

    run: Path
    phase: RestartPhase = field(
        default="decode",
        metadata={
            "config_constraints": (ChoiceConstraint(RESTART_PHASE_CHOICES),),
            "config_rules": (RESTART_SELECTION,),
        },
    )
    latent_stage: LatentStage | None = field(
        default=None, metadata={"config_constraints": (ChoiceConstraint(LATENT_STAGE_CHOICES),)}
    )
    latents: Path | None = None


@config_record
@dataclass(frozen=True)
class DistilledRestart:
    """One public, typed entry point into the distilled station graph."""

    latents: Path
    phase: RestartPhase = field(
        default="decode",
        metadata={
            "config_constraints": (ChoiceConstraint(RESTART_PHASE_CHOICES),),
            "config_rules": (RESTART_ARTIFACTS,),
        },
    )
    latent_stage: LatentStage | None = field(
        default=None, metadata={"config_constraints": (ChoiceConstraint(LATENT_STAGE_CHOICES),)}
    )
    text_conditioning: Path | None = None
    source_model_generation: str | None = None

    @property
    def selected_latent_stage(self) -> LatentStage:
        return resolved_latent_stage(self.phase, self.latent_stage)

    @classmethod
    def decode(
        cls,
        latents: Path | str,
        *,
        latent_stage: LatentStage = "final",
        source_model_generation: str | None = None,
    ) -> DistilledRestart:
        """Select direct decoding of a final or stage-1 latent product."""
        return cls(
            latents=Path(latents),
            phase="decode",
            latent_stage=latent_stage,
            source_model_generation=source_model_generation,
        )

    @classmethod
    def stage_2(
        cls,
        latents: Path | str,
        *,
        text_conditioning: Path | str,
        source_model_generation: str | None = None,
    ) -> DistilledRestart:
        """Select stage 2 from a saved stage-1 latent and text product."""
        return cls(
            latents=Path(latents),
            phase="stage-2",
            latent_stage="stage-1",
            text_conditioning=Path(text_conditioning),
            source_model_generation=source_model_generation,
        )


@dataclass(frozen=True)
class RestartManifest:
    """Validated prior-run configuration and selected checkpoint inputs."""

    config: RestartConfig
    source_run: Path
    source_status: str | None
    source_model: str
    source_schema_version: object
    source_model_generation: str | None
    source_video_shape: tuple[int, int, int, int, int] | None
    source_video: Path | None
    stage_1_latents: Path | None
    final_latents: Path | None
    text_conditioning: Path | None
    source_output_fingerprints: dict[str, str]
    configuration: PreparedConfig

    @property
    def selected_latents(self) -> Path:
        """Return the latent artifact consumed by this restart selection."""
        selected = self.config.latents
        if selected is None:
            selected = (
                self.stage_1_latents
                if self.config.phase == "stage-2" or self.config.latent_stage == "stage-1"
                else self.final_latents
            )
        if selected is None:
            artifact = (
                "stage_1_latents"
                if self.config.phase == "stage-2" or self.config.latent_stage == "stage-1"
                else "final_latents"
            )
            raise ConfigError(f"restart run {self.source_run} has no completed {artifact} output")
        return selected

    @property
    def selected_latent_stage(self) -> LatentStage:
        """Return the source stage represented by ``selected_latents``."""
        return resolved_latent_stage(self.config.phase, self.config.latent_stage)

    def base_config(self) -> PreparedConfig:
        """Return the prior invocation as a safe lower-precedence config layer."""
        base = copy.deepcopy(self.configuration.values)
        base.pop("restart", None)

        raw_generate = base.get("generate")
        generate: dict[str, object] = (
            {} if not isinstance(raw_generate, dict) else dict(raw_generate)
        )
        if self.config.phase == "stage-2" and self.text_conditioning is not None:
            generate["text_conditioning"] = self.text_conditioning
        if self.source_video_shape is not None:
            batch, channels, frames, height, width = self.source_video_shape
            if (batch, channels) != (1, 3):
                raise ConfigError(
                    f"restart run {self.source_run} has unsupported video shape "
                    f"{self.source_video_shape}"
                )
            generate.update(
                {
                    "frames": frames,
                    "height": height,
                    "width": width,
                    "duration": None,
                }
            )
        base["generate"] = generate

        raw_artifacts = base.get("model_artifacts", {})
        model_artifacts = cast(dict[str, object], raw_artifacts)
        base["model_artifacts"] = station_output_defaults(
            model_artifacts, OUTPUT_POLICIES, station=self.config.phase
        )

        raw_output = base.get("output")
        output: dict[str, object] = {} if not isinstance(raw_output, dict) else dict(raw_output)
        source_video = self.source_video
        source_stem = (
            source_video.stem
            if source_video is not None
            else self.source_run.stem.removesuffix("_run")
        )
        if self.config.phase == "stage-2":
            suffix = "stage2"
        else:
            suffix = f"{self.selected_latent_stage.replace('-', '')}_decode"
        output.update(
            {
                "path": None,
                "directory": (
                    source_video.parent if source_video is not None else self.source_run.parent
                ),
                "prefix": f"{source_stem}_{suffix}",
            }
        )
        base["output"] = output
        return rebase_configuration(self.configuration, base)

    def to_record(
        self,
        *,
        text_conditioning: Path | None,
        decoder_seed: int,
    ) -> dict[str, object]:
        """Return parent-run provenance and observational input identities."""
        latent_name = (
            "stage_1_latents" if self.selected_latent_stage == "stage-1" else "final_latents"
        )
        inputs: dict[str, dict[str, object]] = {
            latent_name: input_receipt(
                self.selected_latents,
                name=latent_name,
                expected_sha256=self.source_output_fingerprints.get(latent_name),
                expected_metadata=_expected_metadata(latent_name),
            )
        }
        if self.config.phase == "stage-2":
            if text_conditioning is None:
                raise ConfigError("stage-2 restart requires a text-conditioning sidecar")
            inputs["text_conditioning"] = input_receipt(
                text_conditioning,
                name="text_conditioning",
                expected_sha256=self.source_output_fingerprints.get("text_conditioning"),
                expected_metadata=_expected_metadata("text_conditioning"),
            )
        return {
            "source_run": str(self.source_run),
            "source_status": self.source_status,
            "source_schema_version": self.source_schema_version,
            "source_model": self.source_model,
            "source_model_generation": self.source_model_generation,
            "phase": self.config.phase,
            "latent_stage": self.selected_latent_stage,
            "decoder_seed": decoder_seed,
            "inputs": inputs,
            "identity_mismatch": any(
                record.get("matches_parent") is False for record in inputs.values()
            ),
            "metadata_mismatch": any(
                bool(record.get("metadata_mismatches")) for record in inputs.values()
            ),
        }


def _expected_metadata(name: str) -> dict[str, str]:
    if name == "stage_1_latents":
        return {"pipeline": "distilled_two_stage", "stage": "1", "final": "false"}
    if name == "final_latents":
        return {"pipeline": "distilled_two_stage", "stage": "2", "final": "true"}
    if name == "text_conditioning":
        return {"artifact": "ltx2_text_conditioning"}
    return {}


def _artifact_path(raw: object, *, name: str, source: Path) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"restart run {source}: outputs.{name} must be a path string")
    path = Path(raw).expanduser()
    candidates = (path, source.parent / path.name) if not path.is_absolute() else (path,)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return candidates[0].absolute()


def _source_generation(
    payload: Mapping[str, object],
    invocation: Mapping[str, object],
) -> str | None:
    generation = payload.get("generation")
    if isinstance(generation, dict):
        value = generation.get("model_generation")
        if isinstance(value, str):
            return value
    settings = invocation.get("model_settings")
    if isinstance(settings, dict):
        value = settings.get("model_generation")
        if isinstance(value, str):
            return value
    return None


def _source_video_shape(
    payload: Mapping[str, object],
    source: Path,
) -> tuple[int, int, int, int, int] | None:
    generation = payload.get("generation")
    if not isinstance(generation, dict) or generation.get("video_shape") is None:
        return None
    raw = generation["video_shape"]
    if (
        not isinstance(raw, list)
        or len(raw) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in raw)
    ):
        raise ConfigError(f"restart run {source}: generation.video_shape is invalid")
    return cast(tuple[int, int, int, int, int], tuple(raw))


def _source_output_fingerprints(
    payload: Mapping[str, object],
    source: Path,
) -> dict[str, str]:
    raw = payload.get("output_fingerprints")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _LOG.warning(
            "Restart run %s has an invalid advisory output_fingerprints receipt; ignoring it",
            source,
        )
        return {}
    fingerprints: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str):
            _LOG.warning(
                "Restart run %s has a malformed advisory output fingerprint; ignoring it",
                source,
            )
            continue
        normalized = value.lower()
        if len(normalized) == 64 and all(
            character in "0123456789abcdef" for character in normalized
        ):
            normalized = f"sha256:{normalized}"
        digest = normalized.removeprefix("sha256:")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            _LOG.warning(
                "Restart run %s has an invalid advisory fingerprint for %s; ignoring it",
                source,
                name,
            )
            continue
        fingerprints[name] = f"sha256:{digest}"
    return fingerprints


def load_restart_manifest(config: RestartConfig) -> RestartManifest:
    """Load one KinoMLX run JSON and resolve only confirmed output artifacts."""
    validate_record(config, location="[restart]")
    run_log = read_restart_run(config.run, model="ltx2")
    source = run_log.source
    payload = run_log.payload
    configuration = run_log.configuration
    effective = run_log.effective
    outputs = run_log.outputs
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ConfigError(f"restart run {source}: model must be a string")

    status = payload.get("status")
    if status is not None and not isinstance(status, str):
        raise ConfigError(f"restart run {source}: status must be a string")

    manifest = RestartManifest(
        config=RestartConfig(
            run=source,
            phase=config.phase,
            latent_stage=config.latent_stage,
            latents=(None if config.latents is None else config.latents.expanduser().absolute()),
        ),
        source_run=source,
        source_status=status,
        source_model=model,
        source_schema_version=payload.get("schema_version"),
        source_model_generation=_source_generation(payload, effective),
        source_video_shape=_source_video_shape(payload, source),
        source_video=_artifact_path(outputs.get("video"), name="video", source=source),
        stage_1_latents=_artifact_path(
            outputs.get("stage_1_latents"),
            name="stage_1_latents",
            source=source,
        ),
        final_latents=_artifact_path(
            outputs.get("final_latents"),
            name="final_latents",
            source=source,
        ),
        text_conditioning=_artifact_path(
            outputs.get("text_conditioning"),
            name="text_conditioning",
            source=source,
        ),
        source_output_fingerprints=_source_output_fingerprints(payload, source),
        configuration=configuration,
    )
    selected = manifest.selected_latents
    if not selected.is_file():
        raise ConfigError(f"restart latent artifact does not exist: {selected}")
    return manifest


__all__ = [
    "LATENT_STAGE_CHOICES",
    "RESTART_PHASE_CHOICES",
    "DistilledRestart",
    "LatentStage",
    "RestartConfig",
    "RestartManifest",
    "RestartPhase",
    "load_restart_manifest",
]
