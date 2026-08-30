"""Typed restart selection and prior-run artifact resolution."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kinomlx.config import ConfigError
from kinomlx.io.fingerprints import file_sha256

RESTART_PHASE_CHOICES = ("decode", "stage-2")
LATENT_STAGE_CHOICES = ("final", "stage-1")
_RESTART_PHASES = frozenset(RESTART_PHASE_CHOICES)
_LATENT_STAGES = frozenset(LATENT_STAGE_CHOICES)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestartConfig:
    """User-facing selection of a prior run and its first repeated station."""

    run: Path
    phase: str = "decode"
    latent_stage: str | None = None
    latents: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.run, str):
            object.__setattr__(self, "run", Path(self.run))
        if isinstance(self.latents, str):
            object.__setattr__(self, "latents", Path(self.latents))
        if self.phase not in _RESTART_PHASES:
            valid = ", ".join(sorted(_RESTART_PHASES))
            raise ValueError(f"restart phase must be one of: {valid}")
        if self.phase == "stage-2":
            if self.latent_stage is not None:
                raise ValueError("restart latent_stage applies only when phase=decode")
            return
        latent_stage = "final" if self.latent_stage is None else self.latent_stage
        if latent_stage not in _LATENT_STAGES:
            valid = ", ".join(sorted(_LATENT_STAGES))
            raise ValueError(f"restart latent_stage must be one of: {valid}")
        object.__setattr__(self, "latent_stage", latent_stage)


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
    invocation: dict[str, object]

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
    def selected_latent_stage(self) -> str:
        """Return the source stage represented by ``selected_latents``."""
        if self.config.phase == "stage-2":
            return "stage-1"
        return self.config.latent_stage or "final"

    def base_config(self) -> dict[str, object]:
        """Return the prior invocation as a safe lower-precedence config layer."""
        base = copy.deepcopy(self.invocation)
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

        if self.config.phase == "decode":
            # Latents and prompt products are restart inputs, not newly
            # produced artifacts of a decode-only run. Current CLI/TOML flags
            # may still override these values and will then fail explicitly.
            base["model_artifacts"] = {
                "save_latents": False,
                "save_text_conditioning": False,
            }
        else:
            raw_artifacts = base.get("model_artifacts")
            model_artifacts: dict[str, object] = (
                {} if not isinstance(raw_artifacts, dict) else dict(raw_artifacts)
            )
            model_artifacts["save_text_conditioning"] = False
            base["model_artifacts"] = model_artifacts

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
        return base

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
            latent_name: _input_receipt(
                self.selected_latents,
                name=latent_name,
                expected_sha256=self.source_output_fingerprints.get(latent_name),
            )
        }
        if self.config.phase == "stage-2":
            if text_conditioning is None:
                raise ConfigError("stage-2 restart requires a text-conditioning sidecar")
            inputs["text_conditioning"] = _input_receipt(
                text_conditioning,
                name="text_conditioning",
                expected_sha256=self.source_output_fingerprints.get("text_conditioning"),
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


def _input_receipt(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None,
) -> dict[str, object]:
    """Inspect one consumed sidecar without turning identity into a load gate."""
    source = path.expanduser().absolute()
    if not source.is_file():
        raise ConfigError(f"restart {name} artifact does not exist: {source}")

    hash_error = None
    try:
        actual_sha256 = file_sha256(source)
    except OSError as exc:
        actual_sha256 = None
        hash_error = f"{type(exc).__name__}: {exc}"
        _LOG.warning(
            "Could not fingerprint restart input %s: %s; continuing to structural loading",
            source,
            exc,
        )

    matches_parent = None
    if expected_sha256 is not None and actual_sha256 is not None:
        matches_parent = actual_sha256 == expected_sha256
        if not matches_parent:
            _LOG.warning(
                "Restart input %s differs from the parent receipt "
                "(expected %s, got %s); continuing because identity is observational",
                source,
                expected_sha256,
                actual_sha256,
            )

    declared_metadata: dict[str, str] = {}
    metadata_mismatches: dict[str, dict[str, str | None]] = {}
    metadata_error = None
    try:
        # Keep CLI config/help importable without the MLX runtime. Header
        # inspection is needed only once execution resolves a restart receipt.
        from kinomlx.io.safetensors import read_metadata

        metadata = read_metadata(source)
    except (OSError, ValueError) as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"
    else:
        expected_metadata = _expected_metadata(name)
        declared_metadata = {key: metadata[key] for key in expected_metadata if key in metadata}
        metadata_mismatches = {
            key: {"expected": expected, "declared": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if metadata_mismatches:
            _LOG.warning(
                "Restart input %s has advisory metadata differences %s; "
                "continuing to structural tensor validation",
                source,
                ", ".join(sorted(metadata_mismatches)),
            )

    return {
        "path": str(source),
        "sha256": actual_sha256,
        "parent_sha256": expected_sha256,
        "matches_parent": matches_parent,
        "hash_error": hash_error,
        "declared_metadata": declared_metadata,
        "metadata_mismatches": metadata_mismatches,
        "metadata_error": metadata_error,
    }


def _object(value: object, *, label: str, source: Path) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"restart run {source}: {label} must be an object")
    return dict(cast(Mapping[str, object], value))


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
    source = config.run.expanduser().absolute()
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read restart run {source}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid restart run JSON {source}: {exc}") from exc
    payload = _object(decoded, label="root", source=source)
    invocation = _object(payload.get("invocation"), label="invocation", source=source)
    outputs = _object(payload.get("outputs"), label="outputs", source=source)
    model = payload.get("model", invocation.get("model"))
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
        source_model_generation=_source_generation(payload, invocation),
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
        invocation=invocation,
    )
    selected = manifest.selected_latents
    if not selected.is_file():
        raise ConfigError(f"restart latent artifact does not exist: {selected}")
    return manifest


__all__ = [
    "LATENT_STAGE_CHOICES",
    "RESTART_PHASE_CHOICES",
    "RestartConfig",
    "RestartManifest",
    "load_restart_manifest",
]
