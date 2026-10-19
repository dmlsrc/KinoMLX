"""Typed Krea 2 restart selection and prior-run artifact resolution."""

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kinomlx.config import ConfigError, ModelConfigSpec
from kinomlx.config.persistence import rebase_configuration
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stages import stage_output_defaults
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record
from kinomlx.image_restart_config import (
    IMAGE_CONSUMES,
    ImageRestartArtifactFields,
    ImageRestartSelectionFields,
    ImageStage,
)
from kinomlx.prompts import is_empty_prompt_batch
from kinomlx.restart_receipts import input_receipt, read_restart_run, restart_object
from kinomlx.restart_selection import (
    CANCELED_RUN_STATUS,
    PLANNED_IMAGE_COUNT,
    SEED_OUTPUTS,
    fingerprint_key,
    resolve_restart_index,
    selected_artifact,
    source_image_count,
    unpublished_image,
)

from .output_policies import OUTPUT_POLICIES

_LOG = logging.getLogger(__name__)


@config_record
@dataclass(frozen=True)
class RestartConfig(ImageRestartSelectionFields):
    """Select a prior Krea run and its first repeated stage."""


@config_record
@dataclass(frozen=True)
class Krea2Restart(ImageRestartArtifactFields):
    """One public, typed entry point into the Krea stage graph."""

    @classmethod
    def decode(cls, latents: Path | str) -> Krea2Restart:
        """Decode a saved final-latent product."""
        return cls(latents=Path(latents), stage=ImageStage.DECODE)

    @classmethod
    def denoise(cls, text_conditioning: Path | str) -> Krea2Restart:
        """Resume from saved Qwen output at the single Krea denoising stage."""
        return cls(stage=ImageStage.DENOISE, text_conditioning=Path(text_conditioning))


@dataclass(frozen=True)
class RestartManifest:
    """Validated prior-run configuration and selected stage inputs."""

    config: RestartConfig
    source_index: int
    source_image_count: int
    source_run: Path
    source_status: str | None
    source_model: str
    source_schema_version: object
    source_variant: str | None
    source_vae_decoder: str | None
    final_latents: Path | None
    text_conditioning: Path | None
    source_outputs: tuple[Path, ...]
    source_output_fingerprints: dict[str, str]
    configuration: PreparedConfig
    generation: dict[str, object]

    @property
    def selected_latents(self) -> Path:
        """Return the final-latent artifact consumed by this restart."""
        selected = self.config.latents if self.config.latents is not None else self.final_latents
        if selected is None:
            raise ConfigError(
                f"restart run {self.source_run} has no completed final_latents output"
            )
        return selected

    @property
    def selected_text_conditioning(self) -> Path:
        """Return the saved Qwen output consumed by a denoise restart."""
        if self.text_conditioning is None:
            raise ConfigError(
                f"restart run {self.source_run} has no completed text_conditioning output; "
                "supply --text-conditioning or use a run saved with that sidecar"
            )
        return self.text_conditioning

    def _selected_image(self) -> dict[str, object]:
        """Narrow a multi-image source run to the one image this restart resumes.

        The inherited prompt batch is already one row per image, and the seed
        list would otherwise multiply it again, so both collapse to the selected
        row before the request is typed.
        """
        index = self.source_index
        prompts = self.generation.get("prompts")
        if not isinstance(prompts, list) or index >= len(prompts):
            raise ConfigError(
                f"restart run {self.source_run} produced {self.source_image_count} images but "
                "records no per-image prompts; supply --prompt and --seed for the selected image"
            )
        selected: dict[str, object] = {
            "prompt": prompts[index],
            "images_per_prompt": 1,
            "seeds": None,
        }
        negatives = self.generation.get("negative_prompts")
        if isinstance(negatives, list) and index < len(negatives):
            selected["negative_prompt"] = negatives[index]
        noise = self.generation.get("noise")
        seeds = noise.get("per_image_seeds") if isinstance(noise, dict) else None
        if isinstance(seeds, list) and index < len(seeds) and isinstance(seeds[index], int):
            selected["seed"] = seeds[index]
        return selected

    def base_config(self) -> PreparedConfig:
        """Return the prior invocation as a safe lower-precedence config layer."""
        base = copy.deepcopy(self.configuration.values)
        base.pop("restart", None)
        base["operation"] = {"dry_run": False}

        raw_model_settings = base.get("model_settings")
        model_settings: dict[str, object] = (
            {} if not isinstance(raw_model_settings, dict) else dict(raw_model_settings)
        )
        if self.source_variant is not None:
            model_settings["variant"] = self.source_variant
        if self.source_vae_decoder is not None:
            model_settings["vae_decoder"] = self.source_vae_decoder
        base["model_settings"] = model_settings

        raw_generate = base.get("generate")
        generate: dict[str, object] = (
            {} if not isinstance(raw_generate, dict) else dict(raw_generate)
        )
        if self.config.stage == "denoise":
            if self.text_conditioning is not None:
                generate["text_conditioning"] = self.text_conditioning
        else:
            generate["text_conditioning"] = None
            if generate.get("prompt") is None and "prompts" in self.generation:
                # Generation metadata contains the final expanded batch, including
                # prompts inherited from a text-conditioning replay.
                generate["prompt"] = self.generation["prompts"]
                if is_empty_prompt_batch(self.generation["prompts"]):
                    generate["prompt_unconditional"] = True
                generate["negative_prompt"] = self.generation.get("negative_prompts")
                generate["images_per_prompt"] = 1
        if self.source_image_count > 1:
            generate.update(self._selected_image())
        base["generate"] = generate

        raw_output = base.get("output")
        output: dict[str, object] = {} if not isinstance(raw_output, dict) else dict(raw_output)
        source_stem = self.source_run.stem.removesuffix("_run")
        output.update(
            {
                "path": None,
                "directory": self.source_run.parent,
                "prefix": (
                    f"{source_stem}_final_decode"
                    if self.config.stage == "decode"
                    else f"{source_stem}_denoise"
                ),
            }
        )
        if self.config.stage == "decode":
            output.update(
                {
                    "save_all_sidecars": False,
                }
            )
        base["output"] = stage_output_defaults(output, OUTPUT_POLICIES, stage=self.config.stage)
        return rebase_configuration(self.configuration, base)

    def to_record(
        self,
        *,
        text_conditioning: Path | None = None,
    ) -> dict[str, object]:
        """Return parent-run provenance and observational input identity."""
        if self.config.stage == "decode":
            inputs = {
                "final_latents": input_receipt(
                    self.selected_latents,
                    name="final_latents",
                    expected_sha256=self.source_output_fingerprints.get("final_latents"),
                    expected_metadata=_expected_metadata("final_latents"),
                )
            }
        else:
            selected_text = (
                self.selected_text_conditioning
                if text_conditioning is None
                else text_conditioning.expanduser().absolute()
            )
            inputs = {
                "text_conditioning": input_receipt(
                    selected_text,
                    name="text_conditioning",
                    expected_sha256=self.source_output_fingerprints.get("text_conditioning"),
                    expected_metadata=_expected_metadata("text_conditioning"),
                )
            }
        return {
            "source_run": str(self.source_run),
            "source_status": self.source_status,
            "source_schema_version": self.source_schema_version,
            "source_model": self.source_model,
            "source_variant": self.source_variant,
            "source_vae_decoder": self.source_vae_decoder,
            "stage": self.config.stage,
            "inputs": inputs,
            "identity_mismatch": any(
                receipt.get("matches_parent") is False for receipt in inputs.values()
            ),
            "metadata_mismatch": any(
                bool(receipt.get("metadata_mismatches")) for receipt in inputs.values()
            ),
        }


def _expected_metadata(name: str) -> dict[str, str]:
    if name == "final_latents":
        return {
            "schema": "kinomlx.krea2.tensor.v1",
            "artifact": "final_latents",
        }
    if name == "text_conditioning":
        return {"artifact": "krea2_text_conditioning"}
    return {}


def _artifact_path(raw: object, *, label: str, source: Path) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"restart run {source}: {label} must be a path string")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else source.parent / path).absolute()


def _published_seed_outputs(
    outputs: Mapping[str, object], source: Path
) -> dict[int, Mapping[str, object]]:
    """Index a canceled run's per-image records by the image each one names.

    A canceled run has no flat per-artifact lists; it records one entry per
    image it finished. Entries written before those entries carried their own
    index are read in the order the run published them, which is the order it
    generates in.
    """
    entries = outputs.get(SEED_OUTPUTS)
    if not isinstance(entries, list):
        return {}
    published: dict[int, Mapping[str, object]] = {}
    for position, raw in enumerate(entries):
        entry = restart_object(raw, label=f"outputs.{SEED_OUTPUTS}[{position}]", source=source)
        declared = entry.get("index")
        published[declared if type(declared) is int else position] = entry
    return published


def _source_artifacts(
    payload: Mapping[str, object],
    outputs: Mapping[str, object],
    config: RestartConfig,
    source: Path,
    *,
    index: int,
    count: int,
    published: Mapping[str, object] | None = None,
) -> tuple[dict[str, Path | None], dict[str, str], tuple[Path, ...]]:
    """Prefer produced artifacts, then the immediate parent's consumed inputs.

    A multi-image source run records one artifact path per image, so ``index``
    selects this restart's image before any path is resolved. ``published`` is
    the selected image's own record when the run was canceled and therefore
    kept no flat lists.
    """
    restart = payload.get("restart")
    raw_inputs = restart.get("inputs") if isinstance(restart, dict) else None
    inputs = (
        {}
        if raw_inputs is None
        else restart_object(raw_inputs, label="restart.inputs", source=source)
    )
    fingerprints = _source_output_fingerprints(payload, source, index=index, count=count)
    artifacts: dict[str, Path | None] = {}
    input_paths: list[Path] = []
    (selected_name,) = IMAGE_CONSUMES[config.stage]
    for name in ("final_latents", "text_conditioning"):
        label = f"outputs.{name}"
        try:
            entry = selected_artifact(outputs.get(name), index=index, label=label)
        except ValueError as exc:
            raise ConfigError(f"restart run {source}: {exc}") from exc
        if entry is None and published is not None:
            entry = published.get(name)
            label = f"outputs.{SEED_OUTPUTS}.{name}"
        produced = _artifact_path(entry, label=label, source=source)
        raw_receipt = inputs.get(name)
        receipt = (
            {}
            if raw_receipt is None
            else restart_object(raw_receipt, label=f"restart.inputs.{name}", source=source)
        )
        consumed = _artifact_path(
            receipt.get("path"), label=f"restart.inputs.{name}.path", source=source
        )
        if consumed is not None:
            input_paths.append(consumed)
        artifacts[name] = produced if produced is not None else consumed
        if produced is None and consumed is not None:
            fingerprints.pop(name, None)
            fingerprints.update(
                _source_output_fingerprints(
                    {"output_fingerprints": {name: receipt.get("sha256")}}, source
                )
            )
            if name == selected_name and config.latents is None:
                _LOG.info("Restart uses recorded input %s from %s: %s", name, source, consumed)
    return artifacts, fingerprints, tuple(input_paths)


def _source_output_paths(outputs: Mapping[str, object], source: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in outputs.values():
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value).expanduser()
            paths.append((path if path.is_absolute() else source.parent / path).absolute())
    return tuple(dict.fromkeys(paths))


def _source_variant(
    payload: Mapping[str, object],
    invocation: Mapping[str, object],
) -> str | None:
    generation = payload.get("generation")
    if isinstance(generation, dict):
        value = generation.get("variant")
        if isinstance(value, str):
            return value
    settings = invocation.get("model_settings")
    if isinstance(settings, dict):
        value = settings.get("variant")
        if isinstance(value, str):
            return value
    return None


def _source_vae_decoder(
    payload: Mapping[str, object],
    invocation: Mapping[str, object],
) -> str | None:
    generation = payload.get("generation")
    if isinstance(generation, dict):
        decode = generation.get("decode")
        if isinstance(decode, dict):
            value = decode.get("vae_decoder")
            if isinstance(value, str):
                return value
    settings = invocation.get("model_settings")
    if isinstance(settings, dict):
        value = settings.get("vae_decoder")
        if isinstance(value, str):
            return value
    return None


def _source_output_fingerprints(
    payload: Mapping[str, object],
    source: Path,
    *,
    index: int = 0,
    count: int = 1,
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
    result: dict[str, str] = {}
    for name in ("final_latents", "text_conditioning"):
        fingerprint = raw.get(fingerprint_key(name, index=index, count=count))
        if not isinstance(fingerprint, str):
            continue
        normalized = fingerprint.lower()
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
        result[name] = f"sha256:{digest}"
    return result


def load_restart_manifest(config: RestartConfig, *, schema: ModelConfigSpec) -> RestartManifest:
    """Load one KinoMLX Krea run JSON and resolve its stage artifacts."""
    validate_record(config, location="[restart]")
    run_log = read_restart_run(config.run, schema=schema)
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

    override = None if config.latents is None else config.latents.expanduser().absolute()
    try:
        count = source_image_count(outputs, planned=payload.get(PLANNED_IMAGE_COUNT))
        index = resolve_restart_index(config.index, count)
    except ValueError as exc:
        raise ConfigError(f"restart run {source}: {exc}") from exc
    published = None
    if status == CANCELED_RUN_STATUS:
        # A canceled run is restartable for every image it finished, and for
        # none of the ones it never reached. A failed run is not read this way
        # at all: it stopped for a reason it cannot describe.
        by_index = _published_seed_outputs(outputs, source)
        if index not in by_index:
            raise ConfigError(f"restart run {source}: {unpublished_image(index, count)}")
        published = by_index[index]
    artifacts, fingerprints, input_paths = _source_artifacts(
        payload, outputs, config, source, index=index, count=count, published=published
    )
    manifest = RestartManifest(
        config=RestartConfig(run=source, stage=config.stage, latents=override, index=config.index),
        source_index=index,
        source_image_count=count,
        source_run=source,
        source_status=status,
        source_model=model,
        source_schema_version=payload.get("schema_version"),
        source_variant=_source_variant(payload, effective),
        source_vae_decoder=_source_vae_decoder(payload, effective),
        final_latents=artifacts["final_latents"],
        text_conditioning=artifacts["text_conditioning"],
        source_outputs=tuple(dict.fromkeys((*_source_output_paths(outputs, source), *input_paths))),
        source_output_fingerprints=fingerprints,
        configuration=configuration,
        generation=restart_object(payload.get("generation", {}), label="generation", source=source),
    )
    if config.stage == "decode":
        selected = manifest.selected_latents
        if not selected.is_file():
            raise ConfigError(f"restart latent artifact does not exist: {selected}")
    return manifest


__all__ = [
    "Krea2Restart",
    "RestartConfig",
    "RestartManifest",
    "load_restart_manifest",
]
