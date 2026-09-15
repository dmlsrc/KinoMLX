"""Model-neutral field groups reused by model config contributions."""

from pathlib import Path

from kinomlx.config import ConfigFieldSpec, ConfigTableSpec
from kinomlx.config.stations import StationPolicy
from kinomlx.config_surfaces import CONFIG_SURFACES, SurfacePolicy
from kinomlx.settings import Settings

_INERT_ADAPTER_CHECK_REASON = "No adapters or denoising stage on this model; the check is inert."
_INERT_ADAPTER_CHECK = SurfacePolicy((("cli", _INERT_ADAPTER_CHECK_REASON),))


def _mutable(stations: tuple[str, ...], reason: str) -> tuple[StationPolicy, ...]:
    return tuple(StationPolicy(station, "mutable", reason=reason) for station in stations)


def _adapter_check(
    stations: tuple[str, ...], applicable: frozenset[str]
) -> tuple[StationPolicy, ...]:
    return tuple(
        StationPolicy(
            station,
            "mutable" if station in applicable else "irrelevant",
            reason=(
                "Adapter validation applies to the repeated denoising stage."
                if station in applicable
                else "No adapter weights are checked at this restart station."
            ),
            irrelevant_action=None if station in applicable else "ignore",
        )
        for station in stations
    )


def settings_table(
    *,
    stations: tuple[str, ...],
    adapter_weight_check_stations: frozenset[str],
) -> ConfigTableSpec:
    """Declare shared settings, binding policy at each field for the model's stations."""
    if not adapter_weight_check_stations <= set(stations):
        raise ValueError("settings need applicable adapter-check stations")
    return ConfigTableSpec(
        path=("settings",),
        cli_generated=True,
        cli_group="Infrastructure settings (override env vars)",
        record=Settings,
        stations=stations,
        station_record=Settings if stations else None,
        fields=(
            ConfigFieldSpec(
                persistence="intent",
                name="app_cache_dir",
                purpose="Set the manual converted-weight and native-build cache directory",
                valid="a directory path",
                example=Path("~/Library/Caches/KinoMLX"),
                cli_dest="app_cache_dir",
                default_text='"~/Library/Caches/KinoMLX"',
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations,
                    "The cache root locates resources; replay validates required component identity.",
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="check_adapter_weights",
                purpose="Enable one-time adapter weight checks after the first denoise step",
                valid="true or false; latent checks remain enabled",
                example=True,
                cli_dest="check_adapter_weights",
                surfaces=CONFIG_SURFACES if adapter_weight_check_stations else _INERT_ADAPTER_CHECK,
                station_policies=_adapter_check(stations, adapter_weight_check_stations),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="hf_home",
                purpose="Set the Hugging Face state root",
                valid="a directory path",
                example=Path("~/.cache/huggingface"),
                cli_dest="hf_home",
                default_text='"~/.cache/huggingface"',
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations,
                    "The Hugging Face cache root locates resources; replay validates required identity.",
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="verbose",
                purpose="Enable verbose logs",
                valid="true or false",
                example=False,
                cli_dest="verbose",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "Verbose presentation applies to the new run only."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="quiet",
                purpose="Disable live presentation",
                valid="true or false",
                example=False,
                cli_dest="quiet",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "Quiet presentation applies to the new run only."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="json_output",
                purpose="Emit machine-readable standard output",
                valid="true or false",
                example=False,
                cli_dest="json_output",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "Machine-readable presentation applies to the new run only."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="profile_signposts",
                purpose="Emit native profiling signposts",
                valid="true or false",
                example=False,
                cli_dest="profile_signposts",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "Profiling instrumentation applies to the new run only."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="profile_signpost_log",
                purpose="Save profiling signposts to a log",
                valid="a path when profiling signposts are enabled",
                example=Path("kinomlx-signposts.log"),
                cli_dest="profile_signpost_log",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "The new run may write profiling signposts to another log."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="mlx_cache_limit_gb",
                purpose="Set the MLX cache limit in GiB",
                valid="a finite number greater than or equal to 0",
                example=1.0,
                cli_dest="mlx_cache_limit_gb",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "The allocator cache limit controls execution of the new run."
                ),
            ),
            ConfigFieldSpec(
                persistence="intent",
                name="mlx_wired_limit_gb",
                purpose="Keep MLX buffers resident up to this many GiB",
                valid=(
                    "a finite number from 0 to the device wired limit; 0 leaves buffers "
                    "unwired; unset uses the device's recommended working set"
                ),
                example=60.0,
                cli_dest="mlx_wired_limit_gb",
                surfaces=CONFIG_SURFACES,
                station_policies=_mutable(
                    stations, "The wired-memory limit controls execution of the new run."
                ),
            ),
        ),
    )


__all__ = ["settings_table"]
