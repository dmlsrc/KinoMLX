"""Model-neutral field groups reused by model config contributions."""

from __future__ import annotations

from pathlib import Path

from kinomlx.config import ConfigFieldSpec, ConfigTableSpec
from kinomlx.settings import CACHE_MODE_CHOICES, Settings


def _build_settings_table() -> ConfigTableSpec:
    return ConfigTableSpec(
        path=("settings",),
        record=Settings,
        fields=(
            ConfigFieldSpec(
                name="cache_dir",
                purpose="Set the generated KinoMLX cache root",
                valid="a directory path",
                example=Path("~/.cache/kinomlx"),
                cli_dest="cache_dir",
                default_text='"~/.cache/kinomlx"',
            ),
            ConfigFieldSpec(
                name="cache_mode",
                purpose="Select the generated-cache policy",
                valid="",
                choices=CACHE_MODE_CHOICES,
                choice_help=(
                    ("auto", "Reuse compatible generated cache entries"),
                    ("rebuild", "Regenerate compatible cache entries before use"),
                ),
                example="auto",
                cli_dest="cache_mode",
            ),
            ConfigFieldSpec(
                name="hf_home",
                purpose="Set the Hugging Face state root",
                valid="a directory path",
                example=Path("~/.cache/huggingface"),
                cli_dest="hf_home",
                default_text='"~/.cache/huggingface"',
            ),
            ConfigFieldSpec(
                name="verbose",
                purpose="Enable verbose logs",
                valid="true or false",
                example=False,
                cli_dest="verbose",
            ),
            ConfigFieldSpec(
                name="quiet",
                purpose="Disable live presentation",
                valid="true or false",
                example=False,
                cli_dest="quiet",
            ),
            ConfigFieldSpec(
                name="json_output",
                purpose="Emit machine-readable standard output",
                valid="true or false",
                example=False,
                cli_dest="json_output",
            ),
            ConfigFieldSpec(
                name="profile_signposts",
                purpose="Emit native profiling signposts",
                valid="true or false",
                example=False,
                cli_dest="profile_signposts",
            ),
            ConfigFieldSpec(
                name="profile_signpost_log",
                purpose="Save profiling signposts to a log",
                valid="a path when profiling signposts are enabled",
                example=Path("kinomlx-signposts.log"),
                cli_dest="profile_signpost_log",
            ),
            ConfigFieldSpec(
                name="mlx_cache_limit_gb",
                purpose="Set the MLX cache limit in GiB",
                valid="a finite number greater than or equal to 0",
                example=1.0,
                cli_dest="mlx_cache_limit_gb",
            ),
        ),
    )


SETTINGS_TABLE = _build_settings_table()


def settings_table() -> ConfigTableSpec:
    """Return the one shared infrastructure-settings contribution."""
    return SETTINGS_TABLE


__all__ = ["SETTINGS_TABLE", "settings_table"]
