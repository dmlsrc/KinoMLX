"""Shared prompt controls and prompt-file resolution at the invocation boundary."""

from kinomlx.config import CliBinding, ConfigFieldSpec
from kinomlx.config_surfaces import CONFIG_SURFACES


def prompt_unconditional_field(*, groups: frozenset[str] = frozenset()) -> ConfigFieldSpec:
    """One shared CLI/TOML definition for all prompt-based models."""
    return ConfigFieldSpec(
        persistence="intent",
        name="prompt_unconditional",
        purpose="Generate with an empty user prompt using normal text encoding",
        valid="boolean; exclusive with nonempty prompt or prompt_file",
        example=True,
        cli_dest="prompt_unconditional",
        cli=CliBinding("Generation", action="boolean"),
        groups=groups,
        surfaces=CONFIG_SURFACES,
    )
