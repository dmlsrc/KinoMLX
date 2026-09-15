"""Pure shared output-selection policies owned by configuration fields."""

from collections.abc import Mapping

from kinomlx.config_rules import RecordRule


def _exact_replacement(values: Mapping[str, object]) -> None:
    if values["force"] and values["path"] is None:
        raise ValueError("force requires an exact output path")


EXACT_REPLACEMENT = RecordRule(
    "exact_replacement", frozenset({"force", "path"}), _exact_replacement
)
