"""The scalar types a configuration value may take on the wire.

Configuration makes a round trip: a CLI token or TOML scalar becomes a typed
field, and a typed field becomes a TOML scalar again. Three layers take part.
``config_bindings`` decides what primitive argparse should parse into,
``config_values.coerce_config_value`` turns that primitive into the declared
field type, and ``config_values.dataclass_to_mapping`` projects the field back.

Each layer answers a different question, but all three range over the same set
of transport types, so that set is declared once here rather than re-enumerated
in each. Supporting a new transport type means adding it here and teaching the
layers that need bespoke handling, instead of discovering the omission one
runtime error at a time.

``config.dump.dump_toml_value`` also accepts ``datetime``, ``date`` and ``time``,
which TOML carries natively, but ``coerce_config_value`` cannot read one back.
They are therefore not transport types here: a configuration value must survive
the whole round trip, not only the way out. No record declares such a field.
"""

from enum import StrEnum
from pathlib import Path

TRANSPORT_SCALARS: tuple[type[object], ...] = (str, int, float, bool, Path)
"""What a projected configuration may contain, beside containers and ``None``."""

CLI_PARSED_SCALARS: tuple[type[object], ...] = (str, int, float, Path)
"""What argparse can parse with ``type=``.

``bool`` is absent on purpose: a boolean field arrives through a store action or
a paired flag, never through a value parser, so a binding that asks for one is a
declaration error rather than a missing codec.
"""


def enum_vocabulary(annotation: object) -> type[StrEnum] | None:
    """Return the string vocabulary an annotation names, or None.

    A ``StrEnum`` member is a string, so it needs no special projection: it
    travels as its value. Only the inbound layers care, and they ask here so
    they cannot disagree about what counts as a vocabulary.
    """
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation
    return None


__all__ = ["CLI_PARSED_SCALARS", "TRANSPORT_SCALARS", "enum_vocabulary"]
