"""Seed-selection precedence and canonical configuration serialization."""


def seed_config(values: dict[str, object]) -> dict[str, object]:
    if values.get("seeds") is not None:
        values = dict(values)
        values.pop("seed", None)
    return values
