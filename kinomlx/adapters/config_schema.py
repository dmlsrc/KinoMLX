"""Adapter record contributions to the shared configuration and grouped CLI engine."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

from kinomlx.config import CliBinding, ConfigFieldSpec, ConfigRecordSpec
from kinomlx.config_bindings import argument_value
from kinomlx.config_constraints import ItemChoicesConstraint, field_constraints
from kinomlx.config_surfaces import CONFIG_SURFACES

from .inputs import AdapterInput, ScheduledAdapterInput
from .schedule import parse_step_strengths


def parse_categories(text: str) -> tuple[str, ...]:
    """Decode one CLI comma list; the typed field owns the accepted vocabulary."""
    if not text.strip():
        return ()
    values = tuple(item.strip() for item in text.split(","))
    if any(not item for item in values):
        raise ValueError("adapter categories must be a comma-separated list without empty items")
    return values


def adapter_strength_field(name: str, purpose: str) -> ConfigFieldSpec:
    """Declare a finite magnitude member with canonical and LoRA CLI spellings."""
    flag = name.replace("_", "-")
    return ConfigFieldSpec(
        name,
        purpose,
        "a finite number, including zero or negative values",
        0.5,
        "adapter_" + name,
        persistence="intent",
        cli=CliBinding("Adapters", flags=("--adapter-" + flag, "--lora-" + flag)),
        surfaces=CONFIG_SURFACES,
    )


_PATH = ConfigFieldSpec(
    "path",
    "Select an adapter checkpoint",
    "a nonempty literal path",
    Path("style.safetensors"),
    "adapter_path",
    persistence="intent",
    surfaces=CONFIG_SURFACES,
)
_STRENGTH = adapter_strength_field("strength", "Set this adapter's common strength")
_EXCLUDE = ConfigFieldSpec(
    "exclude",
    "Exclude this adapter's projection categories",
    "an array of model-supported categories",
    (),
    "adapter_exclude",
    persistence="intent",
    cli=CliBinding(
        "Adapters",
        flags=("--adapter-exclude", "--lora-exclude"),
        parse=argument_value(parse_categories, "adapter categories"),
    ),
    surfaces=CONFIG_SURFACES,
)
_STEPS = ConfigFieldSpec(
    "step_strengths",
    "Set this adapter's absolute strength at each denoise step",
    "a nonempty finite number array matching the final recipe step count",
    (1.0, 0.0),
    "adapter_step_strengths",
    persistence="intent",
    cli=CliBinding(
        "Adapters",
        flags=("--adapter-step-strengths", "--lora-step-strengths"),
        parse=argument_value(parse_step_strengths, "adapter step strengths"),
    ),
    surfaces=CONFIG_SURFACES,
)
_WINDOW = ConfigFieldSpec(
    "sigma_window",
    "Select this adapter's active noise-level bands",
    "nonempty comma-separated low, low:high, or low:high@strength bands",
    "0.5:1",
    "adapter_sigma_window",
    persistence="intent",
    cli=CliBinding("Adapters", flags=("--adapter-sigma-window", "--lora-sigma-window")),
    surfaces=CONFIG_SURFACES,
)


def adapter_record_spec(
    record: type[AdapterInput], *, extra_fields: tuple[ConfigFieldSpec, ...] = ()
) -> ConfigRecordSpec:
    """Bind actual model capabilities; exact typed member coverage is checked by the registry."""
    scheduled = (_STEPS, _WINDOW) if issubclass(record, ScheduledAdapterInput) else ()
    owner = next(item for item in fields(record) if item.name == _EXCLUDE.name)
    choices = next(
        rule.choices for rule in field_constraints(owner) if isinstance(rule, ItemChoicesConstraint)
    )
    exclude = replace(_EXCLUDE, valid="an array of categories: " + ", ".join(sorted(choices)))
    return ConfigRecordSpec(record, (_PATH, _STRENGTH, exclude, *scheduled, *extra_fields))


@dataclass(frozen=True)
class AdapterShorthand:
    """CLI-only suffix decoding into owned fields, without probing the filesystem."""

    schema: ConfigRecordSpec
    stage_fields: tuple[ConfigFieldSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.stage_fields and len(self.stage_fields) != 2:
            raise ValueError("adapter stage shorthand requires exactly two declared fields")
        if any(
            not any(field is owned for owned in self.schema.fields)
            for field in (_PATH, _STRENGTH, *self.stage_fields)
        ):
            raise ValueError("adapter shorthand fields must belong to its record schema")

    def __call__(self, text: str) -> dict[str, object]:
        members: dict[str, object] = {_PATH.name: text}
        prefix, separator, suffix = text.rpartition(".safetensors:")
        if not separator:
            return members
        try:
            numbers = tuple(float(part) for part in suffix.split(":"))
        except ValueError:
            return members
        targets = (_STRENGTH,) if len(numbers) == 1 else self.stage_fields
        if len(numbers) != len(targets):
            allowed = "one or two" if self.stage_fields else "one"
            raise ValueError(f"adapter shorthand requires {allowed} numeric suffix values")
        members[_PATH.name] = prefix + ".safetensors"
        members.update((field.name, number) for field, number in zip(targets, numbers, strict=True))
        # The ordinary record action performs field-owned local admission.
        return members


def _collection_example(schema: ConfigRecordSpec) -> tuple[dict[str, object], ...]:
    """Show every member this model accepts, in records a user could actually write.

    A step array replaces the common strength, so it cannot share a record with
    it or with a sigma window. Model-specific members go in their own record too,
    which keeps each rendered record inside the template's comment width.
    """
    owned = {member.name: member for member in schema.fields}
    shared = [_PATH.name, _STRENGTH.name, _EXCLUDE.name]
    if _WINDOW.name in owned:
        shared.append(_WINDOW.name)
    groups = [shared]
    if _STEPS.name in owned:
        groups.append([_PATH.name, _STEPS.name])
    extra = [name for name in owned if name not in {*group for group in groups}]
    if extra:
        groups.append([_PATH.name, *extra])
    return tuple(
        {
            name: Path("detail.safetensors")
            if index and name == _PATH.name
            else owned[name].example
            for name in group
        }
        for index, group in enumerate(groups)
    )


def adapter_collection_field(
    schema: ConfigRecordSpec, *, stage_fields: tuple[ConfigFieldSpec, ...] = ()
) -> ConfigFieldSpec:
    """Declare one replaceable ordered collection and its local item/member CLI bindings."""
    return ConfigFieldSpec(
        "adapters",
        "Select ordered user adapters",
        "an ordered array of adapter records",
        _collection_example(schema),
        "adapters",
        persistence="intent",
        item_schema=schema,
        cli_item=_PATH.name,
        cli=CliBinding(
            "Adapters",
            flags=("--adapter", "--lora"),
            parse=argument_value(AdapterShorthand(schema, stage_fields), "adapter"),
        ),
        surfaces=CONFIG_SURFACES,
    )
