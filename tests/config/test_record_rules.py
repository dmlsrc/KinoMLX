"""Field-owned semantic checks run only on complete surviving records."""

import importlib
from dataclasses import dataclass, field, fields, make_dataclass
from functools import partial
from types import MappingProxyType

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config_records import config_record
from kinomlx.config_rules import RecordRule, record_rules, validate_record
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.settings import EnvironmentSettings


def check_budget(values):
    if values["usage"] > values["limit"]:
        raise ValueError("usage exceeds limit")


BUDGET = RecordRule("budget", frozenset({"usage", "limit"}), check_budget)


@config_record
@dataclass(frozen=True)
class Budget(EnvironmentSettings):
    limit: int = 2
    usage: int = field(default=0, metadata={"config_rules": (BUDGET,)})


def test_new_field_owned_rule_uses_the_common_final_validator():
    record = Budget(usage=3)
    # Local admission permits a primitive-valid record until final semantic validation.
    with pytest.raises(ValueError, match="usage exceeds limit"):
        record.validate()
    record.with_overrides(limit=4).validate()


@pytest.mark.parametrize("reads", [frozenset({"missing"}), frozenset({"limit"})])
def test_unknown_reads_or_missing_owner_are_rejected(reads):
    rule = RecordRule("invalid", reads, check_budget)
    record = make_dataclass(
        "Invalid",
        [("usage", int, field(default=0, metadata={"config_rules": (rule,)})), ("limit", int, 2)],
    )
    with pytest.raises(ValueError, match="must read its owner and only declared fields"):
        record_rules(record)


def test_conflicting_rule_identities_are_rejected():
    other = RecordRule("budget", frozenset({"limit"}), lambda values: None)
    record = make_dataclass(
        "Invalid",
        [
            ("usage", int, field(default=0, metadata={"config_rules": (BUDGET,)})),
            ("limit", int, field(default=2, metadata={"config_rules": (other,)})),
        ],
    )
    with pytest.raises(ValueError, match="conflicting semantic rule"):
        record_rules(record)


def test_rule_receives_only_declared_readonly_inputs():
    calls = []

    def check(values):
        assert dict(values) == {"enabled": False}
        with pytest.raises(TypeError):
            values["enabled"] = True
        calls.append(dict(values))

    @config_record
    @dataclass(frozen=True)
    class Options(EnvironmentSettings):
        enabled: bool = field(
            default=False,
            metadata={"config_rules": (RecordRule("check", frozenset({"enabled"}), check),)},
        )
        unrelated: str = "not read"

    options = Options()
    assert calls == []
    options.validate()
    assert calls == [{"enabled": False}]
    assert options.enabled is False


def test_later_override_repairs_a_cross_field_conflict(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[model_settings]\nreference_denoise = true\ntransformer_dtype = "float16"\n')
    args = ["--prompt", "test", "--config", str(path), "--print-config"]
    with pytest.raises(
        ConfigError, match=r"\[model_settings\]: Krea 2 reference denoising does not support"
    ):
        assemble(build_parser().parse_args(args))
    invocation = assemble(build_parser().parse_args([*args, "--transformer-dtype", "bfloat16"]))
    assert invocation.model_settings.reference_denoise is True
    assert invocation.model_settings.transformer_dtype == "bfloat16"


def test_schema_inspection_exposes_semantic_owner_and_reads():
    inventory = configuration_inventory(
        config_registry().model("krea2"), field="model_settings.reference_denoise"
    )
    entry = inventory["fields"][0]
    assert entry["validation_rules"] == [
        {
            "rule": "reference_precision",
            "phase": "semantic",
            "reads": ["reference_denoise", "transformer_dtype"],
            "effects": [],
        }
    ]


@pytest.mark.parametrize("model", ["flux2", "gmnet"])
@pytest.mark.parametrize("boundary", ["assembly", "resources"])
def test_new_semantic_rule_runs_at_each_model_boundary_before_resource_access(
    monkeypatch, model, boundary
):
    settings_module = importlib.import_module(f"kinomlx.models.{model}.settings")
    record = settings_module.Flux2Settings if model == "flux2" else settings_module.GMNetSettings

    def reject(values):
        raise ValueError("new owned rule executed")

    owner = next(item for item in fields(record) if item.name == "variant")
    monkeypatch.setattr(
        owner,
        "metadata",
        MappingProxyType(
            {
                **owner.metadata,
                "config_rules": (RecordRule("new_rule", frozenset({"variant"}), reject),),
            }
        ),
    )
    if boundary == "assembly":
        parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
        module = importlib.import_module(f"kinomlx.models.{model}.config")
        invoke = partial(module.assemble, parser.parse_args(["--print-config"]))
    else:
        from kinomlx.settings import Settings

        resources = importlib.import_module(f"kinomlx.models.{model}.resources")
        prepare = resources.inspect_resources if model == "flux2" else resources.prepare_resources
        invoke = partial(prepare, record(), infrastructure=Settings())
    with pytest.raises(ValueError, match="new owned rule executed"):
        invoke()


def _no_owner(values):
    if values["usage"] > values["limit"]:
        raise ValueError("usage exceeds limit")


NESTED_BUDGET = RecordRule("nested_budget", frozenset({"usage", "limit"}), _no_owner)


@config_record
@dataclass(frozen=True)
class Allowance:
    limit: int = 2
    usage: int = field(default=0, metadata={"config_rules": (NESTED_BUDGET,)})


@config_record
@dataclass(frozen=True)
class Ledger:
    primary: Allowance = field(default_factory=Allowance)
    entries: tuple[Allowance, ...] = ()


def test_rule_violation_names_the_table_and_not_the_class_or_the_rule():
    with pytest.raises(ConfigError) as error:
        validate_record(Allowance(usage=3), location="[budget]")
    assert str(error.value) == "[budget]: usage exceeds limit"
    assert "Allowance" not in str(error.value)
    assert "nested_budget" not in str(error.value)


def test_rule_violation_on_a_nested_member_names_its_table_path():
    with pytest.raises(ConfigError) as member:
        validate_record(Ledger(primary=Allowance(usage=3)), location="[ledger]")
    assert str(member.value) == "[ledger].primary: usage exceeds limit"
    with pytest.raises(ConfigError) as item:
        validate_record(Ledger(entries=(Allowance(), Allowance(usage=3))), location="[ledger]")
    assert str(item.value) == "[ledger].entries[1]: usage exceeds limit"


def test_rule_violations_are_configuration_errors_every_surface_can_render():
    with pytest.raises(ValueError, match="usage exceeds limit") as error:
        validate_record(Allowance(usage=3), location="[budget]")
    # ConfigError subclasses ValueError, so every existing handler still catches
    # it while a CLI boundary can present it without a traceback.
    assert isinstance(error.value, ConfigError)
