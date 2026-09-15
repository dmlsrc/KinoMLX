"""Constructor presence survives typed coercion, recursive intent, and controlled copies."""

import dataclasses
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from kinomlx.config import dataclass_from_mapping, dataclass_to_mapping
from kinomlx.config_records import config_record, record_intent, replace_config, supplied_fields
from kinomlx.models.ltx2.types import ImageConditioningConfig, VideoVAETilingConfig


@config_record
@dataclasses.dataclass(frozen=True)
class Layer:
    source: Path
    opacity: float = 1.0
    enabled: bool = True
    categories: tuple[str, ...] = ()


@config_record
@dataclasses.dataclass(frozen=True)
class Fixture:
    layers: tuple[Layer, ...] = ()
    count: int = 0


@config_record
@dataclasses.dataclass(frozen=True)
class Pulse:
    magnitude: float = 1.0
    levels: tuple[float, ...] = ()
    enabled: bool = True

    def __post_init__(self):
        if "magnitude" in supplied_fields(self) and self.levels:
            raise ValueError("explicit magnitude conflicts with levels")


def test_equal_default_values_have_distinct_presence_without_changing_equality():
    omitted = Layer(Path("image.png"))
    explicit = Layer(Path("image.png"), opacity=1.0, enabled=True, categories=())
    assert omitted == explicit
    assert hash(omitted) == hash(explicit)
    assert record_intent(omitted) == {"source": Path("image.png")}
    assert record_intent(explicit) == {
        "source": Path("image.png"),
        "opacity": 1.0,
        "enabled": True,
        "categories": (),
    }
    assert supplied_fields(Fixture()) == frozenset()
    assert record_intent(Fixture(count=0)) == {"count": 0}


def test_decorator_retains_signature_annotations_and_dataclass_fields():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Plain:
        enabled: bool = False
        count: int = 0

    signature = inspect.signature(Plain)
    hints = get_type_hints(Plain)
    fields = dataclasses.fields(Plain)
    tracked = config_record(Plain)
    assert inspect.signature(tracked) == signature
    assert get_type_hints(tracked) == hints
    assert dataclasses.fields(tracked) == fields
    assert record_intent(tracked(enabled=False)) == {"enabled": False}
    with pytest.raises(TypeError):
        tracked(False)


def test_nested_record_presence_survives_mapping_and_typed_coercion():
    value = dataclass_from_mapping(
        Fixture,
        {
            "layers": [
                {"source": "first.png"},
                {"source": "second.png", "opacity": 1.0},
            ]
        },
        table="fixture",
    )
    expected = {
        "layers": ({"source": Path("first.png")}, {"source": Path("second.png"), "opacity": 1.0})
    }
    assert record_intent(value) == expected
    assert record_intent(dataclass_from_mapping(Fixture, value, table="fixture")) == expected
    assert set(dataclass_to_mapping(value)) == {"layers", "count"}
    assert "__kinomlx_config_presence__" not in repr(value)


def test_controlled_copy_preserves_presence_before_post_init_validation():
    value = Pulse(levels=(0.5,))
    assert supplied_fields(value) == frozenset({"levels"})
    changed = replace_config(value, enabled=False)
    assert record_intent(changed) == {"levels": (0.5,), "enabled": False}
    assert record_intent(dataclass_from_mapping(Pulse, changed, table="pulse")) == record_intent(
        changed
    )
    with pytest.raises(ValueError, match="conflicts"):
        replace_config(value, magnitude=1.0)
    with pytest.raises(ValueError, match="unknown configuration fields"):
        replace_config(value, typo=True)
    # Ordinary replacement supplies every default-filled field; it cannot preserve intent.
    with pytest.raises(ValueError, match="conflicts"):
        dataclasses.replace(value, enabled=False)


def test_collection_replacement_does_not_retain_prior_item_presence():
    original = Fixture(layers=(Layer(Path("old.png"), opacity=0.25),))
    replacement = replace_config(original, layers=(Layer(Path("new.png")),))
    assert record_intent(replacement) == {"layers": ({"source": Path("new.png")},)}
    assert record_intent(original)["layers"][0]["opacity"] == 0.25


def test_existing_ltx_records_preserve_public_constructor_and_track_geometry():
    image = ImageConditioningConfig(Path("image.png"))
    assert record_intent(image) == {"path": Path("image.png")}
    changed = replace_config(image, strength=0.95)
    assert record_intent(changed) == {"path": Path("image.png"), "strength": 0.95}
    tiling = dataclass_from_mapping(VideoVAETilingConfig, {"mode": "custom"}, table="tiling")
    assert record_intent(tiling) == {"mode": "custom"}
    assert tiling.temporal_overlap_frames == 24
    with pytest.raises(ValueError, match="strength"):
        replace_config(image, strength=2.0)


def test_untracked_records_do_not_invent_presence():
    @dataclasses.dataclass(frozen=True)
    class Ordinary:
        count: int = 0

    assert supplied_fields(Ordinary()) is None
    with pytest.raises(TypeError, match="not a presence-tracked"):
        record_intent(Ordinary())


def test_inspection_reports_actual_constructor_presence_registration():
    from kinomlx.config.inspection import configuration_inventory
    from kinomlx.model_registry import config_registry

    schema = config_registry().model("ltx2")
    item = configuration_inventory(schema, field="generate.image.strength")["fields"][0]
    assert item["constructor_presence"] == "tracked"
    root = configuration_inventory(schema, field="generate.width")["fields"][0]
    assert root["constructor_presence"] == "tracked"


def test_unsupported_record_shapes_fail_at_decoration():
    @dataclasses.dataclass
    class Mutable:
        count: int = 0

    @dataclasses.dataclass(frozen=True, slots=True)
    class Slotted:
        count: int = 0

    with pytest.raises(TypeError, match="frozen dataclass"):
        config_record(Mutable)
    with pytest.raises(TypeError, match="instance dictionary"):
        config_record(Slotted)


def test_record_subclasses_own_presence_instead_of_inheriting_a_marker():
    class Untracked(Layer):
        pass

    with pytest.raises(TypeError, match="requires its own config_record declaration"):
        Untracked(Path("source.png"))


def test_tracked_subclass_coercion_projects_presence_to_the_target_record():
    @config_record
    @dataclasses.dataclass(frozen=True)
    class Extended(Layer):
        label: str = ""

    value = Extended(Path("source.png"), label="extra")
    assert record_intent(value) == {"source": Path("source.png"), "label": "extra"}
    narrowed = dataclass_from_mapping(Layer, value, table="layer")
    assert type(narrowed) is Layer
    assert record_intent(narrowed) == {"source": Path("source.png")}
    copied = replace_config(value, opacity=1.0)
    assert type(copied) is Extended
    assert record_intent(copied) == {
        "source": Path("source.png"),
        "opacity": 1.0,
        "label": "extra",
    }


def test_every_registered_table_tracks_constructor_presence():
    from kinomlx.config_records import tracks_presence
    from kinomlx.model_registry import config_registry

    registry = config_registry()
    for model in registry.models():
        for table in registry.model(model).tables:
            assert tracks_presence(table.record), (model, table.path, table.record)


@pytest.mark.parametrize("model", ["flux2", "gmnet", "krea2", "ltx2"])
def test_public_tables_preserve_omission_explicit_defaults_and_controlled_copies(model):
    from kinomlx.model_registry import config_registry

    for table in config_registry().model(model).tables:
        required = {
            item.name: Path("saved-run.json")
            for item in dataclasses.fields(table.record)
            if item.init
            and item.default is dataclasses.MISSING
            and item.default_factory is dataclasses.MISSING
        }
        original = dataclass_from_mapping(table.record, required, table="record")
        assert supplied_fields(original) == frozenset(required)
        assert record_intent(original) == required
        candidates = [
            item for item in dataclasses.fields(original) if item.init and item.name not in required
        ]
        if not candidates:
            continue
        name = candidates[0].name
        explicit = replace_config(original, **{name: getattr(original, name)})
        assert original == explicit
        assert supplied_fields(explicit) == frozenset((*required, name))
        readmitted = dataclass_from_mapping(table.record, explicit, table="record")
        assert record_intent(readmitted) == record_intent(explicit)
        assert supplied_fields(original) == frozenset(required)


def test_krea_request_projection_retains_presence_and_resolved_values():
    from kinomlx.config_records import construct_record
    from kinomlx.models.krea2.types import Krea2GenerateConfig

    source = construct_record(
        Krea2GenerateConfig,
        {"prompt": "a tree", "steps": 8, "guidance": 1.0, "seed": 0},
        frozenset({"prompt", "seed"}),
    )
    request = source.request()
    assert request is not None
    assert request.steps == 8
    assert request.guidance == 1.0
    assert record_intent(request) == {"prompt": ("a tree",), "seed": 0}
    assert supplied_fields(request) == supplied_fields(source)


def test_ltx_settings_environment_and_override_preserve_only_supplied_values(monkeypatch):
    from kinomlx.models.ltx2.settings import LTX2Settings

    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_DTYPE", "float16")
    bare = LTX2Settings()
    assert record_intent(bare) == {}
    environment = LTX2Settings.from_env_fields("transformer_dtype")
    assert record_intent(environment) == {"transformer_dtype": "float16"}
    explicit = bare.with_overrides(transformer_dtype="bfloat16")
    assert record_intent(explicit) == {"transformer_dtype": "bfloat16"}


def test_derived_copy_keeps_presence_without_marking_resolved_values():
    from kinomlx.config_records import derive_config

    original = Pulse(levels=(0.5,))
    derived = derive_config(original, enabled=False)
    assert derived.enabled is False
    assert supplied_fields(derived) == frozenset({"levels"})
    assert record_intent(derived) == {"levels": (0.5,)}
    assert original.enabled is True
    assert supplied_fields(replace_config(derived, enabled=False)) == frozenset(
        {"levels", "enabled"}
    )
    with pytest.raises(ValueError, match="unknown configuration fields"):
        derive_config(original, missing=1)
    explicit = Pulse(magnitude=1.0)
    with pytest.raises(ValueError, match="explicit magnitude conflicts"):
        derive_config(explicit, levels=(0.5,))


@pytest.mark.parametrize("explicit", [False, True])
def test_ltx_preset_resolution_preserves_supplied_low_level_controls(explicit):
    from kinomlx.models.ltx2.settings import STREAM_TRANSFORMER_RESIDENT_BLOCKS, LTX2Settings

    original = (
        LTX2Settings(stream_transformer=True, transformer_resident_blocks=3)
        if explicit
        else LTX2Settings(stream_transformer=True)
    )
    resolved = original.resolve_presets()
    assert resolved.transformer_resident_blocks == (
        3 if explicit else STREAM_TRANSFORMER_RESIDENT_BLOCKS
    )
    assert supplied_fields(resolved) == supplied_fields(original)
    assert record_intent(resolved) == record_intent(original)
    assert record_intent(resolved.resolve_presets()) == record_intent(original)


def test_krea_recipe_defaults_do_not_become_supplied_during_assembly():
    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.config import assemble

    invocation = assemble(build_parser().parse_args(["--prompt", "a tree", "--seed", "0"]))
    assert invocation.request.steps is not None
    assert invocation.request.guidance is not None
    assert supplied_fields(invocation.request) == frozenset({"prompt", "seed"})


def test_source_aspect_derivation_keeps_original_dimension_presence():
    from kinomlx.models.krea2.sizing import resolve_source_sizing
    from kinomlx.models.krea2.types import Krea2Request

    original = Krea2Request(
        prompt="a tree", source_images=(Path("portrait.png"),), match_source_aspect="long"
    )
    sized, _ = resolve_source_sizing(original, source_size=(720, 1280))
    assert (sized.width, sized.height) == (576, 1024)
    assert supplied_fields(sized) == supplied_fields(original)
    assert record_intent(sized) == record_intent(original)


def test_environment_settings_reject_untracked_construction_and_copies():
    from kinomlx.settings import EnvironmentSettings

    @dataclasses.dataclass(frozen=True)
    class UntrackedSettings(EnvironmentSettings):
        count: int = 0

    with pytest.raises(TypeError, match="not a presence-tracked configuration record"):
        UntrackedSettings.from_env_fields("count")
    with pytest.raises(TypeError, match="not a presence-tracked configuration record"):
        UntrackedSettings().with_overrides(count=1)
