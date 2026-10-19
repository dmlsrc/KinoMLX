"""Stage permissions belong to fields and fail registration when incomplete."""

from dataclasses import dataclass, replace

import pytest

from kinomlx.config import ConfigFieldSpec, ConfigTableSpec
from kinomlx.config.stages import OutputPolicy, StagePolicy
from kinomlx.config_surfaces import MODEL_SURFACES, STRUCTURED_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.types import Krea2Request


@dataclass(frozen=True)
class Capture:
    count: int = 1
    gain: float = 1.0


def capture_table():
    return ConfigTableSpec(
        path=("capture",),
        record=Capture,
        stages=("process",),
        stage_record=Capture,
        fields=(
            ConfigFieldSpec(
                "count",
                "Number of samples",
                "integer",
                1,
                "count",
                persistence="intent",
                stage_policies=(
                    StagePolicy("process", "locked", reason="Processing already used the count."),
                ),
                surfaces=STRUCTURED_SURFACES,
            ),
            ConfigFieldSpec(
                "gain",
                "Processing gain",
                "number",
                1.0,
                "gain",
                persistence="intent",
                stage_policies=(
                    StagePolicy("process", "mutable", reason="Processing runs with a new gain."),
                ),
                surfaces=STRUCTURED_SURFACES,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("arguments", "keywords", "message"),
    [
        (("", "mutable"), {}, "requires a stage"),
        (("denoise", "sideways"), {}, "requires a stage"),
        (("denoise", "mutable"), {"reason": "   "}, "nonempty prose"),
        (("denoise", "locked", "not an attribute"), {}, "typed record attribute"),
        (("denoise", "mutable", "prompts"), {}, "may select a projection"),
        (("denoise", "irrelevant"), {}, "ignore or reject action"),
        (("denoise", "mutable"), {"irrelevant_action": "ignore"}, "ignore or reject action"),
    ],
)
def test_malformed_stage_policy_declarations_are_rejected(arguments, keywords, message):
    """Each guard rejects its own malformed declaration, not just one of them."""
    with pytest.raises(ValueError, match=message):
        StagePolicy(*arguments, **keywords)


@pytest.mark.parametrize(
    "produced_at",
    [
        {"generate"},
        frozenset({"not an identifier"}),
        frozenset({"generate", 7}),
    ],
)
def test_output_policy_requires_frozen_stage_names(produced_at):
    with pytest.raises(ValueError, match="frozen stage names"):
        OutputPolicy(produced_at)


def test_stage_declarations_are_immutable():
    policy = StagePolicy("denoise", "mutable", reason="A denoise restart reruns this.")
    with pytest.raises(AttributeError):
        policy.mode = "locked"
    with pytest.raises(AttributeError):
        OutputPolicy(frozenset({"generate"})).produced_at = frozenset()


def test_generic_stage_comparison_uses_owned_fields():
    table = capture_table()
    assert table.locked_changes(Capture(gain=2), Capture(), stage="process") == ()
    assert table.locked_changes(Capture(count=2), Capture(), stage="process") == (
        ("capture", "count"),
    )
    with pytest.raises(ValueError, match="unsupported stage"):
        table.locked_changes(Capture(), Capture(), stage="missing")
    with pytest.raises(TypeError, match="typed records"):
        table.locked_changes({}, Capture(), stage="process")


def test_missing_duplicate_and_stale_stage_classification_fail_registration():
    table = capture_table()
    count, gain = table.fields
    for policies in ((), (StagePolicy("wrong", "locked"),), count.stage_policies * 2):
        with pytest.raises(ValueError, match="stage policy|duplicate stage"):
            replace(table, fields=(replace(count, stage_policies=policies), gain))
    with pytest.raises(ValueError, match="unknown stage projection"):
        replace(
            table,
            fields=(
                replace(count, stage_policies=(StagePolicy("process", "locked", "missing"),)),
                gain,
            ),
        )


def test_new_request_field_cannot_silently_escape_stage_coverage():
    @dataclass(frozen=True)
    class ExtendedCapture(Capture):
        exposure: float = 0.5

    table = capture_table()
    with pytest.raises(ValueError, match="stage record field coverage drift"):
        replace(table, stage_record=ExtendedCapture)
    with pytest.raises(ValueError, match="exposure: stage policy coverage drift"):
        replace(
            table,
            record=ExtendedCapture,
            stage_record=ExtendedCapture,
            fields=(
                *table.fields,
                ConfigFieldSpec(
                    "exposure",
                    "Exposure",
                    "number",
                    0.5,
                    "exposure",
                    persistence="intent",
                    surfaces=STRUCTURED_SURFACES,
                ),
            ),
        )


def krea_table():
    return next(
        table for table in config_registry().model("krea2").tables if table.path == ("generate",)
    )


def test_krea_policy_preserves_existing_stage_permissions():
    table = krea_table()
    mutable = {
        stage: {
            field.name
            for field in table.fields
            for policy in field.stage_policies
            if policy.stage == stage and policy.mode == "mutable"
        }
        for stage in table.stages
    }
    assert mutable["decode"] == {"vae_tiling", "vae_tile_size", "vae_tile_stride"}
    assert mutable["denoise"] == mutable["decode"] | {
        "width",
        "height",
        "seed",
        "steps",
        "guidance",
        "mu",
        "sigmas",
        "noise_backend",
        "adapters",
        "turbo_4step",
        "turbo_4step_strength",
        "turbo_4step_exclusions",
        "turbo_4step_step_strengths",
        "turbo_4step_sigma_window",
        "turbo_sda",
        "turbo_sda_strength",
        "turbo_sda_exclusions",
        "turbo_sda_step_strengths",
        "turbo_sda_sigma_window",
        "text_conditioning",
    }


def test_krea_compares_prompt_meaning_and_applies_stage_specific_permissions():
    table = krea_table()
    parent = Krea2Request(prompt="hello")
    assert table.locked_changes(replace(parent, prompt=("hello",)), parent, stage="decode") == ()
    assert table.locked_changes(replace(parent, prompt="changed"), parent, stage="denoise") == (
        ("generate", "prompt"),
    )
    resized = replace(parent, width=1152)
    assert table.locked_changes(resized, parent, stage="denoise") == ()
    assert table.locked_changes(resized, parent, stage="decode") == (("generate", "width"),)
    assert table.locked_changes(replace(parent, vae_tiling="single"), parent, stage="decode") == ()


def test_flux_policy_preserves_existing_denoise_allowances_and_decode_locks():
    from kinomlx.models.flux2.types import Flux2Request

    table = next(
        table for table in config_registry().model("flux2").tables if table.path == ("generate",)
    )
    mutable = {
        stage: {
            field.name
            for field in table.fields
            for policy in field.stage_policies
            if policy.stage == stage and policy.mode == "mutable"
        }
        for stage in table.stages
    }
    assert mutable["decode"] == set()
    assert mutable["denoise"] == {
        "adapters",
        "steps",
        "sigmas",
        "guidance",
        "cfg_mode",
        "recompute_reference_kv",
        "seed",
        "noise_backend",
        "text_conditioning",
    }
    parent = Flux2Request(prompt="hello", steps=4)
    changed = replace(parent, steps=6)
    assert table.locked_changes(changed, parent, stage="denoise") == ()
    assert table.locked_changes(changed, parent, stage="decode") == (("generate", "steps"),)
    changed = replace(parent, width=1024)
    assert table.locked_changes(changed, parent, stage="denoise") == (("generate", "width"),)


def test_nested_optional_stage_records_preserve_presence_and_child_coverage():
    from kinomlx.config import ModelConfigSpec

    @dataclass(frozen=True)
    class Envelope:
        capture: Capture | None = None

    child = replace(capture_table(), path=("session", "capture"))
    root = ConfigTableSpec(
        path=("session",),
        record=Envelope,
        fields=(),
        nested=("capture",),
        stages=("process",),
        stage_record=Envelope,
    )
    schema = ModelConfigSpec(
        model="probe",
        title="Probe",
        introduction=("Probe nested policy.",),
        stages=("acquire", "process"),
        restart_stages=("process",),
        root_fields=(
            ConfigFieldSpec(
                "model",
                "Select model",
                "",
                "probe",
                "model",
                persistence="intent",
                virtual=True,
                default="probe",
                choices=("probe",),
                surfaces=MODEL_SURFACES,
                cli=MODEL_CONTROL.binding,
            ),
        ),
        tables=(root, child),
    )
    assert schema.locked_changes(("session",), Envelope(), Envelope(), stage="process") == ()
    assert schema.locked_changes(
        ("session",), Envelope(Capture()), Envelope(), stage="process"
    ) == (("session", "capture", "count"),)
    assert schema.locked_changes(
        ("session",), Envelope(), Envelope(Capture()), stage="process"
    ) == (("session", "capture", "count"),)
    assert (
        schema.locked_changes(
            ("session",), Envelope(Capture(gain=2)), Envelope(Capture()), stage="process"
        )
        == ()
    )
    unclassified = replace(
        child,
        stages=(),
        stage_record=None,
        fields=tuple(replace(field, stage_policies=()) for field in child.fields),
    )
    with pytest.raises(ValueError, match="probe.session.capture: stage coverage drift"):
        replace(schema, tables=(root, unclassified))


def test_ltx_stage_decisions_match_prior_locks_including_whole_conditioning_records():
    schema = config_registry().model("ltx2")
    locked = {
        stage: {
            ".".join((*table.path[1:], field.name))
            for table in schema.tables
            if table.path[0] == "generate"
            for field in table.fields
            for policy in field.stage_policies
            if policy.stage == stage and policy.mode == "locked"
        }
        for stage in ("decode", "stage-2")
    }
    both = {
        "adapters",
        "duration",
        "first_frame_mode",
        "fps",
        "frames",
        "generated_keyframes",
        "height",
        "images",
        "pad_prompt_to_max",
        "prompt",
        "prompt_unconditional",
        "reference_aligned_audio",
        "stage_1_dense_layers",
        "stage_1_sampler",
        "stage_1_sigmas",
        "stage_1_sparse_attention",
        "width",
    }
    assert locked["stage-2"] == both
    assert locked["decode"] == both | {
        "stage_2_dense_layers",
        "stage_2_sampler",
        "stage_2_sigmas",
        "stage_2_sparse_attention",
        "text_conditioning",
        "hdr_reference.path",
        "hdr_reference.strength",
    }


def test_ltx_nested_changes_report_canonical_fields_and_preserve_stage_behavior():
    from pathlib import Path

    from kinomlx.config_records import replace_config
    from kinomlx.models.ltx2.types import (
        LTX2ImageConditioningConfig,
        LTX2Request,
        LTX2VideoVAETilingConfig,
    )

    schema = config_registry().model("ltx2")
    parent = LTX2Request(images=(LTX2ImageConditioningConfig(Path("image.png")),))
    # The common strength feeds stage 1, so moving it moves the stage-1 profile.
    changed = replace(parent, images=(replace_config(parent.images[0], strength=0.5),))
    assert schema.locked_changes(("generate",), changed, parent, stage="decode") == (
        ("generate", "images"),
    )
    assert schema.locked_changes(("generate",), changed, parent, stage="stage-2") == (
        ("generate", "images"),
    )
    # A stage-2 strength is the one member the stage-1 projection does not read.
    changed = replace(parent, images=(replace_config(parent.images[0], stage2_strength=0.5),))
    assert schema.locked_changes(("generate",), changed, parent, stage="stage-2") == ()
    assert schema.locked_changes(("generate",), changed, parent, stage="decode") == (
        ("generate", "images"),
    )
    # The frame-0 mode is one run-level field and is locked at both restarts.
    changed = replace(parent, first_frame_mode="guide")
    assert schema.locked_changes(("generate",), changed, parent, stage="stage-2") == (
        ("generate", "first_frame_mode"),
    )
    assert schema.locked_changes(("generate",), changed, parent, stage="decode") == (
        ("generate", "first_frame_mode"),
    )
    changed = replace(parent, vae_tiling=LTX2VideoVAETilingConfig(mode="single"))
    assert schema.locked_changes(("generate",), changed, parent, stage="decode") == ()


@pytest.mark.parametrize("stage", ["denoise", "decode"])
def test_flux_model_setting_locks_are_declared_per_field(stage):
    from kinomlx.config_records import replace_config
    from kinomlx.models.flux2.settings import Flux2Settings

    baseline = Flux2Settings()
    changed = replace_config(baseline, variant="base")
    schema = config_registry().model("flux2")
    assert schema.locked_changes(("model_settings",), changed, baseline, stage=stage) == (
        ("model_settings", "variant"),
    )
    assert schema.locked_changes(("model_settings",), baseline, baseline, stage=stage) == ()


@pytest.mark.parametrize(
    "field",
    [
        "transformer_dtype",
        "source_path",
        "tokenizer_path",
        "text_encoder_path",
        "transformer_path",
        "vae_path",
    ],
)
@pytest.mark.parametrize("stage", ["denoise", "decode"])
def test_flux_resource_selection_declares_irrelevance_instead_of_a_lock(field, stage):
    """Replay reopens recorded components, so a new resource choice is not a lock."""
    schema = config_registry().model("flux2")
    policy = next(
        policy
        for policy in schema.field(("model_settings", field)).stage_policies
        if policy.stage == stage
    )
    assert (policy.mode, policy.irrelevant_action) == ("irrelevant", "reject")
    assert policy.reason


def test_flux_model_settings_cannot_omit_one_field_stage_policy():
    schema = config_registry().model("flux2")
    table = next(table for table in schema.tables if table.path == ("model_settings",))
    field, *others = table.fields
    with pytest.raises(ValueError, match="stage policy coverage drift"):
        replace(table, fields=(replace(field, stage_policies=()), *others))
