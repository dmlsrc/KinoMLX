"""Artifact availability is derived from each family's stage model, not written by hand."""

from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stages import produced_at_stages
from kinomlx.config_rules import validate_record
from kinomlx.model_registry import config_registry


def test_derivation_reproduces_the_retired_hand_written_sets():
    """These literals are the eleven sets the families declared before derivation.

    They stay literal so that a change to a stage model is a visible change
    here. ``text`` is the first stage, which the old sets spelled ``generate``.
    """
    from kinomlx.models.flux2.output_policies import OUTPUT_POLICIES as klein
    from kinomlx.models.krea2.output_policies import OUTPUT_POLICIES as krea
    from kinomlx.models.ltx2.output_policies import ARTIFACT_STAGES as ltx

    image = {
        "save_initial_noise": frozenset({"text", "denoise"}),
        "save_final_latents": frozenset({"text", "denoise", "decode"}),
        "save_text_conditioning": frozenset({"text", "denoise"}),
    }
    assert {name: policy.produced_at for name, policy in klein.items()} == image
    assert {name: policy.produced_at for name, policy in krea.items()} == image
    assert {name: policy.produced_at for name, policy in ltx.items()} == {
        "text_conditioning": frozenset({"text", "stage-2"}),
        "stage_1_conditioning": frozenset({"text"}),
        "stage_2_conditioning": frozenset({"text", "stage-2"}),
        "stage_1_latents": frozenset({"text", "stage-2", "decode"}),
        "final_latents": frozenset({"text", "stage-2", "decode"}),
    }


def test_an_artifact_is_producible_where_its_stage_runs_or_where_it_is_re_emitted():
    stages = produced_at_stages(
        ("text", "denoise", "decode"),
        {"text": "text", "noise": "denoise", "latents": "denoise"},
        {"denoise": ("text",), "decode": ("latents",)},
    )
    assert stages == {
        "text": frozenset({"text", "denoise"}),
        "noise": frozenset({"text", "denoise"}),
        "latents": frozenset({"text", "denoise", "decode"}),
    }


@pytest.mark.parametrize(
    ("stages", "produces", "consumes", "message"),
    [
        ((), {}, {}, "distinct stages"),
        (("text", "text"), {}, {}, "distinct stages"),
        (("text", "decode"), {"latents": "denoise"}, {}, "undeclared stage"),
        (("text", "decode"), {"latents": "text"}, {"text": ("latents",)}, "after the first"),
        (("text", "decode"), {"latents": "text"}, {"encode": ("latents",)}, "after the first"),
        (("text", "decode"), {"latents": "text"}, {"decode": ("noise",)}, "not produced earlier"),
        (("text", "decode"), {"pixels": "decode"}, {"decode": ("pixels",)}, "not produced earlier"),
    ],
)
def test_stage_model_declarations_are_checked(stages, produces, consumes, message):
    with pytest.raises(ValueError, match=message):
        produced_at_stages(stages, produces, consumes)


def test_ltx_restarts_read_the_latest_latent_before_them_unless_told_otherwise():
    from kinomlx.models.ltx2.restart_rules import (
        FINAL_LATENTS,
        STAGE_1_LATENTS,
        TEXT_CONDITIONING,
        LTX2Stage,
        consumed_artifacts,
        default_latent_stage,
        latent_stages_before,
    )

    assert consumed_artifacts(LTX2Stage.STAGE_2) == (TEXT_CONDITIONING, STAGE_1_LATENTS)
    assert consumed_artifacts(LTX2Stage.DECODE) == (FINAL_LATENTS,)
    assert consumed_artifacts(LTX2Stage.DECODE, LTX2Stage.STAGE_1) == (STAGE_1_LATENTS,)
    # Stage 2 has one latent written before it and decode has two, so only
    # decode offers a choice, and both default to the latest.
    assert latent_stages_before(LTX2Stage.STAGE_2) == (LTX2Stage.STAGE_1,)
    assert latent_stages_before(LTX2Stage.DECODE) == (LTX2Stage.STAGE_1, LTX2Stage.STAGE_2)
    assert default_latent_stage(LTX2Stage.STAGE_2) == LTX2Stage.STAGE_1
    assert default_latent_stage(LTX2Stage.DECODE) == LTX2Stage.STAGE_2


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"stage": "decode"}, "decode restart takes final_latents, got nothing"),
        (
            {"stage": "decode", "latents": Path("x"), "text_conditioning": Path("t")},
            "decode restart takes final_latents, got final_latents, text_conditioning",
        ),
        ({"stage": "denoise"}, "denoise restart takes text_conditioning, got nothing"),
        (
            {"stage": "denoise", "text_conditioning": Path("t"), "latents": Path("x")},
            "denoise restart takes text_conditioning, got final_latents, text_conditioning",
        ),
    ],
)
@pytest.mark.parametrize("family", ["flux2", "krea2"])
def test_image_restart_records_must_carry_exactly_what_the_stage_reads(family, record, message):
    from kinomlx.models.flux2.restart import Flux2Restart
    from kinomlx.models.krea2.restart import Krea2Restart

    restart = {"flux2": Flux2Restart, "krea2": Krea2Restart}[family](**record)
    with pytest.raises(ConfigError, match=f"\\[restart\\]: {message}"):
        validate_record(restart, location="[restart]")


def test_image_restart_selection_offers_a_latent_override_only_where_one_is_read(tmp_path):
    from kinomlx.models.krea2.restart import RestartConfig

    validate_record(RestartConfig(run=tmp_path, stage="decode", latents=Path("x")), location="[r]")
    with pytest.raises(ConfigError, match="restart latents apply only when stage=decode"):
        validate_record(
            RestartConfig(run=tmp_path, stage="denoise", latents=Path("x")), location="[r]"
        )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {"stage": "stage-2", "latent_stage": "stage-2", "text_conditioning": Path("t")},
            "stage-2 restart cannot read latents written at stage-2",
        ),
        (
            {"stage": "stage-2"},
            "stage-2 restart takes text_conditioning, stage_1_latents, got stage_1_latents",
        ),
        (
            {"stage": "decode", "text_conditioning": Path("t")},
            "decode restart takes final_latents, got final_latents, text_conditioning",
        ),
    ],
)
def test_ltx_restart_records_must_carry_exactly_what_the_stage_reads(record, message):
    from kinomlx.models.ltx2.restart import LTX2Restart

    with pytest.raises(ConfigError, match=f"\\[restart\\]: {message}"):
        validate_record(LTX2Restart(latents=Path("x"), **record), location="[restart]")


def test_ltx_restart_selection_admits_any_latent_written_before_the_stage(tmp_path):
    from kinomlx.models.ltx2.restart import RestartConfig

    for stage, latent_stage in (
        ("decode", "stage-1"),
        ("decode", "stage-2"),
        ("stage-2", "stage-1"),
    ):
        validate_record(
            RestartConfig(run=tmp_path, stage=stage, latent_stage=latent_stage), location="[r]"
        )
    with pytest.raises(ConfigError, match="stage-2 restart cannot read latents written at stage-2"):
        validate_record(
            RestartConfig(run=tmp_path, stage="stage-2", latent_stage="stage-2"), location="[r]"
        )


def test_an_unknown_stage_is_refused_before_any_rule_can_treat_it_as_one():
    """The typed record refuses the stage at construction; the rules refuse it on a direct call."""
    from kinomlx.models.krea2.restart import Krea2Restart
    from kinomlx.models.ltx2.restart import LTX2Restart
    from kinomlx.models.ltx2.restart_rules import latent_stages_before

    with pytest.raises(ConfigError, match="stage: expected one of .text., .denoise., .decode."):
        Krea2Restart(stage="bogus", latents=Path("x"))
    with pytest.raises(ConfigError, match="stage: expected one of .text., .stage-1., .stage-2."):
        LTX2Restart(latents=Path("x"), stage="bogus")
    # A stage that exists but cannot be started at is refused by the restart choices.
    with pytest.raises(ConfigError, match="stage"):
        Krea2Restart(stage="text", latents=Path("x"))
    with pytest.raises(ConfigError, match="stage"):
        LTX2Restart(latents=Path("x"), stage="stage-1")
    with pytest.raises(ValueError, match="unknown restart stage .bogus."):
        latent_stages_before("bogus")


def test_each_model_declares_its_stages_in_execution_order():
    registry = config_registry()
    assert registry.model("krea2").stages == ("text", "denoise", "decode")
    assert registry.model("flux2").stages == ("text", "denoise", "decode")
    assert registry.model("ltx2").stages == ("text", "stage-1", "stage-2", "decode")
    assert registry.model("gmnet").stages == ("expand",)
    # A run starts at the first stage or at a restart stage; LTX's stage 1 is neither.
    assert registry.model("ltx2").start_stages == {"text", "stage-2", "decode"}
    assert registry.model("gmnet").start_stages == {"expand"}


def test_a_fresh_run_prepares_at_the_first_stage_by_default():
    state = PreparedConfig({"output": {"save_initial_noise": False}}, {})
    schema = config_registry().model("krea2")
    prepared = schema.prepare_stage_layer(state, stage=schema.stages[0])
    assert prepared.values == state.values
