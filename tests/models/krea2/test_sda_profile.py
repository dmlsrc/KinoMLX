"""SDA diversity-profile selection, its default gate, and the pairing warning."""

import re
from pathlib import Path

import pytest

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.config_rules import validate_record
from kinomlx.models.krea2.adapters.profiles import resolve_profile_adapters
from kinomlx.models.krea2.adapters.sda_profile import (
    SDA_ADAPTER_FILENAME,
    SDA_ADAPTER_REPO,
    SDA_DEFAULT_SIGMA_WINDOW,
    SDA_TRAINED_BAND,
    SDAProfile,
    resolve_sda_adapter,
    sda_pairing_warning,
)
from kinomlx.models.krea2.adapters.turbo_profile import (
    TURBO_ADAPTER_FILENAME,
    TURBO_ADAPTER_REPO,
    TurboFourStepProfile,
)
from kinomlx.models.krea2.types import Krea2Request

# The real Turbo schedules; step i is evaluated at levels[i].
EIGHT = (1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0.0)
FOUR = (1.0, 0.904531, 0.759511, 0.512844, 0.0)


def _cache(tmp_path, *repos):
    for repo_id, filename in repos:
        repo = tmp_path / "hub" / ("models--" + repo_id.replace("/", "--"))
        snapshot = repo / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / filename).touch()
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("a" * 40)
    return tmp_path


@pytest.fixture
def sda_cache(tmp_path):
    return _cache(tmp_path, (SDA_ADAPTER_REPO, SDA_ADAPTER_FILENAME))


@pytest.fixture
def both_cache(tmp_path):
    return _cache(
        tmp_path,
        (SDA_ADAPTER_REPO, SDA_ADAPTER_FILENAME),
        (TURBO_ADAPTER_REPO, TURBO_ADAPTER_FILENAME),
    )


def test_the_profile_pins_the_complete_key_spelling() -> None:
    # The diffusers-style conversion in the same repository loses 32 of 264
    # targets, so a first-class switch must not be able to select it.
    assert SDA_ADAPTER_FILENAME.endswith("_comfy.safetensors")


def test_disabled_returns_the_request_untouched() -> None:
    selections, receipt = resolve_sda_adapter(
        None,
        enabled=False,
        strength=1.0,
        sigma_window=SDA_DEFAULT_SIGMA_WINDOW,
        step_strengths=(),
        exclusions="",
        additional=(),
    )
    assert selections == ()
    assert receipt is None


def test_the_default_gate_is_the_adapter_s_own_training_band(sda_cache) -> None:
    request = Krea2Request(prompt="scene", turbo_sda=True)
    (selected,), _, receipt = resolve_profile_adapters(
        sda_cache,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(),
    )
    assert selected.sigma_window is not None
    assert selected.schedule is None
    assert selected.sigma_window.bands[0].low == SDA_TRAINED_BAND[0]
    assert receipt is not None
    assert receipt["gate"] == {"sigma_window": SDA_DEFAULT_SIGMA_WINDOW}


def test_the_default_gate_reproduces_the_card_recipe_on_the_native_schedule() -> None:
    # Two of eight steps, which is the gate the adapter's model card prescribes.
    from kinomlx.adapters.schedule import parse_sigma_window

    window = parse_sigma_window(SDA_DEFAULT_SIGMA_WINDOW)
    assert window.resolve(EIGHT, strength=1.0, label="sda").per_step == (
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_a_step_array_replaces_the_default_window(sda_cache) -> None:
    request = Krea2Request(
        prompt="scene", turbo_sda=True, turbo_sda_step_strengths=(1.0, 1.0, 0.0, 0.0)
    )
    (selected,), _, _ = resolve_profile_adapters(
        sda_cache,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(),
    )
    assert selected.schedule is not None
    assert selected.sigma_window is None
    assert selected.schedule.per_step == (1.0, 1.0, 0.0, 0.0)


def test_a_strength_rides_alongside_the_window(sda_cache) -> None:
    request = Krea2Request(prompt="scene", turbo_sda=True, turbo_sda_strength=1.5)
    (selected,), _, _ = resolve_profile_adapters(
        sda_cache,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(),
    )
    assert selected.strength == 1.5
    assert selected.sigma_window is not None


@pytest.mark.parametrize(
    "change",
    [
        {"turbo_sda_sigma_window": "0.93", "turbo_sda_step_strengths": (1.0, 0.0)},
        {"turbo_sda_sigma_window": "0.97:0.93"},
        {"turbo_sda_step_strengths": (1.0, float("nan"))},
    ],
)
def test_gate_values_and_relationships_are_rejected_at_their_boundaries(change) -> None:
    with pytest.raises(ValueError, match="not both|above high|must be finite"):
        validate_record(
            Krea2Request(prompt="scene", turbo_sda=True, **change), location="[generate]"
        )


def test_gate_controls_require_the_switch() -> None:
    message = "turbo_sda_sigma_window and turbo_sda_step_strengths require turbo_sda"
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        validate_record(
            Krea2Request(prompt="scene", turbo_sda_sigma_window="0.93"), location="[generate]"
        )
    assert str(error.value).endswith(message)


def test_the_distilled_profile_leads_the_diversity_profile(both_cache) -> None:
    request = Krea2Request(prompt="scene", turbo_sda=True, turbo_4step=True)
    selections, four_receipt, sda_receipt = resolve_profile_adapters(
        both_cache,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(),
    )
    assert [path.path.name for path in selections] == [
        TURBO_ADAPTER_FILENAME,
        SDA_ADAPTER_FILENAME,
    ]
    assert four_receipt is not None
    assert sda_receipt is not None


def test_a_users_own_selections_follow_both_profiles(both_cache, tmp_path) -> None:
    own = AdapterSelection(tmp_path / "style.safetensors", 1.0)
    request = Krea2Request(prompt="scene", turbo_sda=True, turbo_4step=True)
    selections, _, _ = resolve_profile_adapters(
        both_cache,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(own,),
    )
    assert selections[-1] is own


def test_listing_the_profile_adapter_by_hand_is_refused(sda_cache) -> None:
    duplicate = AdapterSelection(
        Path(sda_cache)
        / "hub"
        / ("models--" + SDA_ADAPTER_REPO.replace("/", "--"))
        / "snapshots"
        / ("a" * 40)
        / SDA_ADAPTER_FILENAME,
        1.0,
    )
    with pytest.raises(ValueError, match="also listed in --adapter"):
        resolve_sda_adapter(
            sda_cache,
            enabled=True,
            strength=1.0,
            sigma_window=SDA_DEFAULT_SIGMA_WINDOW,
            step_strengths=(),
            exclusions="",
            additional=(duplicate,),
        )


def test_the_native_schedule_reaches_the_training_band() -> None:
    assert sda_pairing_warning(8, EIGHT) is None


def test_a_schedule_that_misses_the_band_is_named_not_silently_run() -> None:
    warning = sda_pairing_warning(4, FOUR)
    assert warning is not None
    assert "0.9045" in warning
    assert "unvalidated" in warning


def test_profiles_carry_their_own_switches_without_a_request_type() -> None:
    # The records let the resolver stay free of the model's request type, which
    # is what keeps the adapter package out of an import cycle back into it.
    assert TurboFourStepProfile().enabled is False
    assert SDAProfile(enabled=True, strength=2.0).strength == 2.0


@pytest.mark.parametrize("profile", ["turbo_sda", "turbo_4step"])
def test_profile_gate_conflict_names_real_fields_at_both_boundaries(tmp_path, profile):
    from kinomlx.models.krea2.adapters.turbo_profile import resolve_turbo_adapters

    message = f"use {profile}_sigma_window or {profile}_step_strengths, not both"
    with pytest.raises(ValueError, match="not both") as request_error:
        validate_record(
            Krea2Request(
                prompt="scene",
                **{
                    profile: True,
                    f"{profile}_sigma_window": "0.93",
                    f"{profile}_step_strengths": (1.0, 0.0),
                },
            ),
            location="[generate]",
        )
    assert str(request_error.value).endswith(message)
    resolver = resolve_sda_adapter if profile == "turbo_sda" else resolve_turbo_adapters
    with pytest.raises(ValueError, match="not both") as direct_error:
        resolver(
            tmp_path,
            enabled=True,
            strength=1.0,
            sigma_window="0.93",
            step_strengths=(1.0, 0.0),
            exclusions="",
            additional=(),
        )
    assert str(direct_error.value) == message


def test_an_absent_gate_is_recorded_as_none_like_the_four_step_profile(sda_cache) -> None:
    from kinomlx.models.krea2.adapters.sda_profile import resolve_sda_adapter

    _, receipt = resolve_sda_adapter(
        sda_cache,
        enabled=True,
        strength=1.0,
        sigma_window="",
        step_strengths=(),
        exclusions="",
        additional=(),
    )
    assert receipt is not None
    assert receipt["gate"] is None
