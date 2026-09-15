"""Noise-level bands, resolved against the schedule a sampler actually walks."""

import pytest

from kinomlx.adapters.inputs import ScheduledAdapterInput
from kinomlx.adapters.schedule import (
    AdapterSigmaBand,
    AdapterSigmaWindow,
    parse_sigma_window,
    render_strength_plan,
)
from kinomlx.adapters.selection import (
    AdapterSelection,
    require_adapter_schedule_steps,
)

# The real Krea Turbo schedules; step i is evaluated at EIGHT[i].
EIGHT = (1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0.0)
FOUR = (1.0, 0.904531, 0.759511, 0.512844, 0.0)


def _resolve(spec, sigmas=EIGHT, strength=1.0):
    return parse_sigma_window(spec).resolve(sigmas, strength=strength, label="adapter a").per_step


def test_a_low_bound_selects_every_level_above_it() -> None:
    # The community SDA adapter records 0.93 as its training floor; on the eight-step
    # schedule that selects exactly the two steps its model card prescribes.
    assert _resolve("0.93") == (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_a_band_selects_only_levels_inside_it() -> None:
    assert _resolve("0.93:0.97") == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_the_same_spec_narrows_honestly_on_a_shorter_schedule() -> None:
    # No four-step level falls in the training band, so the gate is one step, not two.
    assert _resolve("0.93", sigmas=FOUR) == (1.0, 0.0, 0.0, 0.0)


def test_the_selection_strength_fills_a_band_that_declares_none() -> None:
    assert _resolve("0.93", strength=0.75) == (0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_each_band_can_declare_its_own_strength() -> None:
    assert _resolve("0.93@1,0.6:0.93@0.4") == (1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0)


def test_several_bands_express_a_taper_in_one_spec() -> None:
    assert _resolve("0.9:1@1,0.7:0.9@0.6,0:0.7@0.2") == (
        1.0,
        1.0,
        1.0,
        0.6,
        0.6,
        0.2,
        0.2,
        0.2,
    )


def test_the_first_band_holding_a_level_wins() -> None:
    # Both bands hold every level; the earlier one decides.
    assert _resolve("0:1@0.3,0:1@0.9") == (0.3,) * 8


def test_a_declared_band_strength_survives_a_selection_strength() -> None:
    assert _resolve("0.93@0.5", strength=0.1) == (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_the_final_level_is_never_a_step_of_its_own() -> None:
    # Nine levels drive eight steps, so a band holding only zero holds no step.
    with pytest.raises(ValueError, match="selects no step"):
        _resolve("0:0")


def test_a_zero_strength_means_the_step_applies_nothing() -> None:
    # A band that holds levels and declares zero is a deliberate carve-out, not a
    # mistake, so it stands and those steps simply contribute nothing.
    assert _resolve("0.93@1,0.6:0.93@0,0:0.6@0.3") == (
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3,
        0.3,
    )
    assert _resolve("0:1@0") == (0.0,) * 8
    assert _resolve("0.93", strength=0.0) == (0.0,) * 8


def test_a_window_that_selects_nothing_is_refused_with_its_bands_named() -> None:
    with pytest.raises(ValueError, match=r"selects no step.*0\.99 to 1"):
        _resolve("0.99:1", sigmas=(0.5, 0.2, 0.0))


def test_resolution_needs_a_schedule_with_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        _resolve("0.5", sigmas=(1.0,))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("0.5:0.4", "above high"),
        ("1.5", "between 0 and 1"),
        ("-0.1", "between 0 and 1"),
        ("a", "unreadable"),
        ("0.5@", "must be"),
        ("0.5@x", "unreadable"),
        ("", "one or more bands"),
        ("0.5,,0.2", "one or more bands"),
        ("0.1:0.2:0.3", "must be"),
    ],
)
def test_malformed_specs_are_refused(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sigma_window(spec)


def test_a_window_needs_at_least_one_band() -> None:
    with pytest.raises(ValueError, match="at least one band"):
        AdapterSigmaWindow(())


def test_a_band_strength_must_be_finite() -> None:
    with pytest.raises(ValueError, match="strength must be finite"):
        AdapterSigmaBand(0.5, 1.0, float("nan"))


def test_metadata_records_every_band_verbatim() -> None:
    assert parse_sigma_window("0.93,0.6:0.93@0.4").to_metadata() == {
        "bands": [
            {"low": 0.93, "high": 1.0, "strength": None},
            {"low": 0.6, "high": 0.93, "strength": 0.4},
        ]
    }


def test_rendering_names_whichever_spelling_was_used() -> None:
    from kinomlx.adapters.schedule import AdapterSchedule

    assert render_strength_plan(None, None) is None
    assert render_strength_plan(AdapterSchedule((1.0, 0.0)), None) == ("step strengths", "1, 0")
    window = parse_sigma_window("0.93,0.6:0.93@0.4")
    assert render_strength_plan(None, window) == (
        "sigma window",
        "0.93 to 1; 0.6 to 0.93 at 0.4",
    )


def test_a_selection_reports_a_window_as_a_varying_strength_plan(tmp_path) -> None:
    selection = ScheduledAdapterInput(
        tmp_path / "a.safetensors", strength=0.8, sigma_window="0.93"
    ).selection()
    assert selection.sigma_window is not None
    assert selection.schedule is None
    assert selection.varies
    assert selection.strength_plan is selection.sigma_window
    # A window pairs with a constant strength; it is the magnitude inside the bands.
    assert selection.strength == 0.8
    assert selection.effective


def test_bands_that_declare_their_own_strength_outlive_a_zero_fallback(tmp_path) -> None:
    # Writing every band's magnitude into the window means the scalar is never
    # read, so a zero there is not a request to drop the adapter.
    banded = ScheduledAdapterInput(
        tmp_path / "a.safetensors", strength=0.0, sigma_window="0.93@1,0.6:0.93@0.4"
    ).selection()
    assert banded.effective
    assert banded.effective_strength == 1.0
    # A window whose bands all defer to the scalar still reads a zero as off.
    deferring = ScheduledAdapterInput(
        tmp_path / "a.safetensors", strength=0.0, sigma_window="0.93"
    ).selection()
    assert not deferring.effective
    assert deferring.effective_strength == 0.0
    # A deliberate zero carve-out in the leading band is reported as that zero.
    carved = ScheduledAdapterInput(
        tmp_path / "a.safetensors", strength=0.6, sigma_window="0.93@0,0.6:0.93"
    ).selection()
    assert carved.effective
    assert carved.effective_strength == 0.0


def test_a_window_and_a_step_array_are_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="not both"):
        ScheduledAdapterInput(
            tmp_path / "a.safetensors", step_strengths=(1.0, 0.0), sigma_window="0.93"
        ).selection()


def test_a_window_needs_no_step_count_so_the_early_gate_skips_it(tmp_path) -> None:
    selection = AdapterSelection(
        tmp_path / "a.safetensors", 1.0, sigma_window=parse_sigma_window("0.93")
    )
    for steps in (1, 4, 8, 52):
        require_adapter_schedule_steps((selection,), steps)


def test_a_window_whose_bands_all_apply_zero_is_not_effective(tmp_path) -> None:
    path = tmp_path / "gate.safetensors"
    zero_bands = AdapterSigmaWindow(
        (AdapterSigmaBand(0.9, 1.0, 0.0), AdapterSigmaBand(0.5, 0.9, 0.0))
    )
    assert AdapterSelection(path, 1.0, sigma_window=zero_bands).effective is False
    # A band without its own strength applies the scalar fallback.
    fallback = AdapterSigmaWindow((AdapterSigmaBand(0.9, 1.0, 0.0), AdapterSigmaBand(0.5, 0.9)))
    assert AdapterSelection(path, 1.0, sigma_window=fallback).effective is True
    assert AdapterSelection(path, 0.0, sigma_window=fallback).effective is False
    owned = AdapterSigmaWindow((AdapterSigmaBand(0.9, 1.0, 0.4),))
    assert AdapterSelection(path, 0.0, sigma_window=owned).effective is True
