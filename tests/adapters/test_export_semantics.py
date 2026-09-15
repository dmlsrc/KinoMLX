"""Complete exports or actionable errors, never guessed training state."""

import mlx.core as mx
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.peft import parse_configured_adapter


def _lora():
    return {"layer.lora_A": mx.ones((2, 4)), "layer.lora_B": mx.ones((8, 2))}


@pytest.mark.parametrize("family", ["ADAMSS", "FROD", "OSF", "PSOFT"])
def test_missing_initialization_state_requests_conversion(family):
    with pytest.raises(ValueError, match="exports omit.*convert.*original base"):
        parse_configured_adapter({}, {"peft_type": family})


@pytest.mark.parametrize("family", ["LILY", "POLY", "XLORA", "SHADOW"])
def test_dynamic_tuners_are_not_approximated_as_a_static_delta(family):
    with pytest.raises(ValueError, match="not a static weight merge"):
        parse_configured_adapter({}, {"peft_type": family})


@pytest.mark.parametrize(
    "family",
    [
        "PROMPT_TUNING",
        "MULTITASK_PROMPT_TUNING",
        "P_TUNING",
        "PREFIX_TUNING",
        "ADAPTION_PROMPT",
        "CPT",
        "TRAINABLE_TOKENS",
        "CARTRIDGE",
    ],
)
def test_prompt_and_embedding_adapters_need_their_own_integration(family):
    with pytest.raises(ValueError, match="transformer weight-adapter loader cannot apply"):
        parse_configured_adapter({}, {"peft_type": family})


@pytest.mark.parametrize(
    "initialization", ["pissa", "pissa_niter_4", "corda", "olora", "loftq", "lora_ga"]
)
def test_changed_base_lora_requires_original_base_conversion(initialization):
    with pytest.raises(ValueError, match="changes the frozen base.*re-export"):
        parse_configured_adapter(
            _lora(), {"peft_type": "LORA", "init_lora_weights": initialization}
        )


@pytest.mark.parametrize("initialization", [True, False, "gaussian", "eva", "orthogonal", "mica"])
def test_saved_lora_factors_do_not_replay_harmless_initialization(initialization):
    inventory = parse_configured_adapter(
        _lora(), {"peft_type": "LORA", "r": 2, "lora_alpha": 2, "init_lora_weights": initialization}
    )
    delta = reconstruct_delta(inventory.operations[0], mx.zeros((8, 4)))
    assert mx.array_equal(delta, mx.full((8, 4), 2.0)).item()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("alora_invocation_tokens", [1, 2], "token-conditional"),
        ("arrow_config", {}, "input-dependent"),
        ("kasa_config", {}, "SVD truncation"),
    ],
)
def test_lora_runtime_variants_cannot_silently_use_plain_fusion(field, value, error):
    with pytest.raises(ValueError, match=error):
        parse_configured_adapter(_lora(), {"peft_type": "LORA", field: value})


@pytest.mark.parametrize("family", [None, [], {}, 123])
def test_config_family_has_a_clean_diagnostic(family):
    with pytest.raises(ValueError, match="peft_type"):
        parse_configured_adapter(_lora(), {"peft_type": family})


def test_unrepresentable_configuration_number_has_a_clean_error():
    with pytest.raises(ValueError, match="finite number"):
        parse_configured_adapter(_lora(), {"peft_type": "LORA", "lora_alpha": 10**1000})


@pytest.mark.parametrize("key", ["base_model.tinylora_v.1", "base_model.tinylora_v.style.2"])
def test_tinylora_multiple_groups_require_the_unsaved_mapping(key):
    with pytest.raises(ValueError, match="multi-group module mapping.*convert"):
        parse_configured_adapter({key: mx.ones(3)}, {"peft_type": "TINYLORA"})


def test_tinylora_requires_saved_projections():
    weights = {
        "layer.tinylora_A": mx.ones((2, 4)),
        "layer.tinylora_B": mx.ones((8, 2)),
        "base_model.tinylora_v.0": mx.ones(3),
    }
    with pytest.raises(ValueError, match="save_projection=true"):
        parse_configured_adapter(weights, {"peft_type": "TINYLORA", "r": 2, "u": 3})


def test_tinylora_missing_second_group_cannot_reuse_the_first_silently():
    weights = {"base_model.tinylora_v.0": mx.ones(3)}
    for layer in ("first", "second"):
        weights.update(
            {
                f"{layer}.tinylora_A": mx.ones((2, 4)),
                f"{layer}.tinylora_B": mx.ones((8, 2)),
                f"{layer}.tinylora_P": mx.ones((3, 2, 2)),
            }
        )
    with pytest.raises(ValueError, match="requires multiple groups.*module mapping"):
        parse_configured_adapter(weights, {"peft_type": "TINYLORA", "r": 2, "u": 3})


def test_inconsistent_wavelet_table_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="48 taps.*corrected dense"):
        parse_configured_adapter(
            {"layer.waveft_spectrum": mx.ones(3)},
            {"peft_type": "WAVEFT", "n_frequency": 3, "wavelet_family": "coif5"},
        )


@pytest.mark.parametrize("family", ["BEFT", "ROAD", "HIRA"])
def test_saved_parameters_cannot_have_integer_weights(family):
    with pytest.raises(ValueError, match="saved parameter.*floating point"):
        parse_configured_adapter(
            {"layer.weight": mx.ones((8, 4), dtype=mx.int32)},
            {"peft_type": family, "modules_to_save": ["layer"]},
        )


@pytest.mark.parametrize("activation", ["prelu", "xielu"])
def test_parameterized_peanut_activations_need_complete_export(activation):
    with pytest.raises(ValueError, match="parameterized activation state"):
        parse_configured_adapter(
            {"layer.peanut_A.weight": mx.ones((2, 8)), "layer.peanut_B.weight": mx.ones((8, 2))},
            {"peft_type": "PEANUT", "r": 2, "act_fn": activation},
        )
