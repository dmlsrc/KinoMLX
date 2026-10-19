"""The intrinsic-scale determination adapters report beneath user strength."""

import logging

import mlx.core as mx
import pytest

from kinomlx.adapters.peft import parse_configured_adapter
from kinomlx.adapters.scaling import (
    ALPHA_TENSOR_SOURCE,
    ScaleDetermination,
    determine_scale,
    log_scale_determination,
)

_LOGGER = "tests.adapters.scaling"


def _pair(prefix: str, rank: int) -> dict[str, mx.array]:
    return {
        f"{prefix}.lora_A.weight": mx.ones((rank, 4)),
        f"{prefix}.lora_B.weight": mx.ones((4, rank)),
    }


def test_a_configuration_sets_the_scale_and_names_its_fields() -> None:
    config = {"peft_type": "LORA", "r": 32, "lora_alpha": 64}
    inventory = parse_configured_adapter(_pair("base_model.model.layer", 32), config)
    scale = determine_scale("adapter_config.json", config, inventory.operations)
    assert scale == ScaleDetermination("adapter_config.json", (2.0,), "lora_alpha=64, r=32")


def test_rank_patterns_and_rank_stabilization_are_reported() -> None:
    config = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8,
        "use_rslora": True,
        "rank_pattern": {"wide": 16},
        "alpha_pattern": {},
    }
    weights = {**_pair("base_model.model.narrow", 4), **_pair("base_model.model.wide", 16)}
    inventory = parse_configured_adapter(weights, config)
    scale = determine_scale("adapter_config.json", config, inventory.operations)
    assert scale.scales == pytest.approx((8 / 16**0.5, 8 / 4**0.5))
    assert scale.terms == "lora_alpha=8, r=4, use_rslora, rank_pattern (1 entry)"


def test_a_unit_configuration_still_counts_as_a_determination() -> None:
    config = {"peft_type": "LORA", "r": 8, "lora_alpha": 8}
    inventory = parse_configured_adapter(_pair("base_model.model.layer", 8), config)
    assert determine_scale("adapter_config.json", config, inventory.operations).scales == (1.0,)


def test_alpha_tensors_count_only_when_they_move_the_scale() -> None:
    scaled = {**_pair("layer", 16), "layer.alpha": mx.array(4.0)}
    unit = {**_pair("layer", 16), "layer.alpha": mx.array(16.0)}
    operations = parse_configured_adapter(scaled, {}).operations
    assert determine_scale(None, {}, operations) == ScaleDetermination(ALPHA_TENSOR_SOURCE, (0.25,))
    assert determine_scale(None, {}, parse_configured_adapter(unit, {}).operations) == (
        ScaleDetermination()
    )
    plain = parse_configured_adapter(_pair("layer", 16), {}).operations
    assert determine_scale(None, {}, plain) == ScaleDetermination()


def test_the_console_line_states_the_source_and_the_effective_scale(caplog) -> None:
    logger = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_scale_determination(
            logger,
            model="LTX-2.3",
            name="adapter_model.safetensors",
            determination=ScaleDetermination("adapter_config.json", (2.0,), "lora_alpha=64, r=32"),
            strength=1.5,
        )
        log_scale_determination(
            logger,
            model="Krea 2",
            name="mixed.safetensors",
            determination=ScaleDetermination(ALPHA_TENSOR_SOURCE, (0.5, 1.0)),
            strength=2.0,
        )
        log_scale_determination(
            logger, model="Klein", name="plain", determination=ScaleDetermination(), strength=1.0
        )
    assert caplog.messages == [
        "LTX-2.3 adapter adapter_model.safetensors: scale 2 from adapter_config.json "
        "(lora_alpha=64, r=32); effective 3 at strength 1.5",
        "Krea 2 adapter mixed.safetensors: scales 0.5 to 1 from alpha tensors; "
        "effective 1 to 2 at strength 2",
    ]
