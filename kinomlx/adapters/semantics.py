"""Explicit boundaries between static inference and omitted trainer state."""

import re
from collections.abc import Mapping

import mlx.core as mx

from kinomlx._json import JsonObject

from .formats import validate_adapter_values

_TRAINER_STATE = {
    "ADAMSS": "cluster row assignments and initialization subspaces",
    "FROD": "the frozen per-target U basis",
    "OSF": "the frozen high-singular-value basis",
    "PSOFT": "the original frozen A/B initialization bases",
}
_DYNAMIC = {
    "LILY": "input-dependent routing",
    "POLY": "task-dependent adapter routing",
    "XLORA": "input-dependent adapter routing",
    "SHADOW": "a parallel hidden-state path and input-dependent gates",
}
_PROMPT = frozenset(
    {
        "PROMPT_TUNING",
        "MULTITASK_PROMPT_TUNING",
        "P_TUNING",
        "PREFIX_TUNING",
        "ADAPTION_PROMPT",
        "CPT",
        "TRAINABLE_TOKENS",
        "CARTRIDGE",
    }
)


def validate_static_config(config: JsonObject) -> None:
    if not config:
        return
    family = config.get("peft_type")
    if not isinstance(family, str):
        raise ValueError("adapter configuration must declare a string peft_type")
    if family in _TRAINER_STATE:
        raise ValueError(
            f"PEFT {family} exports omit {_TRAINER_STATE[family]}; "
            "convert in the producer environment to a static LoRA or dense delta "
            "against the original base checkpoint"
        )
    if family in _DYNAMIC:
        raise ValueError(f"PEFT {family} needs {_DYNAMIC[family]}, not a static weight merge")
    if family in _PROMPT:
        raise ValueError(
            f"PEFT {family} needs a prompt, embedding, or attention-runtime integration; "
            "the transformer weight-adapter loader cannot apply it"
        )
    if family != "LORA":
        return
    if config.get("alora_invocation_tokens") is not None:
        raise ValueError("aLoRA needs token-conditional activation, not a static weight merge")
    if config.get("arrow_config") is not None:
        raise ValueError("Arrow needs input-dependent routing, not a static weight merge")
    if config.get("kasa_config") is not None:
        raise ValueError(
            "KaSA changes the frozen base by SVD truncation; export a dense delta "
            "against the original base checkpoint in the producer environment"
        )
    initialization = config.get("init_lora_weights", True)
    if isinstance(initialization, str) and (
        initialization.lower().startswith(("pissa", "corda"))
        or initialization.lower() in ("olora", "loftq", "lora_ga")
    ):
        raise ValueError(
            f"LoRA initialization {initialization!r} changes the frozen base; re-export "
            "with the initialization delta included against the original checkpoint"
        )
    if not isinstance(initialization, bool) and initialization not in (
        "gaussian",
        "eva",
        "orthogonal",
        "mica",
    ):
        raise ValueError(
            "unknown LoRA initialization requires a converted export against the original checkpoint"
        )


def training_only_keys(
    weights: Mapping[str, mx.array], config: JsonObject, *, validate_values: bool = True
) -> tuple[str, ...]:
    """Account for declared training-only state without applying it to inference."""
    if config.get("peft_type") != "LORA":
        return ()
    patterns = []
    if config.get("velora_config") is not None:
        patterns.append(r".*\.lora_velora_embed(?:\.[^.]+)?")
    if config.get("monteclora_config") is not None:
        patterns.append(
            r".*\.lora_monteclora_sampler\.(?:[^.]+\.)?"
            r"(?:std_prior|expert_weights_prior|gaussian_var_prior|expert_weights)"
        )
    keys = tuple(sorted(key for key in weights if any(re.fullmatch(p, key) for p in patterns)))
    if keys and validate_values:
        validate_adapter_values({key: weights[key] for key in keys})
    return keys
