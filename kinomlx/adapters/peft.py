"""PEFT sidecar configuration and serialization semantics, without PEFT."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from kinomlx._json import JsonObject

from .algebra import ia3_input_scale
from .configuration import boolean, matches, module_name, number, pattern_value
from .formats import (
    AdapterInventory,
    AdapterKind,
    AdapterOperation,
    FourierOptions,
    GLoRAOptions,
    OFTOptions,
    SavedParameter,
    parse_adapter,
    split_factor_key,
    validate_inventory_values,
)
from .peft_formats import PEFT_FAMILIES, parse_peft_adapter
from .semantics import training_only_keys, validate_static_config

PEFT_WEIGHTS_NAME = "adapter_model.safetensors"
PEFT_CONFIG_NAME = "adapter_config.json"
# A selection whose links run longer than this is treated as a loop.
_MAX_LINK_HOPS = 32


def selected_link_chain(path: Path) -> tuple[Path, ...]:
    """Return the selected path and each link it passes through, none of them resolved.

    A Hugging Face snapshot names its files as links into a content-addressed blob
    store, so the resolved weights file has a digest for a name and no neighbors.
    The names the PEFT layout relies on survive only along the selection's links.
    """
    chain = [path.expanduser().absolute()]
    while chain[-1].is_symlink() and len(chain) <= _MAX_LINK_HOPS:
        chain.append(chain[-1].parent / chain[-1].readlink())
    return tuple(chain)


def companion_config_path(path: Path) -> Path | None:
    """Find the configuration beside the standard PEFT weights filename, along links."""
    for hop in selected_link_chain(path):
        companion = hop.with_name(PEFT_CONFIG_NAME)
        if hop.name == PEFT_WEIGHTS_NAME and companion.is_file():
            return companion
    return None


def companion_config(path: Path) -> JsonObject:
    """Read the PEFT configuration beside a selected weights file, or ``{}`` without one.

    Only the standard PEFT weights filename consumes adjacent configuration. A
    missing one is not an error here: the caller decides what an export without
    its configuration means.
    """
    companion = companion_config_path(path)
    if companion is None:
        return {}
    if companion.stat().st_size > 1024 * 1024:
        raise ValueError("adapter_config.json exceeds the 1 MiB configuration limit")
    try:
        value = json.loads(companion.read_text())
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid adapter_config.json") from exc
    if not isinstance(value, dict) or not isinstance(value.get("peft_type"), str):
        raise ValueError("adapter_config.json must contain an object declaring peft_type")
    return cast(JsonObject, value)


def parse_configured_adapter(
    weights: Mapping[str, mx.array], config: JsonObject, *, validate_values: bool = True
) -> AdapterInventory:
    """Identify declared saved modules/biases without guessing from bare weights."""
    validate_static_config(config)
    training_keys = training_only_keys(weights, config, validate_values=validate_values)
    if training_keys:
        weights = {key: value for key, value in weights.items() if key not in training_keys}
    modules = config.get("modules_to_save") or []
    if not isinstance(modules, list) or not all(isinstance(name, str) and name for name in modules):
        raise ValueError("PEFT modules_to_save must be a list of nonempty module names")
    bias_mode = config.get("bias", "none")
    if not isinstance(bias_mode, str) or (
        bias_mode not in ("none", "all") and not bias_mode.endswith("_only")
    ):
        raise ValueError("PEFT bias must be none, all, or a tuner-only selection")
    saved = {}
    for key in weights:
        factor = split_factor_key(key)
        if factor is not None and factor[1] != "weight":
            continue
        if ".original_module." in key:
            continue
        wrapper = re.fullmatch(r"(.*)\.modules_to_save\.([^.]+)\.(.+)", key)
        normalized = key if wrapper is None else f"{wrapper[1]}.{wrapper[3]}"
        name = module_name(normalized)
        is_saved = any(f".{module}." in f".{name}." for module in modules)
        is_bias = normalized.endswith(".bias")
        if not is_saved and not (is_bias and bias_mode != "none"):
            continue
        part: Literal["weight", "bias"] = "bias" if is_bias else "weight"
        prefix = normalized.removesuffix(f".{part}").removesuffix(".base_layer")
        saved[key] = SavedParameter(prefix, part, None if wrapper is None else wrapper[2])
    variant = None
    if config.get("peft_type") == "LORA":
        if config.get("use_bdlora") and config.get("use_qalora"):
            raise ValueError("block-diagonal LoRA and QALoRA cannot be combined")
        variant = (
            "BDLORA" if config.get("use_bdlora") else "QALORA" if config.get("use_qalora") else None
        )
        if variant is not None and (config.get("use_dora") or training_keys):
            raise ValueError("grouped LoRA export combines incompatible variants")
    if variant is not None or config.get("peft_type") in PEFT_FAMILIES:
        inventory = parse_peft_adapter(
            weights, config, saved, family=variant, validate_values=validate_values
        )
    else:
        parsed = parse_adapter(weights, saved_parameters=saved, validate_values=validate_values)
        inventory = replace(
            parsed, operations=tuple(configured_operation(op, config) for op in parsed.operations)
        )
    # bias=all can save both the trained base bias and a LoRA B bias. They
    # compose as a replacement plus the scaled low-rank bias, not two aliases.
    base_biases = {
        op.prefix: op
        for op in inventory.operations
        if op.kind == AdapterKind.SET and op.part == "bias"
    }
    combined = []
    consumed_biases = set()
    for op in inventory.operations:
        replacement = base_biases.get(op.prefix)
        if replacement is not None and (
            (op.kind in (AdapterKind.LORA, AdapterKind.BEFT) and op.part == "bias")
            or op.kind in (AdapterKind.GLORA_PEFT, AdapterKind.ROAD)
            or (op.kind == AdapterKind.IA3_PEFT and not ia3_input_scale(op))
        ):
            op = replace(
                op,
                factors={**op.factors, "base_bias": replacement.factors["delta"]},
                source_keys=(*op.source_keys, *replacement.source_keys),
            )
            consumed_biases.add(replacement.prefix)
        combined.append(op)
    return replace(
        inventory,
        training_keys=training_keys,
        operations=tuple(
            op
            for op in combined
            if not (
                op.kind == AdapterKind.SET and op.part == "bias" and op.prefix in consumed_biases
            )
        ),
    )


def configured_operation(operation: AdapterOperation, config: JsonObject) -> AdapterOperation:
    """Apply the declared family's inference scaling, never filename guesses."""
    if not config:
        return operation
    family = config.get("peft_type")
    families = {
        "lora": "LORA",
        "loha": "LOHA",
        "lokr": "LOKR",
        "adalora": "ADALORA",
        "ia3_peft": "IA3",
        "oft_peft": "OFT",
        "boft_peft": "BOFT",
        "hra": "HRA",
        "glora_peft": "GLORA",
        "fourierft": "FOURIERFT",
    }
    if family not in families.values():
        raise ValueError(f"PEFT {family!r} has no implemented configuration")
    if config.get("fan_in_fan_out"):
        raise ValueError("PEFT fan_in_fan_out targets Conv1D storage, not linear source weights")
    if operation.kind in families and family != families[operation.kind]:
        raise ValueError(f"PEFT {operation.kind} factors disagree with peft_type {family!r}")
    if operation.kind == AdapterKind.GLORA_PEFT:
        rank = number(config.get("r", 8), name="GLoRA rank")
        weight_path = config.get("config_A_B", "lora")
        coupling_path = config.get("config_C", "lora")
        bias_path = config.get("config_D_E", "constant")
        if rank <= 0 or not rank.is_integer():
            raise ValueError("PEFT GLoRA rank must be a positive integer")
        if weight_path not in ("lora", "vector", "constant", "none"):
            raise ValueError("invalid PEFT GLoRA config_A_B")
        if coupling_path not in ("lora", "vector", "none"):
            raise ValueError("invalid PEFT GLoRA config_C")
        if bias_path not in ("vector", "constant", "none"):
            raise ValueError("invalid PEFT GLoRA config_D_E")
        return replace(
            operation,
            glora_options=GLoRAOptions(
                int(rank), str(weight_path), str(coupling_path), str(bias_path)
            ),
        )
    if operation.kind == AdapterKind.FOURIERFT:
        count = number(
            pattern_value(
                config, "n_frequency_pattern", operation.prefix, config.get("n_frequency", 1000)
            ),
            name="n_frequency",
        )
        seed = config.get("random_loc_seed", 777)
        scaling = number(config.get("scaling", 150.0), name="FourierFT scaling")
        if (
            not count.is_integer()
            or count <= 0
            or operation.factors["spectrum"].shape != (int(count),)
        ):
            raise ValueError("FourierFT spectrum disagrees with configured n_frequency")
        if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**64:
            raise ValueError("FourierFT seed must fit the Torch CPU generator range")
        return replace(
            operation,
            scalar=operation.scalar * scaling,
            fourier_options=FourierOptions(int(count), int(seed)),
        )
    if operation.kind == AdapterKind.HRA:
        vectors = operation.factors["hra_vectors"]
        rank = number(config.get("r", 8), name="HRA rank")
        if vectors.ndim != 2 or rank <= 0 or rank != vectors.shape[1]:
            raise ValueError("PEFT HRA vectors disagree with configured rank")
        return replace(operation, hra_apply_gs=boolean(config, "apply_GS", False))
    if operation.kind == AdapterKind.OFT_PEFT:
        terms = number(config.get("num_cayley_neumann_terms", 5), name="Neumann terms")
        eps = number(config.get("eps", 6e-5), name="eps")
        if not terms.is_integer() or not 1 <= terms <= 64 or eps < 0:
            raise ValueError("PEFT OFT needs 1-64 Neumann terms and nonnegative eps")
        return replace(
            operation,
            oft_options=OFTOptions(
                block_share=boolean(config, "block_share", False),
                coft=boolean(config, "coft", False),
                eps=eps,
                use_neumann=boolean(config, "use_cayley_neumann", True),
                neumann_terms=int(terms),
            ),
        )
    if operation.kind == AdapterKind.IA3_PEFT:
        if family != "IA3":
            raise ValueError("PEFT IA3 factors disagree with peft_type")
        modules = config.get("feedforward_modules")
        if modules is None:
            return operation
        name = module_name(operation.prefix)
        if isinstance(modules, str):
            on_input = matches(modules, name, suffix=False)
        elif isinstance(modules, list) and all(isinstance(value, str) for value in modules):
            on_input = any(name == value or name.endswith(f".{value}") for value in modules)
        else:
            raise ValueError("PEFT feedforward_modules must be a string or string list")
        return replace(operation, input_scale=on_input)
    if operation.kind == AdapterKind.ADALORA:
        if family != "ADALORA":
            raise ValueError("AdaLoRA singular-value factors disagree with peft_type")
        rank = operation.factors["down"].shape[0]
        initial = number(config.get("init_r", rank), name="init_r")
        if initial < rank or not initial.is_integer():
            raise ValueError("AdaLoRA init_r must be an integer at least the retained rank")
        patterns = config.get("rank_pattern")
        if patterns is not None:
            if not isinstance(patterns, dict):
                raise ValueError("AdaLoRA rank_pattern must be an object")
            for key, mask in patterns.items():
                prefix = re.sub(r"\.lora_E(?:\.[^.]+)?$", "", key)
                if module_name(prefix) != module_name(operation.prefix):
                    continue
                if (
                    not isinstance(mask, list)
                    or not all(isinstance(bit, bool) for bit in mask)
                    or len(mask) != int(initial)
                    or sum(mask) != rank
                ):
                    raise ValueError("AdaLoRA retained factors disagree with rank_pattern")
        return replace(
            operation,
            alpha=operation.alpha
            if operation.alpha is not None
            else number(config.get("lora_alpha", 8), name="alpha"),
            rank_denominator=initial
            if operation.rank_denominator is None
            else operation.rank_denominator,
        )
    if (
        config.get("use_dora")
        and operation.kind == AdapterKind.LORA
        and operation.part == "weight"
        and "magnitude" not in operation.factors
    ):
        raise ValueError(f"PEFT DoRA target {operation.prefix!r} is missing its magnitude vector")
    if operation.kind not in (AdapterKind.LORA, AdapterKind.LOHA, AdapterKind.LOKR):
        return operation
    if family == "ADALORA" and operation.kind == AdapterKind.LORA:
        raise ValueError("PEFT AdaLoRA is missing its lora_E singular-value factor")
    if operation.kind == AdapterKind.LORA:
        actual_rank = operation.factors["down"].shape[0]
        declared_rank = pattern_value(config, "rank_pattern", operation.prefix, config.get("r"))
        if declared_rank is not None and number(declared_rank, name="rank") != actual_rank:
            raise ValueError("PEFT factors disagree with configured rank")
    if operation.kind == AdapterKind.LOKR:
        configured_rank = number(
            pattern_value(config, "rank_pattern", operation.prefix, config.get("r", 8)),
            name="rank",
        )
        if configured_rank <= 0 or not configured_rank.is_integer():
            raise ValueError("PEFT LoKr requires a positive configured rank")
        operation = replace(operation, rank_denominator=configured_rank)
    alpha: object = operation.alpha
    if alpha is None:
        alpha = pattern_value(
            config,
            "alpha_pattern",
            operation.prefix,
            config.get("lora_alpha", config.get("alpha", 8)),
        )
    if alpha is None:
        return operation
    resolved = number(alpha, name="alpha")
    if config.get("use_rslora"):
        if operation.kind != AdapterKind.LORA:
            raise ValueError("rank-stabilized scaling is only defined here for LoRA")
        resolved *= math.sqrt(operation.factors["down"].shape[0])
    return replace(operation, alpha=resolved)


def parse_lazy_adapter(weights: Mapping[str, mx.array], config: JsonObject) -> AdapterInventory:
    """Defer only ordinary LoRA/LoKr numerical scans; preserve other format checks."""
    inventory = parse_configured_adapter(weights, config, validate_values=False)
    if all(
        (op.kind == AdapterKind.LOKR and "magnitude" not in op.factors)
        or (
            op.kind == AdapterKind.LORA
            and "magnitude" not in op.factors
            and "mid" not in op.factors
        )
        for op in inventory.operations
    ):
        return inventory
    allow_empty = {
        key
        for key in weights
        if config.get("peft_type") == "VBLORA"
        and config.get("topk") == 1
        and key.endswith("_topk_weights")
    }
    validate_inventory_values(weights, inventory, allow_empty=allow_empty)
    return inventory
