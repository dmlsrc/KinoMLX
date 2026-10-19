"""Self-contained PEFT Linear exports beyond the ordinary low-rank formats.

Only a declared companion configuration selects these signatures. Shared
projection tensors keep their original identities and are accounted once,
even when several target operations depend on them.
"""

import re
from collections.abc import Mapping
from typing import Literal, cast

import mlx.core as mx

from kinomlx._json import JsonObject

from .configuration import boolean, integer, module_name, number, pattern_value
from .formats import (
    AdapterInventory,
    AdapterKind,
    AdapterOperation,
    LinearAdapterOptions,
    SavedParameter,
    validate_adapter_values,
)
from .wavelet_filters import LOWPASS

_FIELDS = {
    "BEFT": ("beft_bias",),
    "HIRA": ("hira_A", "hira_B"),
    "DELORA": ("delora_A", "delora_B", "delora_lambda", "delora_w_norm"),
    "C3A": ("c3a_kernel",),
    "MISS": ("miss_block",),
    "ROAD": ("road_theta", "road_alpha"),
    "GRALORA": ("gralora_A", "gralora_B", "gralora_A_general", "gralora_B_general"),
    "PEANUT": ("peanut_A", "peanut_B"),
    "DEFT": ("deft_P", "deft_R"),
    "SHIRA": ("shira_weight", "shira_indices"),
    "SUPERTUNING": (
        "supertuning_values",
        "supertuning_indices",
        "supertuning_lora_A",
        "supertuning_lora_B",
    ),
    "VERA": ("vera_lambda_d", "vera_lambda_b"),
    "PVERA": ("pvera_lambda_d", "pvera_lambda_b"),
    "RANDLORA": ("randlora_lambda", "randlora_gamma", "randlora_A", "randlora_B"),
    "VBLORA": ("vblora_logits_A", "vblora_logits_B"),
    "UNILORA": ("unilora_indices_A", "unilora_indices_B", "unilora_scales_A", "unilora_scales_B"),
    "TINYLORA": ("tinylora_A", "tinylora_B", "tinylora_P"),
    "WAVEFT": ("waveft_spectrum",),
    "BDLORA": ("lora_A", "lora_B"),
    "QALORA": ("lora_A", "lora_B"),
}
PEFT_FAMILIES = frozenset((*_FIELDS, "LN_TUNING")) - {"BDLORA", "QALORA"}
_SHARED = {
    "VERA": ("A", "B"),
    "PVERA": ("A", "B"),
    "RANDLORA": ("A", "B"),
    "VBLORA": ("vector_bank",),
    "UNILORA": ("theta_d",),
    "TINYLORA": ("v",),
}


def _split(key: str, family: str) -> tuple[str, str, str | None] | None:
    if family == "LN_TUNING":
        match = re.fullmatch(r"(.*)\.ln_tuning_layers\.(?:([^.]+)\.)?(weight|bias)", key)
        return None if match is None else (match[1], match[3], match[2])
    if family == "PEANUT":
        match = re.fullmatch(
            r"(.*)\.peanut_(encoders|decoders)\.(?:([^.]+)\.)?([0-9]+)\.weight", key
        )
        if match is not None:
            return match[1], f"{match[2]}.{int(match[4])}", match[3]
    if family == "VBLORA":
        match = re.fullmatch(
            r"(.*)\.vblora_logits_([AB])(?:\.([^.]+))?_topk_(indices|weights)", key
        )
        if match is not None:
            return match[1], f"logits_{match[2]}_topk_{match[4]}", match[3]
    fields = "|".join(re.escape(field) for field in _FIELDS[family])
    match = re.fullmatch(rf"(.*)\.({fields})(?:\.([^.]+))?(?:\.weight)?", key)
    if match is None:
        return None
    adapter_name = None if match[3] == "weight" else match[3]
    stem = "lora" if family in ("BDLORA", "QALORA") else family.lower()
    return match[1], match[2].removeprefix(f"{stem}_"), adapter_name


def _shared_key(key: str, family: str) -> tuple[str, str | None] | None:
    if family == "TINYLORA":
        match = re.fullmatch(r"base_model\.tinylora_v\.(?:([^.]+)\.)?([0-9]+)", key)
        if match is None:
            return None
        if int(match[2]) != 0:
            raise ValueError(
                "TinyLoRA export omits its multi-group module mapping; convert to LoRA "
                "in the producer environment instead of guessing module traversal order"
            )
        return "v", match[1]
    fields = _SHARED.get(family)
    if fields is None:
        return None
    field_pattern = "|".join(fields)
    match = re.fullmatch(rf"base_model\.{family.lower()}_({field_pattern})(?:\.([^.]+))?", key)
    return None if match is None else (match[1], match[2])


def _require(factors: Mapping[str, mx.array], required: set[str], family: str) -> None:
    if set(factors) != required:
        raise ValueError(
            f"PEFT {family}: missing factors {sorted(required - factors.keys())}, "
            f"extra factors {sorted(factors.keys() - required)}"
        )


def _rank(config: JsonObject, prefix: str, default: int) -> int:
    return integer(
        pattern_value(config, "rank_pattern", prefix, config.get("r", default)), name="rank"
    )


def index_bounds(value: mx.array, *, name: str) -> tuple[int, int]:
    """Inspect scratch index data once; structural checks never realize live factors."""
    if value.dtype == mx.bool_ or not value.size:
        raise ValueError(f"PEFT {name} must contain nonempty integer indices")
    if value.dtype in (mx.float16, mx.bfloat16, mx.float32):
        # Some exporters store COO indices as FP32. Accept only exactly integral
        # values in the safe int32 range, never silently truncate a coordinate.
        valid = mx.all((value == mx.floor(value)) & (value >= 0) & (value <= 2147483647))
        if not bool(valid.item()):
            raise ValueError(f"PEFT {name} has non-integral or out-of-range indices")
    low, high = mx.min(value), mx.max(value)
    mx.eval(low, high)
    bounds = int(cast(int, low.item())), int(cast(int, high.item()))
    if bounds[0] < 0 or bounds[1] > 2147483647:
        raise ValueError(f"PEFT {name} indices must fit nonnegative int32")
    return bounds


def _options(
    family: str, prefix: str, factors: Mapping[str, mx.array], config: JsonObject
) -> LinearAdapterOptions:
    if family in _SHARED:
        return _basis_options(family, prefix, factors, config)
    if family in ("BDLORA", "QALORA"):
        return _grouped_lora_options(family, prefix, factors, config)
    if family == "WAVEFT":
        _require(factors, {"spectrum"}, family)
        spectrum = factors["spectrum"]
        count = (
            spectrum.size
            if boolean(config, "proportional_parameters", False)
            else integer(
                pattern_value(
                    config, "n_frequency_pattern", prefix, config.get("n_frequency", 2592)
                ),
                name="n_frequency",
            )
        )
        if spectrum.shape != (count,):
            raise ValueError("WaveFT spectrum disagrees with n_frequency")
        seed = config.get("random_loc_seed", 777)
        if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**64:
            raise ValueError("WaveFT seed must fit the Torch CPU generator range")
        wavelet = config.get("wavelet_family", "db1")
        if not isinstance(wavelet, str):
            raise ValueError("WaveFT wavelet_family must be a string")
        wavelet = "db1" if wavelet.lower() == "haar" else wavelet.lower()
        use_idwt = boolean(config, "use_idwt", True)
        if use_idwt and wavelet == "coif5":
            raise ValueError(
                "the pinned PEFT coif5 table has 48 taps but declares 28-sample reduction; "
                "export a corrected dense WaveFT delta in the producer environment"
            )
        if use_idwt and wavelet not in LOWPASS:
            raise ValueError("WaveFT requires db1-db10, sym2-sym10, coif1-coif4, or haar")
        return LinearAdapterOptions(
            rank=count,
            seed=seed,
            variant=wavelet if use_idwt else "",
            scale=number(config.get("scaling", 25.0), name="scaling"),
        )
    if family == "BEFT":
        _require(factors, {"bias"}, family)
        return LinearAdapterOptions()
    if family == "HIRA":
        _require(factors, {"A", "B"}, family)
        return LinearAdapterOptions(rank=_rank(config, prefix, 32))
    if family == "DELORA":
        _require(factors, {"A", "B", "lambda", "w_norm"}, family)
        return LinearAdapterOptions(rank=_rank(config, prefix, 8))
    if family == "C3A":
        _require(factors, {"kernel"}, family)
        block = integer(
            pattern_value(config, "block_size_pattern", prefix, config.get("block_size", 256)),
            name="block_size",
        )
        return LinearAdapterOptions(block_size=block)
    if family == "MISS":
        _require(factors, {"block"}, family)
        variant = config.get("init_weights", True)
        if not isinstance(variant, bool) and variant not in ("bat", "mini"):
            raise ValueError("PEFT MISS init_weights must be boolean, bat, or mini")
        return LinearAdapterOptions(
            rank=integer(config.get("r", 64), name="rank"),
            block_size=integer(config.get("mini_r", 1), name="mini_r"),
            variant=variant if isinstance(variant, str) else "miss",
        )
    if family == "ROAD":
        _require(factors, {"theta", "alpha"}, family)
        variant = config.get("variant", "road_2")
        block = integer(config.get("group_size", 64), name="group_size")
        if variant not in ("road_1", "road_2", "road_4") or block % 2:
            raise ValueError("PEFT ROAD needs road_1/road_2/road_4 and an even group_size")
        return LinearAdapterOptions(block_size=block, variant=str(variant))
    if family == "GRALORA":
        rank = integer(config.get("r", 32), name="rank")
        hybrid = integer(config.get("hybrid_r", 0), name="hybrid_r", minimum=0)
        groups = integer(config.get("gralora_k", 2), name="gralora_k")
        if rank % groups:
            raise ValueError("PEFT GRALORA rank must be divisible by gralora_k")
        _require(factors, {"A", "B"} | ({"A_general", "B_general"} if hybrid else set()), family)
        return LinearAdapterOptions(
            rank=rank,
            hybrid_rank=hybrid,
            block_size=groups,
            scale=number(config.get("alpha", 64), name="alpha") / (rank + hybrid),
        )
    if family == "PEANUT":
        rank = integer(config.get("r", 32), name="rank")
        depth = integer(config.get("depth", 0), name="depth", minimum=0)
        # Validate the serialized depth against tensor count before expanding a
        # range supplied in untrusted JSON.
        if depth > len(factors) // 2:
            raise ValueError("PEFT PEANUT depth exceeds its saved encoder/decoder factors")
        activation = config.get("act_fn", "relu")
        if activation in ("prelu", "xielu"):
            raise ValueError(
                "PEANUT export omits parameterized activation state; convert to a dense delta in the producer environment"
            )
        if activation not in (
            "relu",
            "silu",
            "swish",
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
            "gelu_python_tanh",
            "gelu_python",
            "gelu_accurate",
            "gelu_10",
            "quick_gelu",
            "hardswish",
            "leaky_relu",
            "relu2",
            "relu6",
            "laplace",
            "mish",
            "sqrtsoftplus",
            "tanh",
            "sigmoid",
            "linear",
        ):
            raise ValueError(f"PEFT PEANUT activation {activation!r} has no MLX implementation")
        _require(
            factors,
            {"A", "B"} | {f"{side}.{i}" for side in ("encoders", "decoders") for i in range(depth)},
            family,
        )
        return LinearAdapterOptions(
            rank=rank,
            depth=depth,
            activation=str(activation),
            scale=number(config.get("scaling", 1.0), name="scaling"),
        )
    if family == "DEFT":
        projection_only = boolean(config, "para", False)
        method = config.get("decomposition_method", "relu")
        if method not in ("relu", "qr"):
            raise ValueError("PEFT DEFT decomposition_method must be relu or qr")
        _require(factors, {"P"} | (set() if projection_only else {"R"}), family)
        rank = integer(config.get("r", 8), name="rank")
        alpha = config.get("alpha")
        return LinearAdapterOptions(
            rank=rank,
            variant=str(method),
            projection_only=projection_only,
            # The effective rank is clipped to target output width in algebra.
            scale=0.0 if alpha is None else number(alpha, name="alpha"),
        )
    if family == "SHIRA":
        _require(factors, {"weight", "indices"}, family)
        indices = factors["indices"]
        if indices.ndim != 2 or indices.shape != (2, factors["weight"].size):
            raise ValueError("PEFT SHIRA needs two COO coordinates per saved value")
        return LinearAdapterOptions(
            index_bounds=tuple(index_bounds(row, name="SHIRA") for row in indices)
        )
    if family == "SUPERTUNING":
        if "indices" not in factors:
            raise ValueError(
                "SUPERTUNING requires saved indices; re-export with save_precomputed_indices=true"
            )
        rank_value = config.get("r")
        rank = 0 if rank_value is None else integer(rank_value, name="rank")
        _require(factors, {"values", "indices"} | ({"lora_A", "lora_B"} if rank else set()), family)
        if factors["indices"].ndim != 1 or factors["indices"].shape != factors["values"].shape:
            raise ValueError("PEFT SUPERTUNING needs one saved flat index per value")
        return LinearAdapterOptions(
            rank=rank,
            scale=number(config.get("lora_alpha", rank), name="lora_alpha") / rank if rank else 1.0,
            index_bounds=(index_bounds(factors["indices"], name="SUPERTUNING"),),
        )
    raise ValueError(f"PEFT {family} has no option interpreter")


def _grouped_lora_options(
    family: str, prefix: str, factors: Mapping[str, mx.array], config: JsonObject
) -> LinearAdapterOptions:
    _require(factors, {"A", "B"}, family)
    rank = _rank(config, prefix, 8)
    alpha = number(
        pattern_value(config, "alpha_pattern", prefix, config.get("lora_alpha", 8)), name="alpha"
    )
    scale = alpha / (rank**0.5 if boolean(config, "use_rslora", False) else rank)
    if family == "QALORA":
        return LinearAdapterOptions(
            rank=rank,
            scale=scale,
            block_size=integer(config.get("qalora_group_size", 16), name="qalora_group_size"),
        )
    block = config["use_bdlora"]
    if not isinstance(block, dict):
        raise ValueError("use_bdlora must contain a block-diagonal configuration object")
    selected = []
    for side in ("a", "b"):
        patterns = block.get(f"target_modules_bd_{side}") or []
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            raise ValueError("block-diagonal target patterns must be string lists")
        if any(p in module_name(prefix) for p in patterns):
            selected.append(side)
    if len(selected) > 1 or (not selected and boolean(block, "match_strict", True)):
        raise ValueError("block-diagonal LoRA target must select exactly one A/B factor")
    return LinearAdapterOptions(
        rank=rank,
        scale=scale,
        variant=selected[0] if selected else "",
        block_size=integer(block.get("nblocks", 1), name="nblocks"),
    )


def _basis_options(
    family: str, prefix: str, factors: Mapping[str, mx.array], config: JsonObject
) -> LinearAdapterOptions:
    if family == "TINYLORA":
        if "P" not in factors:
            raise ValueError(
                "TinyLoRA requires its saved projection; re-export with save_projection=true"
            )
        _require(factors, {"A", "B", "P", "v"}, family)
        count = integer(config.get("u", 64), name="u")
        rank = integer(config.get("r", 2), name="rank")
        if factors["P"].shape != (count, rank, rank) or factors["v"].shape != (count,):
            raise ValueError("TinyLoRA projection or shared vector disagrees with r/u")
        return LinearAdapterOptions(rank=rank)
    if family in ("VERA", "PVERA"):
        _require(factors, {"A", "B", "lambda_d", "lambda_b"}, family)
        if family == "PVERA" and boolean(config, "sample_at_inference", False):
            raise ValueError(
                "PVeRA sample_at_inference needs per-forward sampling, not a static merge"
            )
        return LinearAdapterOptions(rank=_rank(config, prefix, 256))
    if family == "RANDLORA":
        _require(factors, {"A", "B", "lambda", "gamma"}, family)
        rank = integer(config.get("r", 32), name="rank")
        return LinearAdapterOptions(
            rank=rank, scale=number(config.get("randlora_alpha", 640), name="randlora_alpha") / rank
        )
    if family == "UNILORA":
        if not {"indices_A", "indices_B", "scales_A", "scales_B"} <= factors.keys():
            raise ValueError(
                "UniLoRA requires saved indices and scales; re-export with save_indices=true"
            )
        _require(factors, {"indices_A", "indices_B", "scales_A", "scales_B", "theta_d"}, family)
        count = integer(config.get("theta_d_length", 256), name="theta_d_length")
        if factors["theta_d"].shape != (count,):
            raise ValueError("UniLoRA theta_d disagrees with theta_d_length")
        vector_bounds = tuple(
            index_bounds(factors[f"indices_{side}"], name="UniLoRA") for side in ("A", "B")
        )
        if any(high >= count for _, high in vector_bounds):
            raise ValueError("UniLoRA index is outside the shared theta_d vector")
        return LinearAdapterOptions(rank=_rank(config, prefix, 4), index_bounds=vector_bounds)
    count = integer(config.get("num_vectors", 256), name="num_vectors")
    width = integer(config.get("vector_length", 256), name="vector_length")
    topk = integer(config.get("topk", 2), name="topk")
    if factors["vector_bank"].shape != (count, width) or topk > count:
        raise ValueError("VBLoRA vector bank or topk disagrees with its configuration")
    compressed = boolean(config, "save_only_topk_weights", False)
    required = {f"logits_{side}" for side in ("A", "B")}
    if compressed:
        required = {
            f"logits_{side}_topk_{part}" for side in ("A", "B") for part in ("indices", "weights")
        }
    _require(factors, required | {"vector_bank"}, family)
    checks = []
    bounds = []
    for side in ("A", "B"):
        if compressed:
            indices = factors[f"logits_{side}_topk_indices"]
            weights = factors[f"logits_{side}_topk_weights"]
            if (
                indices.ndim != 3
                or indices.shape[-1] != topk
                or weights.shape != (*indices.shape[:-1], topk - 1)
            ):
                raise ValueError(
                    "VBLoRA compressed export needs topk indices and topk-1 probabilities"
                )
            bounds.append(index_bounds(indices, name="VBLoRA"))
            if bounds[-1][1] >= count:
                raise ValueError("VBLoRA index is outside its vector bank")
            ordered = mx.sort(indices.astype(mx.int32), axis=-1)
            checks.append(mx.all(ordered[..., 1:] != ordered[..., :-1]))
            checks.append(
                mx.all(weights >= 0) & mx.all(mx.sum(weights.astype(mx.float32), axis=-1) <= 1)
            )
        else:
            logits = factors[f"logits_{side}"]
            if logits.ndim != 3 or logits.shape[-1] != count:
                raise ValueError("VBLoRA logits must select over the complete vector bank")
            if topk < count:
                ordered = mx.sort(logits.astype(mx.float32), axis=-1)
                checks.append(mx.all(ordered[..., -topk] > ordered[..., -topk - 1]))
    if checks and not bool(mx.all(mx.stack(checks)).item()):
        raise ValueError(
            "VBLoRA has invalid compressed probabilities/indices or a tied topk cutoff; export unambiguous topk weights"
        )
    return LinearAdapterOptions(
        rank=_rank(config, prefix, 4),
        block_size=width,
        topk=topk,
        variant="compressed" if compressed else "logits",
        index_bounds=tuple(bounds),
    )


def parse_peft_adapter(
    weights: Mapping[str, mx.array],
    config: JsonObject,
    saved: Mapping[str, SavedParameter],
    *,
    family: str | None = None,
    validate_values: bool = True,
) -> AdapterInventory:
    family = str(config["peft_type"]) if family is None else family
    if config.get("fan_in_fan_out"):
        raise ValueError("PEFT fan_in_fan_out targets Conv1D storage, not linear source weights")
    groups: dict[str, dict[str, mx.array]] = {}
    names: dict[str, dict[str, str]] = {}
    named_adapters: set[str] = set()
    shared: dict[str, mx.array] = {}
    shared_names: dict[str, str] = {}
    unknown = []
    operations = []
    for key, value in sorted(weights.items()):
        global_factor = _shared_key(key, family)
        if global_factor is not None:
            role, adapter_name = global_factor
            if role in shared:
                raise ValueError(f"PEFT {family} repeats shared factor {role!r}")
            shared[role], shared_names[role] = value, key
            if adapter_name is not None:
                named_adapters.add(adapter_name)
            continue
        matched = _split(key, family)
        if matched is None:
            parameter = saved.get(key)
            if parameter is None:
                unknown.append(key)
                continue
            if value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(f"PEFT saved parameter {key!r} must be floating point")
            role = "delta" if parameter.part == "bias" else "set_weight"
            operations.append(
                AdapterOperation(
                    parameter.prefix, AdapterKind.SET, parameter.part, {role: value}, (key,)
                )
            )
            if parameter.adapter_name is not None:
                named_adapters.add(parameter.adapter_name)
            continue
        prefix, role, adapter_name = matched
        if adapter_name is not None:
            named_adapters.add(adapter_name)
        group = groups.setdefault(prefix, {})
        if role in group:
            raise ValueError(f"PEFT {family} repeats factor {role!r} at {prefix!r}")
        group[role] = value
        names.setdefault(prefix, {})[role] = key
    if len(named_adapters) > 1:
        raise ValueError(f"adapter file contains multiple named adapters: {sorted(named_adapters)}")
    allow_empty = {
        key
        for key in weights
        if family == "VBLORA" and config.get("topk") == 1 and key.endswith("_topk_weights")
    }
    if validate_values:
        validate_adapter_values(weights, allow_empty=allow_empty)
    if family == "TINYLORA":
        tying = number(config.get("weight_tying", 0.0), name="weight_tying")
        if not 0 <= tying <= 1:
            raise ValueError("TinyLoRA weight_tying must be between zero and one")
        if max(1, round(len(groups) * (1.0 - tying))) > 1:
            raise ValueError(
                "TinyLoRA configuration requires multiple groups but the export omits their module mapping; "
                "convert to LoRA in the producer environment"
            )
    if family in _SHARED:
        missing = set(_SHARED[family]) - shared.keys()
        if missing and not (
            family == "RANDLORA"
            and groups
            and all(set(_SHARED[family]) <= group.keys() for group in groups.values())
        ):
            hint = "save_indices=true" if family == "UNILORA" else "save_projection=true"
            raise ValueError(
                f"PEFT {family} is missing shared tensors {sorted(missing)}; re-export a self-contained adapter with {hint}"
            )
        for role, value in shared.items():
            if value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(f"PEFT {family} shared factor {role!r} must be floating point")
    for prefix, factors in groups.items():
        extra = [key for key in unknown if key.startswith(f"{prefix}.")]
        if extra:
            raise ValueError(f"PEFT {family} has unsupported semantic tensor {extra[0]!r}")
        for role, value in factors.items():
            is_indices = (
                role == "indices" or role.startswith("indices_") or role.endswith("_indices")
            )
            if not is_indices and value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(f"PEFT {family} factor {role!r} must be floating point")
        if family == "LN_TUNING":
            for role, value in factors.items():
                part: Literal["weight", "bias"] = "weight" if role == "weight" else "bias"
                factor = "set_weight" if part == "weight" else "delta"
                operations.append(
                    AdapterOperation(
                        prefix, AdapterKind.SET, part, {factor: value}, (names[prefix][role],)
                    )
                )
            continue
        for role in factors.keys() & shared.keys():
            if factors[role].shape != shared[role].shape or not bool(
                mx.all(factors[role] == shared[role]).item()
            ):
                raise ValueError(f"PEFT {family} local and shared {role} projections disagree")
        factors = {**shared, **factors}
        options = _options(family, prefix, factors, config)
        operation = AdapterOperation(
            prefix,
            AdapterKind(family.lower()),
            "bias" if family == "BEFT" else "weight",
            factors,
            tuple(sorted(names[prefix].values())),
            linear_options=options,
            dependency_keys=tuple(sorted(shared_names.values())),
        )
        operations.append(operation)
    if not operations:
        raise ValueError(f"PEFT {family} contains no complete supported operations")
    if shared and not groups:
        raise ValueError(
            f"PEFT {family} contains shared parameters but no saved per-target factors"
        )
    return AdapterInventory(tuple(operations), tuple(unknown), tuple(sorted(shared_names.values())))
