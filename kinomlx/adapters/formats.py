"""Strict, model-neutral recognition of static safetensors adapter operations.

The parser preserves every tensor name. Aliases for a factor are alternatives,
not a priority list: two spellings of the same factor are an error. Model path
normalization and placement belong to the model adapter module.
"""

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

import mlx.core as mx


class AdapterKind(StrEnum):
    """Every adapter format this project can parse and fuse.

    Membership is decided here, not by an upstream release, because a
    format only becomes parseable once its math is implemented.
    """

    LORA = "lora"
    ADALORA = "adalora"
    LOHA = "loha"
    LOKR = "lokr"
    GLORA = "glora"
    GLORA_PEFT = "glora_peft"
    FOURIERFT = "fourierft"
    IA3 = "ia3"
    IA3_PEFT = "ia3_peft"
    OFT = "oft"
    BOFT = "boft"
    OFT_PEFT = "oft_peft"
    BOFT_PEFT = "boft_peft"
    HRA = "hra"
    FULL = "full"
    NORM = "norm"
    SET = "set"
    BEFT = "beft"
    HIRA = "hira"
    DELORA = "delora"
    C3A = "c3a"
    MISS = "miss"
    ROAD = "road"
    GRALORA = "gralora"
    PEANUT = "peanut"
    DEFT = "deft"
    SHIRA = "shira"
    SUPERTUNING = "supertuning"
    VERA = "vera"
    PVERA = "pvera"
    RANDLORA = "randlora"
    VBLORA = "vblora"
    UNILORA = "unilora"
    TINYLORA = "tinylora"
    BDLORA = "bdlora"
    QALORA = "qalora"
    WAVEFT = "waveft"


@dataclass(frozen=True)
class OFTOptions:
    """Inference choices that packed PEFT OFT tensors do not encode."""

    block_share: bool = False
    coft: bool = False
    eps: float = 6e-5
    use_neumann: bool = True
    neumann_terms: int = 5


@dataclass(frozen=True)
class GLoRAOptions:
    rank: int
    weight_path: str
    coupling_path: str
    bias_path: str


@dataclass(frozen=True)
class FourierOptions:
    frequencies: int
    seed: int


@dataclass(frozen=True)
class LinearAdapterOptions:
    """Validated inference scalars and index bounds; never derived factor arrays."""

    rank: int = 0
    block_size: int = 0
    variant: str = ""
    scale: float = 1.0
    hybrid_rank: int = 0
    depth: int = 0
    activation: str = ""
    projection_only: bool = False
    topk: int = 0
    index_bounds: tuple[tuple[int, int], ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class AdapterOperation:
    """One weight or bias operation, before model-specific target resolution."""

    prefix: str
    kind: AdapterKind
    part: Literal["weight", "bias"]
    factors: Mapping[str, mx.array]
    source_keys: tuple[str, ...]
    alpha: float | None = None
    scalar: float = 1.0
    magnitude_is_peft: bool = False
    input_scale: bool | None = None
    rank_denominator: float | None = None
    oft_options: OFTOptions | None = None
    hra_apply_gs: bool | None = None
    glora_options: GLoRAOptions | None = None
    fourier_options: FourierOptions | None = None
    linear_options: LinearAdapterOptions | None = None
    dependency_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterInventory:
    operations: tuple[AdapterOperation, ...]
    unknown_keys: tuple[str, ...]
    shared_keys: tuple[str, ...] = ()
    training_keys: tuple[str, ...] = ()

    @property
    def adapter_types(self) -> tuple[AdapterKind, ...]:
        """Return the canonical adapter types recognized by the shared parser."""
        return tuple(sorted({operation.kind for operation in self.operations}))


@dataclass(frozen=True)
class SavedParameter:
    """A configured full parameter value, not an additive adapter delta."""

    prefix: str
    part: Literal["weight", "bias"]
    adapter_name: str | None = None


_ALIASES = {
    "fourierft_spectrum": "spectrum",
    "oft_r": "oft_legacy",
    "oft_s": "oft_scale",
    "hra_u": "hra_vectors",
    "hra_u.weight": "hra_vectors",
    "oft_R.weight": "oft_packed",
    "oft_R": "oft_packed",
    "boft_R": "boft_rotation",
    "boft_s": "boft_scale",
    "lora_E": "diagonal",
    "lora_E.weight": "diagonal",
    "ia3_l": "ia3_vector",
    "ia3_l.weight": "ia3_vector",
    "lora_A.weight": "down",
    "lora_down.weight": "down",
    "lora_A": "down",
    "lora_down": "down",
    "lora_B.weight": "up",
    "lora_up.weight": "up",
    "lora_B": "up",
    "lora_up": "up",
    "lora_mid.weight": "mid",
    "lora_B.bias": "up_bias",
    "lora_up.bias": "up_bias",
    "lora_magnitude_vector": "magnitude",
    "lora_magnitude_vector.weight": "magnitude",
    "dora_scale": "magnitude",
    "a1.weight": "a1",
    "a2.weight": "a2",
    "b1.weight": "b1",
    "b2.weight": "b2",
    "bm.weight": "bm",
}
_EXACT = frozenset(
    {
        "alpha",
        "ranknum",
        "scalar",
        "diff",
        "diff_b",
        "w_norm",
        "b_norm",
        "set_weight",
        "weight",
        "on_input",
        "oft_blocks",
        "rescale",
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    }
)
_SUFFIXES = tuple(sorted((*_ALIASES, *_EXACT), key=len, reverse=True))
_NAMED_FACTOR = re.compile(
    r"^(.*)\.(lora_[ABE]|lora_magnitude_vector|ia3_l|ranknum|oft_R|oft_r|oft_s|boft_R|boft_s|hra_u|fourierft_spectrum|"
    r"hada_w[12]_[ab]|hada_t[12]|lokr_w[12](?:_[ab])?|lokr_t2)"
    r"\.([^.]+)(?:\.(weight|bias))?$"
)
_GLORA_FACTOR = re.compile(r"^(.*)\.glora_([ABCDE])\.(?:([^.]+)\.)?(Xd|Xu|X|zeros)$")


def split_factor_key(key: str) -> tuple[str, str, str | None] | None:
    glora = _GLORA_FACTOR.fullmatch(key)
    if glora is not None:
        prefix, label, adapter_name, parameter = glora.groups()
        return prefix, f"glora_{label}_{parameter}", adapter_name
    for suffix in _SUFFIXES:
        if suffix != "weight" and key.endswith(f".{suffix}"):
            return key[: -(len(suffix) + 1)], _ALIASES.get(suffix, suffix), None
    named = _NAMED_FACTOR.fullmatch(key)
    if named is None:
        if key.endswith(".weight"):
            return key.removesuffix(".weight"), "weight", None
        return None
    prefix, factor, adapter_name, parameter = named.groups()
    suffix = factor if parameter is None else f"{factor}.{parameter}"
    role = _ALIASES.get(suffix, suffix if suffix in _EXACT else None)
    if role is None:
        return None
    return prefix, role, adapter_name


def _scalar(value: mx.array, name: str) -> float:
    if value.size != 1:
        raise ValueError(f"adapter {name!r} must be a scalar, got {value.shape}")
    result = float(cast(float, value.item()))
    if not math.isfinite(result):
        raise ValueError(f"adapter {name!r} must be finite")
    return result


def validate_adapter_values(
    weights: Mapping[str, mx.array], *, allow_empty: Collection[str] = ()
) -> None:
    """Check all input values, including unplaced and excluded tensors, in batches."""
    pending: list[tuple[str, mx.array]] = []
    for name, value in weights.items():
        if not value.size:
            if name not in allow_empty:
                raise ValueError(f"adapter tensor {name!r} is empty")
            continue
        if value.dtype not in (
            mx.float32,
            mx.float16,
            mx.bfloat16,
            mx.int8,
            mx.int16,
            mx.int32,
            mx.int64,
            mx.uint8,
            mx.uint16,
            mx.uint32,
            mx.uint64,
            mx.bool_,
        ):
            raise ValueError(f"adapter tensor {name!r} has unsupported dtype {value.dtype}")
        pending.append((name, mx.all(mx.isfinite(value.astype(mx.float32)))))
        if len(pending) == 64:
            _check_finite(pending)
            pending.clear()
    _check_finite(pending)


def _check_finite(checks: list[tuple[str, mx.array]]) -> None:
    if not checks:
        return
    mx.eval([check for _, check in checks])
    for name, check in checks:
        if not bool(check.item()):
            raise ValueError(f"adapter tensor {name!r} contains nonfinite values")


def _validate_structure(operation: AdapterOperation) -> None:
    factors = operation.factors
    for role, value in factors.items():
        if role != "on_input" and value.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError(f"adapter {operation.prefix!r} factor {role!r} must be floating point")
    pairs: list[tuple[str, str, str | None]] = []
    if operation.kind in (AdapterKind.LORA, AdapterKind.ADALORA):
        if operation.part == "bias":
            if factors["down"].ndim < 2 or factors["up_bias"].ndim != 1:
                raise ValueError("LoRA B bias must be a vector with a valid down factor")
        else:
            pairs = [("up", "down", "mid")]
        if operation.kind == AdapterKind.ADALORA:
            if factors["down"].ndim != 2 or factors["up"].ndim != 2:
                raise ValueError("AdaLoRA requires two-dimensional factors")
            rank = factors["down"].shape[0]
            if factors["diagonal"].shape != (rank, 1):
                raise ValueError("AdaLoRA singular values must have shape (rank, 1)")
    elif operation.kind == AdapterKind.LOHA:
        pairs = [(f"hada_w{i}_a", f"hada_w{i}_b", f"hada_t{i}") for i in (1, 2)]
    elif operation.kind == AdapterKind.LOKR:
        pairs = [
            (f"lokr_w{i}_a", f"lokr_w{i}_b", "lokr_t2" if i == 2 else None)
            for i in (1, 2)
            if f"lokr_w{i}" not in factors
        ]
    elif operation.kind == AdapterKind.GLORA:
        pairs = [("a1", "a2", None), ("b1", "b2", "bm")]
    for up_key, down_key, mid_key in pairs:
        up, down = factors[up_key], factors[down_key]
        if up.ndim < 2 or down.ndim < 2:
            raise ValueError(f"adapter {operation.prefix!r} factors must have at least two axes")
        mid = None if mid_key is None else factors.get(mid_key)
        transposed_up = mid is not None and operation.kind in (AdapterKind.LOHA, AdapterKind.LOKR)
        up_rank = up.shape[0 if transposed_up else 1]
        if mid is None:
            if up_rank != down.shape[0]:
                raise ValueError(f"adapter {operation.prefix!r} factor ranks do not match")
        elif mid.ndim < 2 or mid.shape[:2] != (up_rank, down.shape[0]):
            raise ValueError(f"adapter {operation.prefix!r} has inconsistent Tucker factors")


def _operation(
    prefix: str,
    factors: dict[str, mx.array],
    keys: dict[str, str],
) -> list[AdapterOperation]:
    remaining = dict(factors)
    alpha_value = remaining.pop("alpha", None)
    alpha = None if alpha_value is None else _scalar(alpha_value, keys["alpha"])
    scalar_value = remaining.pop("scalar", None)
    scalar = 1.0 if scalar_value is None else _scalar(scalar_value, keys["scalar"])
    rank_value = remaining.pop("ranknum", None)
    rank_denominator = None if rank_value is None else _scalar(rank_value, keys["ranknum"])
    if rank_denominator is not None and rank_denominator < 0:
        raise ValueError("AdaLoRA ranknum must be nonnegative")
    biases: list[AdapterOperation] = []
    bias_roles: tuple[tuple[str, AdapterKind], ...] = (
        ("diff_b", AdapterKind.FULL),
        ("b_norm", AdapterKind.NORM),
    )
    for role, bias_kind in bias_roles:
        if role in remaining:
            bias = remaining.pop(role)
            biases.append(
                AdapterOperation(prefix, bias_kind, "bias", {"delta": bias}, (keys[role],))
            )
    if len(biases) > 1:
        raise ValueError(f"adapter {prefix!r} defines more than one bias update")
    if not remaining:
        if alpha_value is not None or scalar_value is not None:
            raise ValueError(f"adapter {prefix!r} has scaling without a weight operation")
        return biases

    signatures: tuple[tuple[AdapterKind, frozenset[str]], ...] = (
        (AdapterKind.LORA, frozenset({"down", "up", "mid", "up_bias", "diagonal"})),
        (
            AdapterKind.LOHA,
            frozenset({"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2"}),
        ),
        (
            AdapterKind.LOKR,
            frozenset(
                {
                    "lokr_w1",
                    "lokr_w1_a",
                    "lokr_w1_b",
                    "lokr_w2",
                    "lokr_w2_a",
                    "lokr_w2_b",
                    "lokr_t2",
                }
            ),
        ),
        (AdapterKind.GLORA, frozenset({"a1", "a2", "b1", "b2", "bm"})),
        (AdapterKind.IA3, frozenset({"on_input"})),
        (AdapterKind.IA3_PEFT, frozenset({"ia3_vector"})),
        (AdapterKind.OFT, frozenset({"oft_blocks"})),
        (AdapterKind.OFT_PEFT, frozenset({"oft_packed", "oft_legacy", "oft_scale"})),
        (AdapterKind.BOFT_PEFT, frozenset({"boft_rotation", "boft_scale"})),
        (AdapterKind.HRA, frozenset({"hra_vectors"})),
        (AdapterKind.FOURIERFT, frozenset({"spectrum"})),
        (
            AdapterKind.GLORA_PEFT,
            frozenset(
                f"glora_{label}_{part}" for label in "ABCDE" for part in ("Xd", "Xu", "X", "zeros")
            ),
        ),
        (AdapterKind.FULL, frozenset({"diff"})),
        (AdapterKind.NORM, frozenset({"w_norm"})),
        (AdapterKind.SET, frozenset({"set_weight"})),
    )
    found = [kind for kind, roles in signatures if roles & remaining.keys()]
    if len(found) != 1:
        raise ValueError(
            f"adapter {prefix!r} has ambiguous or unsupported factors: {sorted(remaining)}"
        )
    kind = found[0]
    if kind == AdapterKind.LORA and "diagonal" in remaining:
        kind = AdapterKind.ADALORA
    if rank_value is not None and kind != AdapterKind.ADALORA:
        raise ValueError("ranknum is only defined for AdaLoRA")
    required: set[str]
    optional: set[str] = set()
    if kind == AdapterKind.LORA:
        required = {"down", "up"}
        optional = {"mid", "up_bias", "magnitude"}
    elif kind == AdapterKind.ADALORA:
        required = {"down", "up", "diagonal"}
    elif kind == AdapterKind.LOHA:
        required = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
        optional = {"hada_t1", "hada_t2", "magnitude"}
        if ("hada_t1" in remaining) != ("hada_t2" in remaining):
            raise ValueError(f"adapter {prefix!r} needs both LoHa Tucker factors")
    elif kind == AdapterKind.LOKR:
        required = set()
        optional = {"lokr_t2", "magnitude"}
        for side in ("lokr_w1", "lokr_w2"):
            if side in remaining:
                required.add(side)
            else:
                required.update({f"{side}_a", f"{side}_b"})
        if "lokr_t2" in remaining and "lokr_w2" in remaining:
            raise ValueError(f"adapter {prefix!r} mixes dense LoKr W2 and Tucker factors")
    elif kind == AdapterKind.GLORA:
        required = {"a1", "a2", "b1", "b2"}
        optional = {"bm"}
    elif kind == AdapterKind.IA3:
        required = {"weight", "on_input"}
        flag = _scalar(remaining["on_input"], keys["on_input"])
        if flag not in (0.0, 1.0):
            raise ValueError(f"adapter {prefix!r} IA3 on_input must be 0 or 1")
    elif kind == AdapterKind.IA3_PEFT:
        required = {"ia3_vector"}
    elif kind == AdapterKind.OFT:
        required = {"oft_blocks"}
        optional = {"rescale"}
        if remaining["oft_blocks"].ndim == 4:
            kind = AdapterKind.BOFT
    elif kind == AdapterKind.OFT_PEFT:
        required = {"oft_legacy", "oft_scale"} if "oft_legacy" in remaining else {"oft_packed"}
    elif kind == AdapterKind.BOFT_PEFT:
        required = {"boft_rotation", "boft_scale"}
    elif kind == AdapterKind.HRA:
        required = {"hra_vectors"}
    elif kind == AdapterKind.FOURIERFT:
        required = {"spectrum"}
    elif kind == AdapterKind.GLORA_PEFT:
        required = set()
        for label in "ABCDE":
            roles = {role for role in remaining if role.startswith(f"glora_{label}_")}
            valid = [{f"glora_{label}_{part}"} for part in ("X", "zeros")]
            valid.append({f"glora_{label}_Xd", f"glora_{label}_Xu"})
            if roles not in valid:
                raise ValueError(f"PEFT GLoRA path {label} has missing or overlapping factors")
            required.update(roles)
    else:
        role = {"full": "diff", "norm": "w_norm", "set": "set_weight"}[kind]
        required = {role}
    missing = required - remaining.keys()
    extra = remaining.keys() - required - optional
    if missing or extra:
        raise ValueError(
            f"adapter {prefix!r}: missing factors {sorted(missing)}, extra factors {sorted(extra)}"
        )
    if (
        kind
        in (
            "full",
            "norm",
            "set",
            "ia3",
            "ia3_peft",
            "oft_peft",
            "boft_peft",
            "hra",
            "glora_peft",
            "fourierft",
        )
        and alpha_value is not None
    ):
        raise ValueError(f"adapter {prefix!r}: alpha has no defined meaning for {kind}")

    consumed = set(factors) - {"diff_b", "b_norm", "up_bias"}
    if "up_bias" in remaining:
        if biases:
            raise ValueError(f"adapter {prefix!r} has both LoRA B bias and a bias delta")
        if "magnitude" in remaining:
            raise ValueError(
                f"adapter {prefix!r}: PEFT does not define DoRA with a LoRA B bias; export an explicit bias delta"
            )
        biases.append(
            AdapterOperation(
                prefix,
                AdapterKind.LORA,
                "bias",
                {"up_bias": remaining.pop("up_bias"), "down": remaining["down"]},
                (keys["up_bias"],),
                alpha,
                scalar,
            )
        )
    return [
        AdapterOperation(
            prefix,
            kind,
            "weight",
            remaining,
            tuple(sorted(keys[role] for role in consumed)),
            alpha,
            scalar,
            "lora_magnitude_vector" in keys.get("magnitude", ""),
            rank_denominator=rank_denominator,
        ),
        *biases,
    ]


def parse_adapter(
    weights: Mapping[str, mx.array],
    *,
    saved_parameters: Mapping[str, SavedParameter] | None = None,
    validate_values: bool = True,
) -> AdapterInventory:
    """Recognize algorithms and reject incomplete or overlapping signatures."""
    groups: dict[str, dict[str, mx.array]] = {}
    names: dict[str, dict[str, str]] = {}
    named_adapters: set[str] = set()
    unknown = []
    operations = []
    for key, value in sorted(weights.items()):
        saved = None if saved_parameters is None else saved_parameters.get(key)
        if saved is not None:
            if saved.adapter_name is not None:
                named_adapters.add(saved.adapter_name)
            role = "delta" if saved.part == "bias" else "set_weight"
            operations.append(
                AdapterOperation(saved.prefix, AdapterKind.SET, saved.part, {role: value}, (key,))
            )
            continue
        matched = split_factor_key(key)
        if matched is None:
            unknown.append(key)
            continue
        prefix, role, adapter_name = matched
        if adapter_name is not None:
            named_adapters.add(adapter_name)
        group = groups.setdefault(prefix, {})
        if role in group:
            raise ValueError(
                f"adapter {prefix!r} repeats factor {role!r}: {names[prefix][role]!r} and {key!r}"
            )
        group[role] = value
        names.setdefault(prefix, {})[role] = key
    if len(named_adapters) > 1:
        raise ValueError(f"adapter file contains multiple named adapters: {sorted(named_adapters)}")
    for prefix, factors in groups.items():
        # A plain weight without an IA3 flag may be unrelated checkpoint baggage.
        if factors.keys() == {"weight"}:
            unknown.append(names[prefix]["weight"])
            continue
        extra = [key for key in unknown if key.startswith(f"{prefix}.")]
        if extra:
            raise ValueError(f"adapter {prefix!r} has an unsupported semantic tensor: {extra[0]!r}")
        operations.extend(_operation(prefix, factors, names[prefix]))
    if not operations:
        first = next(iter(weights), "<empty file>")
        raise ValueError(
            f"no complete supported static adapter operations (first tensor: {first!r})"
        )
    for operation in operations:
        _validate_structure(operation)
    inventory = AdapterInventory(tuple(operations), tuple(sorted(unknown)))
    if validate_values:
        validate_inventory_values(weights, inventory)
    return inventory


def validate_inventory_values(
    weights: Mapping[str, mx.array],
    inventory: AdapterInventory,
    *,
    allow_empty: Collection[str] = (),
) -> None:
    """Check tensor values using an already parsed inventory."""
    empty = set(allow_empty) | {
        key
        for operation in inventory.operations
        if operation.kind == AdapterKind.ADALORA and operation.factors["down"].shape[0] == 0
        for key in operation.source_keys
    }
    validate_adapter_values(weights, allow_empty=empty)
    disabled = [
        mx.all(value == 0)
        for operation in inventory.operations
        if operation.kind == AdapterKind.GLORA_PEFT
        for role, value in operation.factors.items()
        if role.endswith("_zeros")
    ]
    if disabled and not bool(mx.all(mx.stack(disabled)).item()):
        raise ValueError("disabled PEFT GLoRA paths must contain only zeros")
