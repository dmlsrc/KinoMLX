"""Krea path facts; static adapter algebra is shared outside the model package."""

from dataclasses import replace

from kinomlx.adapters.formats import AdapterOperation

from ..metadata import Krea2TransformerConfig
from ..weight_keys import KeyBinding, transformer_source_bindings

_WRAPPERS = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "_orig_mod.",
)
_FLAT_WRAPPERS = ("lora_unet_", "lora_transformer_", "lycoris_")


class Krea2AdapterKeys:
    """Resolve dotted and flattened aliases by enumeration, never underscore guessing."""

    def __init__(self, config: Krea2TransformerConfig) -> None:
        self._aliases: dict[tuple[str, str], KeyBinding] = {}
        for source, binding in transformer_source_bindings(config).items():
            part = "bias" if binding.target_key.endswith(".bias") else "weight"
            module = source
            for suffix in (".weight", ".bias", ".scale"):
                module = module.removesuffix(suffix)
            for alias in (module, source):
                for spelling in (alias, alias.replace(".", "_")):
                    key = (spelling, part)
                    previous = self._aliases.get(key)
                    if previous is not None and previous.target_key != binding.target_key:
                        raise ValueError(f"Krea adapter graph alias collision: {spelling!r}")
                    self._aliases[key] = binding

    @staticmethod
    def _unwrap(prefix: str) -> str:
        while True:
            match = next((wrapper for wrapper in _WRAPPERS if prefix.startswith(wrapper)), None)
            if match is None:
                break
            prefix = prefix.removeprefix(match)
        for wrapper in _FLAT_WRAPPERS:
            if prefix.startswith(wrapper):
                return prefix.removeprefix(wrapper)
        return prefix

    def resolve(self, prefix: str, part: str) -> KeyBinding | None:
        return self._aliases.get((self._unwrap(prefix), part))

    def belongs_to_known_module(self, key: str) -> bool:
        unwrapped = self._unwrap(key)
        return any(unwrapped.startswith(f"{alias}.") for alias, _part in self._aliases)


def canonicalize_adapter_operation(
    operation: AdapterOperation, binding: KeyBinding
) -> AdapterOperation:
    """Apply Krea source-layout facts before shared adapter algebra."""
    if binding.reshape_rows is None or operation.part != "weight":
        return operation
    role = {"full": "diff", "norm": "w_norm", "set": "set_weight"}.get(operation.kind)
    if role is None:
        return operation
    value = operation.factors[role]
    if value.ndim == 2 and value.shape[0] == binding.reshape_rows:
        return operation
    if value.ndim != 1 or value.size % binding.reshape_rows:
        raise ValueError(
            f"Krea adapter {operation.prefix!r} cannot reshape {value.shape} into "
            f"{binding.reshape_rows} modulation rows"
        )
    return replace(
        operation,
        factors={
            **operation.factors,
            role: value.reshape(binding.reshape_rows, value.size // binding.reshape_rows),
        },
    )
