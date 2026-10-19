"""Enumerated Klein BFL/Diffusers adapter aliases and logical row transforms."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.formats import AdapterKind, AdapterOperation

from ..shapes import transformer_shapes


@dataclass(frozen=True)
class AdapterBinding:
    targets: tuple[str, ...]
    swap_halves: bool = False


class Flux2AdapterKeys:
    def __init__(self) -> None:
        self.aliases: dict[str, AdapterBinding] = {}
        for key in transformer_shapes():
            self._add(key.removesuffix(".weight"), AdapterBinding((key.removesuffix(".weight"),)))
        for source, target in (
            ("img_in", "x_embedder"),
            ("txt_in", "context_embedder"),
            ("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1"),
            ("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2"),
            ("final_layer.linear", "proj_out"),
            ("final_layer.adaLN_modulation.1", "norm_out.linear"),
            ("double_stream_modulation_img.lin", "double_stream_modulation_img.linear"),
            ("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear"),
            ("single_stream_modulation.lin", "single_stream_modulation.linear"),
        ):
            self._add(source, AdapterBinding((target,), source == "final_layer.adaLN_modulation.1"))
        for index in range(8):
            source, target = f"double_blocks.{index}", f"transformer_blocks.{index}"
            for stream, projections, out, ff in (
                ("img", ("to_q", "to_k", "to_v"), "to_out.0", "ff"),
                ("txt", ("add_q_proj", "add_k_proj", "add_v_proj"), "to_add_out", "ff_context"),
            ):
                self._add(
                    f"{source}.{stream}_attn.qkv",
                    AdapterBinding(tuple(f"{target}.attn.{name}" for name in projections)),
                )
                self._add(f"{source}.{stream}_attn.proj", AdapterBinding((f"{target}.attn.{out}",)))
                for number, name in ((0, "linear_in"), (2, "linear_out")):
                    self._add(
                        f"{source}.{stream}_mlp.{number}",
                        AdapterBinding((f"{target}.{ff}.{name}",)),
                    )
                for kind in ("query", "key"):
                    norm = ("norm_" if stream == "img" else "norm_added_") + kind[0]
                    self._add(
                        f"{source}.{stream}_attn.norm.{kind}_norm",
                        AdapterBinding((f"{target}.attn.{norm}",)),
                    )
        for index in range(24):
            for source, target in (
                ("linear1", "to_qkv_mlp_proj"),
                ("linear2", "to_out"),
                ("norm.query_norm", "norm_q"),
                ("norm.key_norm", "norm_k"),
            ):
                self._add(
                    f"single_blocks.{index}.{source}",
                    AdapterBinding((f"single_transformer_blocks.{index}.attn.{target}",)),
                )

    def _add(self, source: str, binding: AdapterBinding) -> None:
        for spelling in (source, source.replace(".", "_")):
            if spelling in self.aliases and self.aliases[spelling] != binding:
                raise ValueError(f"ambiguous Klein adapter alias: {spelling}")
            self.aliases[spelling] = binding

    def resolve(self, prefix: str) -> AdapterBinding | None:
        wrappers = (
            "base_model.model.",
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "_orig_mod.",
            "lora_unet_",
            "lora_transformer_",
            "lycoris_",
        )
        while match := next((name for name in wrappers if prefix.startswith(name)), None):
            prefix = prefix.removeprefix(match)
        if prefix.startswith("sformer_blocks."):
            prefix = "transformer_blocks." + prefix.removeprefix("sformer_blocks.")
        elif prefix.startswith("sformer_blocks_"):
            prefix = "transformer_blocks_" + prefix.removeprefix("sformer_blocks_")
        return self.aliases.get(prefix)


def place_operation(
    operation: AdapterOperation,
    binding: AdapterBinding,
    base_for: Callable[[str], mx.array],
) -> tuple[tuple[str, AdapterOperation], ...]:
    """Reuse shared format algebra; transform source output rows exactly once."""
    if len(binding.targets) == 1 and not binding.swap_halves:
        return ((f"{binding.targets[0]}.{operation.part}", operation),)
    plain = operation.kind == AdapterKind.LORA and not any(
        name in operation.factors for name in ("magnitude", "mid", "base_bias")
    )
    if not plain and operation.kind not in {AdapterKind.FULL, AdapterKind.NORM, AdapterKind.SET}:
        # Base-dependent formats must be evaluated in their declared source
        # orientation, including fused QKV and BFL final-AdaLN row ordering.
        def source_base(part: str) -> mx.array:
            value = mx.concatenate(
                [base_for(f"{target}.{part}") for target in binding.targets], axis=0
            )
            if binding.swap_halves:
                low, high = mx.split(value, 2, axis=0)
                value = mx.concatenate([high, low], axis=0)
            return value

        delta = reconstruct_delta(
            operation,
            source_base(operation.part),
            base_weight=source_base("weight")
            if operation.kind == AdapterKind.GLORA_PEFT and operation.part == "bias"
            else None,
        )
        operation = replace(
            operation,
            kind=AdapterKind.FULL,
            scalar=1.0,
            factors={"delta" if operation.part == "bias" else "diff": delta},
            linear_options=None,
        )
    role = (
        ("up_bias" if operation.part == "bias" else "up")
        if operation.kind == AdapterKind.LORA
        else "delta"
        if operation.part == "bias"
        else {"full": "diff", "norm": "w_norm", "set": "set_weight"}[operation.kind]
    )
    factors = dict(operation.factors)
    count = len(binding.targets)
    pieces = []
    for index, target in enumerate(binding.targets):
        placed = dict(factors)
        value = factors[role]
        if value.shape[0] % (count * (2 if binding.swap_halves else 1)):
            raise ValueError(f"Klein adapter output rows cannot be split: {operation.prefix}")
        if binding.swap_halves:
            low, high = mx.split(value, 2, axis=0)
            value = mx.concatenate([high, low], axis=0)
        if count > 1:
            value = mx.split(value, count, axis=0)[index]
        placed[role] = value
        pieces.append((f"{target}.{operation.part}", replace(operation, factors=placed)))
    return tuple(pieces)
