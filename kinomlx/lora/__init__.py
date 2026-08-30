"""Generic LoRA infrastructure.

Per-model LoRA configs (key mappings, stock LoRA paths/strengths,
model-specific prefix conventions) live under
``models/<name>/lora/``.  This package only defines the
model-agnostic math and the standard naming-convention parsers.
"""

from kinomlx.lora.fusion import compute_delta, fuse, fuse_many
from kinomlx.lora.loading import (
    LoRAConfig,
    LoRAEntry,
    LoRAProfile,
    find_lora_entry,
    format_lora_stage_scale_lines,
    iter_lora_entries,
    lora_configs_for_stage,
    lora_configs_have_stage_strengths,
)

__all__ = [
    "LoRAConfig",
    "LoRAEntry",
    "LoRAProfile",
    "compute_delta",
    "find_lora_entry",
    "format_lora_stage_scale_lines",
    "fuse",
    "fuse_many",
    "iter_lora_entries",
    "lora_configs_for_stage",
    "lora_configs_have_stage_strengths",
]
