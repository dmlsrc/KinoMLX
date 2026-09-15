"""Krea 2 adapter exclusion categories."""

ADAPTER_EXCLUDE_CATEGORIES = frozenset(
    {
        "transformer",
        "text_fusion",
        "attn",
        "ff",
        "gate",
        "norm",
        "bias",
        "modulation",
        "input",
        "output",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_gate",
    }
)


def target_categories(target: str) -> frozenset[str]:
    """Return every user-facing exclusion category covering one target."""
    result = set()
    if target.startswith("transformer_blocks."):
        result.add("transformer")
    if target.startswith("text_fusion."):
        result.add("text_fusion")
    if ".attn." in target:
        result.add("attn")
    if ".ff." in target:
        result.add("ff")
    if any(token in target for token in (".norm", "norm1.", "norm2.")):
        result.add("norm")
    if "gate" in target:
        result.add("gate")
    if target.endswith(".bias"):
        result.add("bias")
    if "scale_shift_table" in target or target.startswith("time_mod_proj."):
        result.add("modulation")
    if target.startswith(("img_in.", "txt_in.", "time_embed.")):
        result.add("input")
    if target.startswith("final_layer."):
        result.add("output")
    for projection in ("to_q", "to_k", "to_v", "to_out", "to_gate"):
        if f".attn.{projection}." in target:
            result.add(projection)
    return frozenset(result)
