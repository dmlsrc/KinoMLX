"""Klein adapter selection vocabulary across logical projection targets."""

ADAPTER_EXCLUDE_CATEGORIES = frozenset(
    {
        "transformer",
        "double",
        "single",
        "attn",
        "ff",
        "modulation",
        "norm",
        "bias",
        "input",
        "output",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
    }
)


def target_categories(target: str) -> frozenset[str]:
    categories = {"transformer"}
    if target.startswith("transformer_blocks."):
        categories.add("double")
    if target.startswith("single_transformer_blocks."):
        categories.add("single")
    if ".attn." in target:
        categories.add("attn")
    if ".to_qkv_mlp_proj." in target:
        # Exclusions operate on stored projection modules, not individual rows.
        categories.update(("to_q", "to_k", "to_v"))
    if (
        ".ff" in target
        or ".to_qkv_mlp_proj." in target
        or (target.startswith("single_transformer_blocks.") and ".to_out." in target)
    ):
        categories.add("ff")
    if "modulation" in target or target.startswith("norm_out.linear."):
        categories.add("modulation")
    if ".norm_" in target:
        categories.add("norm")
    if target.endswith(".bias"):
        categories.add("bias")
    if target.startswith(("x_embedder.", "context_embedder.", "time_guidance_embed.")):
        categories.add("input")
    if target.startswith(("proj_out.", "norm_out.")):
        categories.add("output")
    for name, extra in (
        ("to_q", "add_q_proj"),
        ("to_k", "add_k_proj"),
        ("to_v", "add_v_proj"),
        ("to_out", "to_add_out"),
    ):
        if f".{name}." in target or f".{extra}." in target:
            categories.add(name)
    return frozenset(categories)
