"""Complete structural equations for the supported bundled Klein 9B components."""


def text_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (151936, 4096),
        "lm_head.weight": (151936, 4096),
        "model.norm.weight": (4096,),
    }
    for i in range(36):
        prefix = f"model.layers.{i}."
        block = {
            "self_attn.q_proj.weight": (4096, 4096),
            "self_attn.k_proj.weight": (1024, 4096),
            "self_attn.v_proj.weight": (1024, 4096),
            "self_attn.o_proj.weight": (4096, 4096),
            "self_attn.q_norm.weight": (128,),
            "self_attn.k_norm.weight": (128,),
            "mlp.gate_proj.weight": (12288, 4096),
            "mlp.up_proj.weight": (12288, 4096),
            "mlp.down_proj.weight": (4096, 12288),
            "input_layernorm.weight": (4096,),
            "post_attention_layernorm.weight": (4096,),
        }
        shapes.update({prefix + k: v for k, v in block.items()})
    return shapes


def transformer_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "x_embedder.weight": (4096, 128),
        "context_embedder.weight": (4096, 12288),
        "double_stream_modulation_img.linear.weight": (24576, 4096),
        "double_stream_modulation_txt.linear.weight": (24576, 4096),
        "single_stream_modulation.linear.weight": (12288, 4096),
        "time_guidance_embed.timestep_embedder.linear_1.weight": (4096, 256),
        "time_guidance_embed.timestep_embedder.linear_2.weight": (4096, 4096),
        "norm_out.linear.weight": (8192, 4096),
        "proj_out.weight": (128, 4096),
    }
    for i in range(8):
        prefix = f"transformer_blocks.{i}."
        for key in (
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ):
            shapes[prefix + f"attn.{key}.weight"] = (4096, 4096)
        for key in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            shapes[prefix + f"attn.{key}.weight"] = (128,)
        for key in ("ff", "ff_context"):
            shapes[prefix + key + ".linear_in.weight"] = (24576, 4096)
            shapes[prefix + key + ".linear_out.weight"] = (4096, 12288)
    for i in range(24):
        prefix = f"single_transformer_blocks.{i}.attn."
        shapes[prefix + "to_qkv_mlp_proj.weight"] = (36864, 4096)
        shapes[prefix + "to_out.weight"] = (4096, 16384)
        shapes[prefix + "norm_q.weight"] = (128,)
        shapes[prefix + "norm_k.weight"] = (128,)
    return shapes


def vae_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "bn.running_mean": (128,),
        "bn.running_var": (128,),
        "bn.num_batches_tracked": (),
    }

    def pair(name: str, shape: tuple[int, ...]) -> None:
        shapes[name + ".weight"] = shape
        shapes[name + ".bias"] = (shape[0],)

    def resnet(name: str, inp: int, out: int) -> None:
        pair(name + ".norm1", (inp,))
        pair(name + ".conv1", (out, inp, 3, 3))
        pair(name + ".norm2", (out,))
        pair(name + ".conv2", (out, out, 3, 3))
        if inp != out:
            pair(name + ".conv_shortcut", (out, inp, 1, 1))

    for component in ("encoder", "decoder"):
        encoder = component == "encoder"
        channels = [128, 256, 512, 512] if encoder else [512, 512, 256, 128]
        current = channels[0]
        pair(component + ".conv_in", (current, 3 if encoder else 32, 3, 3))
        for i, out in enumerate(channels):
            prefix = f"{component}.{'down' if encoder else 'up'}_blocks.{i}"
            for j in range(2 if encoder else 3):
                resnet(f"{prefix}.resnets.{j}", current, out)
                current = out
            if i < 3:
                pair(prefix + f".{'down' if encoder else 'up'}samplers.0.conv", (out, out, 3, 3))
        for j in range(2):
            resnet(f"{component}.mid_block.resnets.{j}", 512, 512)
        prefix = component + ".mid_block.attentions.0"
        pair(prefix + ".group_norm", (512,))
        for key in ("to_q", "to_k", "to_v", "to_out.0"):
            pair(prefix + "." + key, (512, 512))
        pair(component + ".conv_norm_out", (current,))
        pair(component + ".conv_out", (64 if encoder else 3, current, 3, 3))
    pair("quant_conv", (64, 64, 1, 1))
    pair("post_quant_conv", (32, 32, 1, 1))
    return shapes
