"""Small structurally complete Krea 2 source fixtures."""

import json
from pathlib import Path

import mlx.core as mx

from kinomlx.models.krea2.metadata import Krea2VAEConfig
from kinomlx.models.krea2.weight_keys import expected_vae_keys
from kinomlx.tokenizers.unicode_scan import QWEN_SPLIT_PATTERN

MAIN_BLOCK_TAILS = (
    "attn.norm_k.weight",
    "attn.norm_q.weight",
    "attn.to_gate.weight",
    "attn.to_k.weight",
    "attn.to_out.0.weight",
    "attn.to_q.weight",
    "attn.to_v.weight",
    "ff.down.weight",
    "ff.gate.weight",
    "ff.up.weight",
    "norm1.weight",
    "norm2.weight",
    "scale_shift_table",
)
TEXT_BLOCK_TAILS = MAIN_BLOCK_TAILS[:-1]
BASIC_TRANSFORMER_KEYS = (
    "final_layer.linear.bias",
    "final_layer.linear.weight",
    "final_layer.norm.weight",
    "final_layer.scale_shift_table",
    "img_in.bias",
    "img_in.weight",
    "text_fusion.projector.weight",
    "time_embed.linear_1.bias",
    "time_embed.linear_1.weight",
    "time_embed.linear_2.bias",
    "time_embed.linear_2.weight",
    "time_mod_proj.bias",
    "time_mod_proj.weight",
    "txt_in.linear_1.bias",
    "txt_in.linear_1.weight",
    "txt_in.linear_2.bias",
    "txt_in.linear_2.weight",
    "txt_in.norm.weight",
)
TEXT_LAYER_TAILS = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
_NATIVE_BLOCK_TAILS = {
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.to_gate.weight": "attn.gate.weight",
    "attn.to_k.weight": "attn.wk.weight",
    "attn.to_out.0.weight": "attn.wo.weight",
    "attn.to_q.weight": "attn.wq.weight",
    "attn.to_v.weight": "attn.wv.weight",
    "ff.down.weight": "mlp.down.weight",
    "ff.gate.weight": "mlp.gate.weight",
    "ff.up.weight": "mlp.up.weight",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
    "scale_shift_table": "mod.lin",
}
_NATIVE_BASIC_KEYS = {
    "final_layer.linear.bias": "last.linear.bias",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.scale_shift_table": "last.modulation.lin",
    "img_in.bias": "first.bias",
    "img_in.weight": "first.weight",
    "text_fusion.projector.weight": "txtfusion.projector.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.norm.weight": "txtmlp.0.scale",
}


def transformer_keys() -> tuple[str, ...]:
    keys = [
        f"transformer_blocks.{block}.{tail}" for block in range(28) for tail in MAIN_BLOCK_TAILS
    ]
    keys.extend(
        f"text_fusion.{family}.{block}.{tail}"
        for family in ("layerwise_blocks", "refiner_blocks")
        for block in range(2)
        for tail in TEXT_BLOCK_TAILS
    )
    keys.extend(BASIC_TRANSFORMER_KEYS)
    assert len(keys) == 430
    return tuple(keys)


def text_encoder_keys() -> tuple[str, ...]:
    keys = ["language_model.embed_tokens.weight"]
    keys.extend(
        f"language_model.layers.{layer}.{tail}" for layer in range(36) for tail in TEXT_LAYER_TAILS
    )
    keys.append("language_model.norm.weight")
    keys.extend(("visual.patch_embed.proj.weight", "visual.pos_embed.weight"))
    return tuple(keys)


def vae_keys() -> tuple[str, ...]:
    config = Krea2VAEConfig(
        base_dim=96,
        dim_mult=(1, 2, 4, 4),
        input_channels=3,
        z_dim=16,
        num_res_blocks=2,
        attn_scales=(),
        latent_mean=(0.0,) * 16,
        latent_std=(1.0,) * 16,
    )
    return tuple(sorted(expected_vae_keys(config)))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_weights(
    path: Path,
    keys: tuple[str, ...],
    marker: float,
    *,
    start_index: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for index, key in enumerate(keys):
        value = marker + (start_index + index) / 10_000
        array = mx.array([value], dtype=mx.bfloat16)
        if key.startswith("transformer_blocks.") and key.endswith(".scale_shift_table"):
            array = mx.broadcast_to(array, (6, 1))
        elif key.startswith("transformer_blocks.") and key.endswith(".ff.down.weight"):
            array = (mx.arange(6).reshape(2, 3) / 100 + value).astype(mx.bfloat16)
        weights[key] = array
    mx.save_safetensors(
        str(path),
        weights,
    )


def _native_transformer_key(canonical: str) -> str:
    if canonical in _NATIVE_BASIC_KEYS:
        return _NATIVE_BASIC_KEYS[canonical]
    if canonical.startswith("transformer_blocks."):
        _prefix, block, tail = canonical.split(".", 2)
        return f"blocks.{block}.{_NATIVE_BLOCK_TAILS[tail]}"
    if canonical.startswith("text_fusion."):
        _prefix, family, block, tail = canonical.split(".", 3)
        return f"txtfusion.{family}.{block}.{_NATIVE_BLOCK_TAILS[tail]}"
    raise AssertionError(f"fixture has no native Krea 2 key for {canonical}")


def _write_native_transformer(path: Path, marker: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for index, canonical in enumerate(transformer_keys()):
        source = _native_transformer_key(canonical)
        value = marker + index / 10_000
        array = mx.array([value], dtype=mx.bfloat16)
        if source.endswith(".mod.lin"):
            array = mx.broadcast_to(array, (6,))
        elif canonical.startswith("transformer_blocks.") and canonical.endswith(".ff.down.weight"):
            array = (mx.arange(6).reshape(2, 3) / 100 + value).astype(mx.bfloat16)
        weights[source] = array
    mx.save_safetensors(str(path), weights)


def _write_vae_weights(path: Path, marker: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for index, key in enumerate(vae_keys()):
        value = marker + index / 10_000
        if key == "decoder.conv_out.weight":
            shape = (3, 1, 3, 1, 1)
        elif key == "decoder.conv_out.bias":
            shape = (3,)
        elif key.endswith(".gamma"):
            shape = (1, 1, 1) if ".attentions." in key else (1, 1, 1, 1)
        elif key.endswith(".weight") and (
            ".attentions." in key or ".upsamplers.0.resample.1." in key
        ):
            shape = (1, 1, 1, 1)
        elif key.endswith(".weight"):
            shape = (1, 1, 3, 1, 1)
        else:
            shape = (1,)
        weights[key] = mx.full(shape, value, dtype=mx.bfloat16)
    weights["encoder.conv_in.weight"] = mx.array([marker + 1.0], dtype=mx.bfloat16)
    weights["quant_conv.weight"] = mx.array([marker + 2.0], dtype=mx.bfloat16)
    mx.save_safetensors(str(path), weights)


def make_snapshot(
    root: Path,
    *,
    variant: str = "turbo",
    transformer_marker: float = 1.0,
) -> Path:
    """Create both supported transformer layouts plus shared component files."""
    distilled = variant == "turbo"
    _write_json(
        root / "model_index.json",
        {
            "_class_name": "Krea2Pipeline",
            "is_distilled": distilled,
            "patch_size": 2,
            "text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        },
    )
    _write_json(
        root / "scheduler" / "scheduler_config.json",
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": False,
            "max_image_seq_len": 6400,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        },
    )
    _write_json(
        root / "transformer" / "config.json",
        {
            "_class_name": "Krea2Transformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [32, 48, 48],
            "in_channels": 64,
            "intermediate_size": 16384,
            "norm_eps": 1e-5,
            "num_attention_heads": 48,
            "num_key_value_heads": 12,
            "num_layers": 28,
            "num_layerwise_text_blocks": 2,
            "num_refiner_text_blocks": 2,
            "num_text_layers": 12,
            "rope_theta": 1000.0,
            "text_hidden_dim": 2560,
            "text_intermediate_size": 6912,
            "text_num_attention_heads": 20,
            "text_num_key_value_heads": 20,
            "timestep_embed_dim": 256,
        },
    )
    _write_json(
        root / "text_encoder" / "config.json",
        {
            "architectures": ["Qwen3VLModel"],
            "model_type": "qwen3_vl",
            "text_config": {
                "attention_bias": False,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_theta": 5_000_000,
                    "rope_type": "default",
                },
                "vocab_size": 151936,
            },
            "vision_config": {"depth": 24},
        },
    )
    _write_json(
        root / "vae" / "config.json",
        {
            "_class_name": "AutoencoderKLQwenImage",
            "attn_scales": [],
            "base_dim": 96,
            "dim_mult": [1, 2, 4, 4],
            "dropout": 0.0,
            "input_channels": 3,
            "latents_mean": [0.0] * 16,
            "latents_std": [1.0] * 16,
            "num_res_blocks": 2,
            "temperal_downsample": [False, True, True],
            "z_dim": 16,
        },
    )
    _write_json(
        root / "tokenizer" / "tokenizer.json",
        {
            "version": "1.0",
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN_SPLIT_PATTERN},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": True,
                        "use_regex": False,
                    },
                ],
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": False,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"],
            },
            "added_tokens": [
                {
                    "id": 151643,
                    "content": "<|endoftext|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 151644,
                    "content": "<|im_start|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 151645,
                    "content": "<|im_end|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            ],
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
        },
    )
    _write_json(
        root / "tokenizer" / "tokenizer_config.json",
        {
            "bos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "clean_up_tokenization_spaces": False,
        },
    )
    (root / "tokenizer" / "chat_template.jinja").write_text(
        "unused generic template\n",
        encoding="utf-8",
    )

    text_keys = text_encoder_keys()
    _write_weights(root / "text_encoder" / "model.safetensors", text_keys, 3.0)
    _write_vae_weights(
        root / "vae" / "diffusion_pytorch_model.safetensors",
        4.0,
    )

    keys = transformer_keys()
    midpoint = len(keys) // 2
    shard_names = (
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    )
    _write_weights(root / "transformer" / shard_names[0], keys[:midpoint], transformer_marker)
    _write_weights(
        root / "transformer" / shard_names[1],
        keys[midpoint:],
        transformer_marker,
        start_index=midpoint,
    )
    _write_json(
        root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                key: shard_names[0 if index < midpoint else 1] for index, key in enumerate(keys)
            },
        },
    )

    _write_native_transformer(root / f"{variant}.safetensors", transformer_marker)
    return root


__all__ = [
    "BASIC_TRANSFORMER_KEYS",
    "MAIN_BLOCK_TAILS",
    "TEXT_BLOCK_TAILS",
    "TEXT_LAYER_TAILS",
    "make_snapshot",
    "text_encoder_keys",
    "transformer_keys",
    "vae_keys",
]
