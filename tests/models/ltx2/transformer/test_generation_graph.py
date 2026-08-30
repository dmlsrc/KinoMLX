from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import kinomlx.models.ltx2.metadata as metadata_module
from kinomlx.models.ltx2.cache import ensure_transformer_cache, load_transformer_cache
from kinomlx.models.ltx2.cache.keys import flatten_to_nested
from kinomlx.models.ltx2.metadata import TransformerConstructorConfig
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality

from ._synthetic import build_shaped_ltx_model


def _config(generation: str, *, prompt_adaln: bool = True) -> TransformerConstructorConfig:
    is_25 = generation == "2.5"
    return TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=None,
        num_layers=1,
        video_in_channels=3,
        video_out_channels=3,
        video_heads=2,
        video_head_dim=4,
        audio_heads=2,
        audio_head_dim=2,
        audio_out_channels=2,
        video_context_dim=8,
        audio_context_dim=4,
        caption_channels=8,
        video_max_pos=(8, 8, 8),
        audio_max_pos=(8,),
        positional_embedding_theta=10000.0,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1000.0,
        norm_eps=1e-6,
        ff_bias=not is_25,
        audio_ff_bias=True,
        use_keyframes_abs_pos_embedding=is_25,
        use_prompt_adaln_single=prompt_adaln,
        config_digest=f"synthetic-{generation}-{prompt_adaln}",
    )


def _model(config: TransformerConstructorConfig, *, shaped: bool) -> LTXAVModel:
    def factory() -> LTXAVModel:
        return LTXAVModel.from_config(
            config,
            compute_dtype=mx.float32,
            use_steel_attention=False,
            compile_attention=False,
        )

    return build_shaped_ltx_model(factory) if shaped else factory()


def _video_modality(*, keyframes_mask: mx.array | None = None) -> Modality:
    positions = mx.array([[[[0, 1], [1, 2]], [[0, 1], [1, 2]], [[0, 1], [1, 2]]]])
    return Modality(
        latent=mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        context=mx.ones((1, 1, 8)),
        timesteps=mx.full((1, 2), 0.5),
        sigma=mx.array([0.5]),
        positions=positions,
        keyframes_mask=keyframes_mask,
    )


def test_ltx25_constructor_allocates_only_its_selected_parameter_graph() -> None:
    model = _model(_config("2.5"), shaped=False)
    keys = {key for key, _value in tree_flatten(model.parameters())}

    assert "keyframes_abs_pos_embedding" in keys
    assert "transformer_blocks.0.ff.project_in.proj.bias" not in keys
    assert "transformer_blocks.0.ff.project_out.bias" not in keys
    assert "transformer_blocks.0.audio_ff.project_in.proj.bias" in keys
    assert "transformer_blocks.0.audio_ff.project_out.bias" in keys


@pytest.mark.parametrize(
    ("generation", "prompt_adaln"),
    [("2.3", True), ("2.3", False), ("2.5", True), ("2.5", False)],
)
def test_selected_model_and_central_graph_contract_have_identical_targets(
    generation: str,
    prompt_adaln: bool,
) -> None:
    config = _config(generation, prompt_adaln=prompt_adaln)
    model = _model(config, shaped=False)

    assert {key for key, _value in tree_flatten(model.parameters())} == set(
        metadata_module.transformer_parameter_shapes(config)
    )


def test_keyframe_embedding_is_applied_only_to_marked_projected_tokens() -> None:
    model = _model(_config("2.5"), shaped=True)
    model.patchify_proj.weight = mx.eye(3, 8).T
    model.patchify_proj.bias = mx.zeros((8,))
    model.keyframes_abs_pos_embedding = mx.arange(8, dtype=mx.float32)[None, :]

    unmarked = model._video_preprocessor.prepare(_video_modality(), None).x
    marked = model._video_preprocessor.prepare(
        _video_modality(keyframes_mask=mx.array([[[0.0], [0.25]]])),
        None,
    ).x
    mx.eval(unmarked, marked)

    assert mx.array_equal(marked[:, :1], unmarked[:, :1]).item()
    assert mx.array_equal(
        marked[:, 1:] - unmarked[:, 1:],
        model.keyframes_abs_pos_embedding[None, ...],
    ).item()


def test_static_prompt_modulation_omits_prompt_adaln_parameters_and_runs() -> None:
    model = _model(_config("2.5", prompt_adaln=False), shaped=True)
    keys = {key for key, _value in tree_flatten(model.parameters())}
    assert not any(key.startswith("prompt_adaln_single.") for key in keys)
    assert not any(key.startswith("audio_prompt_adaln_single.") for key in keys)

    output, audio = model(_video_modality())
    mx.eval(output)
    assert audio is None
    assert tuple(output.shape) == (1, 2, 3)
    assert mx.all(mx.isfinite(output)).item()


def test_full_cache_binding_rejects_the_other_generation_before_mutation(tmp_path) -> None:
    source = _model(_config("2.3"), shaped=True)
    cache = tmp_path / "ltx23.safetensors"
    mx.save_safetensors(str(cache), dict(tree_flatten(source.parameters())))
    target = _model(_config("2.5"), shaped=False)
    before = {key: tuple(value.shape) for key, value in tree_flatten(target.parameters())}

    with pytest.raises(ValueError, match="parameter graph mismatch"):
        load_transformer_cache(target, cache)

    after = {key: tuple(value.shape) for key, value in tree_flatten(target.parameters())}
    assert after == before


def test_transformer_header_binding_requires_every_target_and_ignores_baggage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("2.5")
    shapes = metadata_module.transformer_parameter_shapes(config)
    header: dict[str, object] = {}
    for target, shape in shapes.items():
        source = target
        source = source.replace(".ff.project_in.proj.", ".ff.net.0.proj.")
        source = source.replace(".ff.project_out.", ".ff.net.2.")
        source = source.replace(".audio_ff.project_in.proj.", ".audio_ff.net.0.proj.")
        source = source.replace(".audio_ff.project_out.", ".audio_ff.net.2.")
        source = source.replace(".to_out.", ".to_out.0.")
        header[f"community.model.diffusion_model.{source}"] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [0, 0],
        }
    header["community.unused.training_tensor"] = {
        "dtype": "BF16",
        "shape": [7],
        "data_offsets": [0, 0],
    }
    monkeypatch.setattr(metadata_module, "read_header", lambda _path: header)

    bindings = metadata_module.resolve_transformer_bindings(
        Path("community.safetensors"),
        config,
    )

    assert len(bindings) == len(shapes)
    assert {binding.target_key for binding in bindings} == set(shapes)

    missing_target = "transformer_blocks.0.ff.project_out.weight"
    missing_source = next(
        binding.source_key for binding in bindings if binding.target_key == missing_target
    )
    del header[missing_source]
    with pytest.raises(ValueError, match="LTX-2.5.*missing consumed transformer target"):
        metadata_module.resolve_transformer_bindings(
            Path("community.safetensors"),
            config,
        )


def test_transformer_config_identity_changes_with_prompt_policy() -> None:
    dynamic = _config("2.5")
    static = replace(
        dynamic,
        use_prompt_adaln_single=False,
        config_digest="synthetic-2.5-false",
    )
    assert dynamic.cache_identity() != static.cache_identity()


def test_bias_free_graph_matches_zero_bias_synthetic_block() -> None:
    with_bias = _model(_config("2.3"), shaped=True)
    bias_free = _model(_config("2.5"), shaped=True)
    source = dict(tree_flatten(with_bias.parameters()))
    target_keys = {key for key, _value in tree_flatten(bias_free.parameters())}
    shared = {key: value for key, value in source.items() if key in target_keys}
    bias_free.update(flatten_to_nested(shared))
    with_bias.transformer_blocks[0].ff.project_in.proj.bias = mx.zeros((32,))
    with_bias.transformer_blocks[0].ff.project_out.bias = mx.zeros((8,))

    expected = with_bias(_video_modality())[0]
    actual = bias_free(_video_modality())[0]
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def test_binding_driven_cache_build_accepts_wrapper_and_ignores_baggage(
    tmp_path: Path,
) -> None:
    config = _config("2.5")
    weights: dict[str, mx.array] = {}
    for target, shape in metadata_module.transformer_parameter_shapes(config).items():
        source = target
        source = source.replace(".ff.project_in.proj.", ".ff.net.0.proj.")
        source = source.replace(".ff.project_out.", ".ff.net.2.")
        source = source.replace(".audio_ff.project_in.proj.", ".audio_ff.net.0.proj.")
        source = source.replace(".audio_ff.project_out.", ".audio_ff.net.2.")
        source = source.replace(".to_out.", ".to_out.0.")
        weights[f"community.wrapper.diffusion_model.{source}"] = mx.zeros(
            shape,
            dtype=mx.bfloat16,
        )
    weights["community.unused.training_tensor"] = mx.ones((7,))
    checkpoint = tmp_path / "community-transformer.safetensors"
    mx.save_safetensors(str(checkpoint), weights)

    result = ensure_transformer_cache(
        checkpoint,
        cache_mode="rebuild",
        cache_root=tmp_path / "cache",
        include_audio=True,
        video_ff_layout_specs=(),
        video_attn_layout_specs=(),
        constructor_config=config,
    )
    target = _model(config, shaped=False)
    loaded, layouts, quantized = load_transformer_cache(
        target,
        result.cache_path,
        include_audio=True,
    )

    assert loaded == len(metadata_module.transformer_parameter_shapes(config))
    assert (layouts, quantized) == (0, 0)
