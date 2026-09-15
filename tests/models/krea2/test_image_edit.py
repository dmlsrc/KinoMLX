"""Reference geometry, attention, preprocessing, and edit output contracts."""

import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.config_rules import validate_record
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.io.resampling import fit_reference, resize_rgb
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.krea2.image_processing import grounding_patches
from kinomlx.models.krea2.output import plan_krea2_output
from kinomlx.models.krea2.pipelines.image_edit import (
    Krea2ReferenceTokens,
    ReferenceConditionedTransformer,
)
from kinomlx.models.krea2.transformer import Krea2TransformerOutput
from kinomlx.models.krea2.transformer.reference import ReferenceAttention
from kinomlx.models.krea2.types import Krea2Request


def test_reference_bias_matches_dense_attention_and_preserves_prefix() -> None:
    mx.random.seed(39)
    q, k, v = [mx.random.normal((1, 2, 9, 8)) for _ in range(3)]
    bias = mx.array([0, 0, math.log(2), math.log(2), 0, 0, 0, 0, 0])
    split = ReferenceAttention(4, bias).attend(q, k, v, scale=8**-0.5)
    dense = mx.zeros((9, 9))
    dense[4:] = bias
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5, mask=dense)
    assert mx.allclose(split, expected, atol=2e-6).item()


def test_joint_reference_sequence_compacts_text_and_returns_only_target() -> None:
    captured = {}

    def model(tokens, **kwargs):
        captured.update(kwargs)
        captured["tokens"] = tokens
        length = tokens.shape[1] + kwargs["encoder_hidden_states"].shape[1]
        return Krea2TransformerOutput(tokens, length, 0, length)

    wrapper = ReferenceConditionedTransformer(
        model,
        (
            Krea2ReferenceTokens(mx.ones((1, 2, 4)), (1, 2), 2),
            Krea2ReferenceTokens(mx.full((1, 1, 4), 2), (1, 1), 0.5),
        ),
        (2, 2),
        centered=True,
    )
    target = mx.full((1, 4, 4), 3)
    result = wrapper(
        target,
        encoder_hidden_states=mx.ones((1, 3, 2, 4)),
        timestep=mx.array([0.5]),
        position_ids=mx.zeros((1, 7, 3)),
        attention_mask=mx.array([[True, False, True, True, True, True, True]]),
    )
    assert mx.array_equal(result.sample, target).item()
    assert captured["tokens"].shape == (1, 7, 4)
    assert captured["encoder_hidden_states"].shape[1] == 2
    assert captured["attention_mask"] is None
    assert captured["position_ids"][0, 2:5].tolist() == [
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 1.0],
        [2.0, 0.5, 0.5],
    ]
    assert captured["reference_attention"].target_start == 5


@pytest.mark.parametrize("mode", ["bilinear", "bicubic", "area"])
def test_resize_preserves_constant_color(mode) -> None:
    pixels = mx.broadcast_to(mx.array([0.2, 0.4, 0.8]), (1, 37, 53, 3))
    output = resize_rgb(pixels, 19, 61, mode=mode)
    assert mx.allclose(output, mx.array([0.2, 0.4, 0.8]), atol=1e-6).item()


def test_fit_preserves_aspect_ratio_and_legacy_crop_fills_target() -> None:
    pixels = mx.zeros((1, 128, 256, 3))
    assert fit_reference(pixels, 256, 256).shape == (1, 128, 256, 3)
    assert fit_reference(pixels, 256, 256, mode="crop").shape == (1, 256, 256, 3)
    assert fit_reference(mx.zeros((1, 248, 256, 3)), 256, 256).shape == (1, 256, 256, 3)


def test_grounding_patch_channel_time_and_merge_order() -> None:
    pixels = mx.arange(64 * 96 * 3).reshape(1, 64, 96, 3) / (64 * 96 * 3)
    patches, grid = grounding_patches(pixels)
    assert grid == (4, 6)
    assert patches.shape == (24, 1536)
    unpacked = patches.reshape(2, 3, 2, 2, 3, 2, 16, 16)
    restored = unpacked.transpose(5, 4, 0, 2, 6, 1, 3, 7).reshape(2, 3, 64, 96)
    expected = (pixels * 2 - 1).transpose(0, 3, 1, 2)
    assert mx.array_equal(restored[:1], expected).item()
    assert mx.array_equal(restored[1:], expected).item()


def test_edit_save_all_excludes_incomplete_text_restart(tmp_path: Path) -> None:
    request = Krea2Request(prompt="test", source_images=(tmp_path / "source.png",))
    plan = plan_krea2_output(
        request, "turbo", ImageOutputConfig(path=tmp_path / "result.png", save_all_sidecars=True)
    )
    assert plan.artifacts.text_conditioning == ()
    assert plan.artifacts.final_latents
    with pytest.raises(ValueError, match="text-only restart"):
        plan_krea2_output(
            request,
            "turbo",
            ImageOutputConfig(path=tmp_path / "other.png", save_text_conditioning=True),
        )
    with pytest.raises(ValueError, match="text-only restart"):
        plan_krea2_output(
            request,
            "turbo",
            ImageOutputConfig(
                path=tmp_path / "combined.png",
                save_all_sidecars=True,
                save_text_conditioning=True,
            ),
        )


def test_edit_save_all_implication_does_not_hide_explicit_text_sidecar(
    tmp_path: Path,
) -> None:
    from kinomlx.config import ConfigError
    from kinomlx.config_records import supplied_fields
    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.config import assemble, validate_for_execution

    base = [
        "--prompt",
        "scene",
        "--source-image",
        str(tmp_path / "source.png"),
        "--output",
        str(tmp_path / "result.png"),
        "--save-all-sidecars",
    ]
    implied = assemble(build_parser().parse_args(base))
    assert "save_text_conditioning" not in (supplied_fields(implied.output) or frozenset())
    validate_for_execution(implied)
    plan = plan_krea2_output(implied.request, implied.variant, implied.output)
    assert plan.artifacts.final_latents
    assert plan.artifacts.text_conditioning == ()

    selected = assemble(build_parser().parse_args([*base, "--save-text-conditioning"]))
    assert "save_text_conditioning" in (supplied_fields(selected.output) or frozenset())
    with pytest.raises(ConfigError, match="text-only restart"):
        validate_for_execution(selected)


def test_edit_output_cannot_overwrite_source_even_with_force(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"original")
    request = Krea2Request(prompt="test", source_images=(source,))
    with pytest.raises(match="overwrite an input"):
        plan_krea2_output(request, "turbo", ImageOutputConfig(path=source, force=True))
    assert source.read_bytes() == b"original"


def test_reference_count_limit_belongs_to_current_recipe(tmp_path: Path) -> None:
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    request = Krea2Request(prompt="test", source_images=(tmp_path / "source.png",) * 3)
    assert len(request.source_images) == 3
    with pytest.raises(ValueError, match="one or two source"):
        generate_krea2_edit(request, SimpleNamespace(memory_policy=MLXMemoryPolicy()))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("strength", [1.0, 1.3])
def test_prepared_edit_reuses_fixed_products_and_matches_unprepared(
    dtype, strength, monkeypatch
) -> None:
    from kinomlx.models.krea2.transformer import Krea2Transformer
    from kinomlx.models.krea2.transformer.loading import transformer_weight_targets

    from ._transformer_fixtures import populate_transformer, tiny_transformer_config

    model = Krea2Transformer(tiny_transformer_config())
    populate_transformer(model, transformer_weight_targets(model), dtype=dtype)
    refs = (
        Krea2ReferenceTokens(
            mx.ones((1, 2, 4)).astype(dtype), (1, 2), strength, mx.array([True, False])
        ),
    )
    wrapper = ReferenceConditionedTransformer(model, refs, (1, 2), centered=True)
    hidden = mx.ones((1, 3, 3, 16)).astype(dtype)
    positions = mx.zeros((1, 5, 3))
    prepared_calls = []
    original = model.prepare_conditioning

    def prepare(*args):
        prepared_calls.append(1)
        return original(*args)

    monkeypatch.setattr(model, "prepare_conditioning", prepare)
    for t in (0.9, 0.5, 0.1):
        target = mx.full((1, 2, 4), t, dtype=dtype)
        time = mx.array([t], dtype=dtype)
        result = wrapper(
            target,
            encoder_hidden_states=hidden,
            position_ids=positions,
            timestep=time,
            attention_mask=None,
        )
        branch = wrapper._branches[0]
        expected = model(
            mx.concatenate([branch.tokens, target], axis=1),
            encoder_hidden_states=hidden,
            position_ids=branch.positions,
            timestep=time,
            attention_mask=None,
            reference_attention=branch.bias,
        )
        assert mx.allclose(
            result.sample.astype(mx.float32), expected.sample[:, 2:].astype(mx.float32), atol=1e-6
        ).item()
    assert len(prepared_calls) == 1
    assert len(wrapper._branches) == 1
    negative = hidden * 0.5
    wrapper(
        target,
        encoder_hidden_states=negative,
        position_ids=positions,
        timestep=time,
        attention_mask=None,
    )
    wrapper(
        target,
        encoder_hidden_states=hidden,
        position_ids=positions,
        timestep=time,
        attention_mask=None,
    )
    assert len(prepared_calls) == 2
    assert len(wrapper._branches) == 2


@pytest.mark.parametrize("grid", [(2, 3), (3, 2), (1, 1)])
def test_reference_mask_matches_area_average_threshold(grid) -> None:
    from kinomlx.models.krea2.image_processing import reference_boost_selection

    pixels = mx.arange(5 * 7, dtype=mx.float32).reshape(1, 5, 7, 1) / 36
    pixels = mx.repeat(pixels, 3, axis=-1)
    expected = []
    for row in range(grid[0]):
        for col in range(grid[1]):
            top, bottom = math.floor(row * 5 / grid[0]), math.ceil((row + 1) * 5 / grid[0])
            left, right = math.floor(col * 7 / grid[1]), math.ceil((col + 1) * 7 / grid[1])
            values = [float(y * 7 + x) / 36 for y in range(top, bottom) for x in range(left, right)]
            expected.append(sum(values) / len(values) > 0.5)
    assert reference_boost_selection(pixels, grid).tolist() == expected


@pytest.mark.parametrize(("level", "selected"), [(0, False), (0.5, False), (1, True)])
def test_reference_mask_uniform_and_threshold_edges(level, selected) -> None:
    from kinomlx.models.krea2.image_processing import reference_boost_selection

    assert (
        reference_boost_selection(mx.full((1, 4, 4, 3), level), (2, 2)).tolist() == [selected] * 4
    )


@pytest.mark.parametrize("strength", [1.0, 2.0])
def test_region_bias_is_last_reference_only_and_reused(strength) -> None:
    captured = []

    def model(tokens, **kwargs):
        captured.append(kwargs.get("reference_attention"))
        return Krea2TransformerOutput(tokens, 7, 0, 7)

    wrapper = ReferenceConditionedTransformer(
        model,
        (
            Krea2ReferenceTokens(mx.ones((1, 2, 4)), (1, 2), strength),
            Krea2ReferenceTokens(mx.ones((1, 2, 4)), (1, 2), strength, mx.array([True, False])),
        ),
        (1, 2),
        centered=True,
    )
    hidden, positions = mx.ones((1, 1, 2, 4)), mx.zeros((1, 3, 3))
    for time in (0.9, 0.5):
        wrapper(
            mx.ones((1, 2, 4)),
            encoder_hidden_states=hidden,
            position_ids=positions,
            timestep=mx.array([time]),
            attention_mask=None,
        )
    assert len(wrapper._branches) == 1
    if strength == 1:
        assert captured == [None, None]
    else:
        assert captured[0] is captured[1]
        assert mx.allclose(
            captured[0].key_bias, mx.array([0, math.log(2), math.log(2), math.log(2), 0, 0, 0])
        ).item()


def test_mask_is_protected_from_output_overwrite(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"original")
    request = Krea2Request(
        prompt="test",
        source_images=(tmp_path / "source.png",),
        reference_boost_mask=mask,
        reference_strength=2,
    )
    with pytest.raises(match="overwrite an input"):
        plan_krea2_output(request, "turbo", ImageOutputConfig(path=mask, force=True))


def test_edit_grounding_options_roundtrip_and_require_sources(tmp_path: Path) -> None:
    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.types import Krea2GenerateConfig

    (tmp_path / "mask.png").write_bytes(b"mask")
    args = build_parser().parse_args(
        [
            "--source-image",
            str(tmp_path / "source.png"),
            "--reference-boost-mask",
            str(tmp_path / "mask.png"),
            "--grounding-system-prompt",
            "Focus on identity.",
        ]
    )
    request = Krea2GenerateConfig(
        prompt="test",
        source_images=tuple(args.source_images),
        reference_boost_mask=args.reference_boost_mask,
        reference_strength=2,
        grounding_system_prompt=args.grounding_system_prompt,
    ).request()
    assert request.reference_boost_mask == tmp_path / "mask.png"
    assert request.grounding_system_prompt == "Focus on identity."
    with pytest.raises(ValueError, match="require source images"):
        validate_record(
            Krea2Request(prompt="test", grounding_system_prompt="Focus on identity."),
            location="[generate]",
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"reference_boost_regions": ("rect:0,0,50,100",)}, "other than 1"),
        ({"scene_reference_strength": 0.8}, "two source images"),
    ],
)
def test_edit_rejects_ineffective_inputs(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_record(
            Krea2Request(prompt="test", source_images=(Path("source.png"),), **options),
            location="[generate]",
        )


def test_source_mask_and_regions_share_crop_geometry() -> None:
    from kinomlx.io.resampling import reference_fit_geometry
    from kinomlx.models.krea2.image_processing import (
        reference_boost_selection,
    )
    from kinomlx.models.krea2.reference_regions import (
        parse_reference_region,
        reference_region_selection,
    )

    top, left, ch, cw, nh, nw = reference_fit_geometry(64, 128, 64, 64, mode="crop")
    assert (top, left, ch, cw, nh, nw) == (0, 32, 64, 64, 64, 64)
    # Paint the left half of the retained source crop, not the uncropped image.
    mask = mx.broadcast_to(
        ((mx.arange(128) >= 32) & (mx.arange(128) < 64))[None, None, :, None], (1, 64, 128, 3)
    ).astype(mx.float32)
    actual = reference_boost_selection(mask[:, top : top + ch, left : left + cw], (4, 4))
    expected = reference_region_selection((parse_reference_region("rect:0,0,50,100"),), (4, 4))
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    "options",
    [{"prompt": ("one", "two")}, {"source_images": (Path("a.png"),) * 3}],
)
def test_edit_limits_fail_during_execution_validation(options, tmp_path) -> None:
    from dataclasses import replace

    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.config import assemble, validate_for_execution

    invocation = assemble(
        build_parser().parse_args(
            ["--source-image", str(tmp_path / "source.png"), "--prompt", "test"]
        )
    )
    invocation = replace(invocation, request=replace(invocation.request, **options))
    with pytest.raises(ValueError, match="one or two source images|one prompt"):
        validate_for_execution(invocation)


def test_edit_header_lists_sources_controls_and_adapter(tmp_path, caplog) -> None:
    import logging

    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.command import _log_request
    from kinomlx.models.krea2.config import assemble

    source, adapter = tmp_path / "source.png", tmp_path / "edit.safetensors"
    invocation = assemble(
        build_parser().parse_args(
            [
                "--source-image",
                str(source),
                "--prompt",
                "test",
                "--edit-adapter",
                str(adapter),
                "--reference-strength",
                "2",
                "--reference-boost-region",
                "rect:0,0,50,100",
                "--grounding-system-prompt",
                "Keep facial details.",
            ]
        )
    )
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.command"):
        _log_request(invocation)
    for expected in (
        "Action: edit",
        str(source),
        str(adapter),
        "rect:0,0,50,100",
        "Keep facial details.",
        "Edit adapter: explicit",
    ):
        assert expected in caplog.text


def test_mask_file_observation_occurs_only_at_edit_admission(tmp_path, monkeypatch) -> None:
    from kinomlx.models.krea2.edit_validation import validate_edit_request

    def unexpected_file_check(path):
        pytest.fail(f"configuration construction inspected {path}")

    mask = tmp_path / "mask.png"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_file", unexpected_file_check)
        request = Krea2Request(
            prompt="test",
            source_images=(tmp_path / "source.png",),
            reference_boost_mask=mask,
            reference_strength=2,
        )
    with pytest.raises(ValueError, match="reference boost mask is not a file"):
        validate_edit_request(request)
    mask.write_bytes(b"mask contents are decoded by the image loader")
    validate_edit_request(request)


def _edit_resources(tmp_path: Path) -> SimpleNamespace:
    """The component descriptors the edit handoff reads, and nothing heavier."""
    from kinomlx.models.krea2.catalog import Krea2Variant
    from kinomlx.models.krea2.metadata import default_scheduler_config

    return SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        hf_home=tmp_path / "hf",
        source_root=tmp_path / "source-root",
        variant=Krea2Variant.TURBO,
        revision="fixture-revision",
        recipe_identity="fixture-recipe",
        reference_denoise=False,
        check_adapter_weights=True,
        scheduler_config=default_scheduler_config(),
        capabilities=SimpleNamespace(
            spatial_compression=8,
            transformer_patch_size=2,
            latent_channels=16,
        ),
        transformer=SimpleNamespace(dtype_policy="bfloat16"),
        vae=SimpleNamespace(dtype_policy=mx.bfloat16),
    )


class _FakeImageEncoder:
    """Stand in for the reference VAE encoder with the layout the packer expects."""

    def __init__(self, config: object, *, compute_dtype: object) -> None:
        self.config = config
        self.compute_dtype = compute_dtype

    def __call__(self, pixels: mx.array, *, reporter: object = None) -> mx.array:
        del reporter
        return mx.zeros((1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8), dtype=mx.float32)


def _capture_edit_handoff(monkeypatch) -> dict[str, object]:
    """Replace every weight-reading step so only the handoff itself executes."""
    from kinomlx.models.krea2.pipelines import image_edit
    from kinomlx.models.krea2.pipelines.text_to_image import Krea2GenerationOutput

    captured: dict[str, object] = {}

    def fake_generate(request, resources, **kwargs):
        captured["request"] = request
        captured["resources"] = resources
        captured.update(kwargs)
        return Krea2GenerationOutput(images=())

    monkeypatch.setattr(
        image_edit, "load_image", lambda _path: mx.zeros((64, 64, 3), dtype=mx.float32)
    )
    monkeypatch.setattr(image_edit, "parse_vae_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(image_edit, "Krea2ImageEncoder", _FakeImageEncoder)
    monkeypatch.setattr(image_edit, "load_krea2_encoder_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(image_edit, "generate_krea2", fake_generate)
    return captured


def test_edit_handoff_prepends_the_synthesized_adapter_and_keeps_user_adapters(
    tmp_path: Path, monkeypatch
) -> None:
    """The shared path receives the edit adapter first and every user adapter after it."""
    from kinomlx.models.krea2.adapters.inputs import Krea2AdapterConfig
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    captured = _capture_edit_handoff(monkeypatch)
    edit_adapter = tmp_path / "identity-edit.safetensors"
    style = Krea2AdapterConfig(path=tmp_path / "style.safetensors", strength=0.8, exclude=("norm",))
    character = Krea2AdapterConfig(path=tmp_path / "character.safetensors", strength=1.4)
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "source.png",),
        adapters=(style, character),
        edit_adapter_path=edit_adapter,
        edit_adapter_strength=0.6,
        edit_adapter_exclusions="to_k,to_v",
    )

    result = generate_krea2_edit(request, _edit_resources(tmp_path))

    from kinomlx.config_records import record_intent

    derived = captured["request"]
    assert derived is not request
    # The edit adapter travels as a runtime port ahead of the request's own
    # adapters; the request itself still carries exactly what the user wrote.
    (synthesized,) = captured["leading_adapters"]
    assert synthesized.path == edit_adapter.absolute()
    assert synthesized.strength == 0.6
    assert synthesized.exclude == ("to_k", "to_v")
    user = list(derived.adapters)
    assert user == [style, character]
    assert [item.strength for item in user] == [0.8, 1.4]
    assert [item.exclude for item in user] == [("norm",), ()]
    assert record_intent(derived)["adapters"] == (
        {"path": style.path, "strength": 0.8, "exclude": ("norm",)},
        {"path": character.path, "strength": 1.4},
    )
    assert result.metadata["editing"]["edit_adapter_strength"] == 0.6
    assert result.metadata["editing"]["edit_adapter_exclusions"] == ["to_k", "to_v"]
    assert result.metadata["editing"]["adapter_resolution"]["selection"] == "explicit"


def test_edit_handoff_hands_the_shared_path_an_ordinary_generation_request(
    tmp_path: Path, monkeypatch
) -> None:
    """Every edit-only control is resolved here, so the shared path sees none of them."""
    from kinomlx.models.krea2.pipelines.image_edit import (
        ReferenceKrea2Components,
        generate_krea2_edit,
    )

    captured = _capture_edit_handoff(monkeypatch)
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "scene.png", tmp_path / "source.png"),
        edit_adapter_path=tmp_path / "identity-edit.safetensors",
        edit_adapter_strength=0.6,
        grounding_system_prompt="Keep facial details.",
        reference_strength=1.3,
        scene_reference_strength=0.5,
    )

    result = generate_krea2_edit(request, _edit_resources(tmp_path))

    derived = captured["request"]
    assert derived.source_images == ()
    assert derived.edit_adapter_path is None
    assert derived.edit_adapter_strength == 1.0
    assert derived.edit_adapter_exclusions == ""
    assert derived.grounding_system_prompt == ""
    assert derived.scene_reference_strength == 1.0
    assert derived.match_source_aspect is None
    assert derived.prompt == ("a portrait",)
    ports = captured["components"]
    assert isinstance(ports, ReferenceKrea2Components)
    assert [reference.grid for reference in ports.references] == [(4, 4), (4, 4)]
    # The scene image is first and the edited source is last, which is the image
    # --reference-strength applies to.
    assert [reference.strength for reference in ports.references] == [0.5, 1.3]
    assert ports.target_grid == (4, 4)
    assert result.metadata["editing"]["recipe"] == "image-grounded-reference-v1"
    assert result.metadata["editing"]["reference_grids"] == [[4, 4], [4, 4]]
    assert result.metadata["editing"]["reference_strengths"] == [0.5, 1.3]


def test_edit_handoff_leaves_the_adapters_a_restart_would_replay(
    tmp_path: Path, monkeypatch
) -> None:
    """The synthesized adapter is not written back into the caller's own intent.

    resolve_edit_adapters refuses an edit adapter that also appears in --adapter,
    so a request whose recorded adapters had gained the synthesized entry could
    not be replayed at all.
    """
    from kinomlx.config_records import record_intent
    from kinomlx.models.krea2.adapters.edit_profile import resolve_edit_adapters
    from kinomlx.models.krea2.adapters.inputs import Krea2AdapterConfig
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    _capture_edit_handoff(monkeypatch)
    edit_adapter = tmp_path / "identity-edit.safetensors"
    style = Krea2AdapterConfig(path=tmp_path / "style.safetensors", strength=0.8)
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "source.png",),
        adapters=(style,),
        edit_adapter_path=edit_adapter,
        edit_adapter_strength=0.6,
    )

    generate_krea2_edit(request, _edit_resources(tmp_path))

    assert request.adapters == (style,)
    assert record_intent(request)["adapters"] == ({"path": style.path, "strength": 0.8},)
    replayed, _receipt = resolve_edit_adapters(
        tmp_path / "hf",
        edit_path=edit_adapter,
        edit_strength=0.6,
        edit_exclusions="",
        additional=request.resolved_adapters(),
    )
    assert [selection.path for selection in replayed] == [edit_adapter.absolute(), style.path]


def test_edit_refuses_a_source_without_the_original_image_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    """A component-only Krea source has no VAE encoder, so editing cannot start."""
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    _capture_edit_handoff(monkeypatch)
    resources = _edit_resources(tmp_path)
    resources.source_root = None
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "source.png",),
        edit_adapter_path=tmp_path / "identity-edit.safetensors",
    )

    with pytest.raises(ValueError, match="editing requires an original Krea source root"):
        generate_krea2_edit(request, resources)


def test_edit_resolves_sizing_regions_and_cancellation_before_the_shared_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Source-matched sizing, region boosts and cancellation all belong to this stage."""
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    captured = _capture_edit_handoff(monkeypatch)
    checkpoints: list[int] = []
    request = Krea2Request(
        prompt="a portrait",
        width=128,
        height=64,
        source_images=(tmp_path / "source.png",),
        edit_adapter_path=tmp_path / "identity-edit.safetensors",
        reference_strength=1.3,
        reference_boost_regions=("rect:0,0,50,100",),
        match_source_aspect="long",
    )

    result = generate_krea2_edit(
        request,
        _edit_resources(tmp_path),
        checkpoint=lambda: checkpoints.append(len(checkpoints)),
    )

    assert checkpoints == [0]
    # The faked reference image is square, so matching its aspect squares the output.
    derived = captured["request"]
    assert (derived.width, derived.height) == (128, 128)
    assert result.metadata["editing"]["output_sizing"] is not None
    assert result.metadata["editing"]["reference_boost_regions"] == ["rect:0,0,50,100"]
    selection = captured["components"].references[-1].boost_selection
    assert selection is not None
    assert selection.dtype == mx.bool_


def test_edit_combines_a_mask_file_with_declared_regions_on_the_last_reference(
    tmp_path: Path, monkeypatch
) -> None:
    """A boost mask and a boost region select together, not one instead of the other."""
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    captured = _capture_edit_handoff(monkeypatch)
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"decoded by the faked image loader")
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "source.png",),
        edit_adapter_path=tmp_path / "identity-edit.safetensors",
        reference_strength=1.3,
        reference_boost_mask=mask,
        reference_boost_regions=("rect:0,0,50,100",),
    )

    result = generate_krea2_edit(request, _edit_resources(tmp_path))

    selection = captured["components"].references[-1].boost_selection
    assert selection is not None
    assert selection.shape == (16,)
    # The faked mask decodes to zeros, so every selected token comes from the region.
    assert selection.tolist() == [True, True, False, False] * 4
    assert result.metadata["editing"]["reference_boost_mask"] == str(mask)


def test_edit_refuses_a_boost_mask_that_does_not_match_the_edited_image(
    tmp_path: Path, monkeypatch
) -> None:
    """A mask decoded at another size would select the wrong reference tokens."""
    from kinomlx.models.krea2.pipelines import image_edit
    from kinomlx.models.krea2.pipelines.image_edit import generate_krea2_edit

    _capture_edit_handoff(monkeypatch)
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"decoded by the faked image loader")
    monkeypatch.setattr(
        image_edit,
        "load_image",
        lambda path: mx.zeros((32, 32, 3) if path == mask else (64, 64, 3), dtype=mx.float32),
    )
    request = Krea2Request(
        prompt="a portrait",
        width=64,
        height=64,
        source_images=(tmp_path / "source.png",),
        edit_adapter_path=tmp_path / "identity-edit.safetensors",
        reference_strength=1.3,
        reference_boost_mask=mask,
    )

    with pytest.raises(ValueError, match="reference boost mask dimensions must match"):
        generate_krea2_edit(request, _edit_resources(tmp_path))
