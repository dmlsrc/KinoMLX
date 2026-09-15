"""Krea 2 generation request, sampler, batching, and lifecycle contracts."""

import weakref
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.pipelines.text_to_image as text_to_image
from kinomlx.components import ComponentLease
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.krea2.catalog import Krea2VAEDecoder, Krea2Variant
from kinomlx.models.krea2.metadata import default_scheduler_config
from kinomlx.models.krea2.pipelines import generate_krea2
from kinomlx.models.krea2.sampling import (
    Krea2DenoiseError,
    Krea2Recipe,
    denoise_krea2,
    krea_cfg,
    pack_latents,
    prepare_branch,
    resolve_geometry,
    resolve_recipe,
    timestep_schedule,
    unpack_latents,
)
from kinomlx.models.krea2.text_conditioning import (
    Krea2PromptConditioning,
    Krea2TextConditioning,
)
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.models.krea2.vae import Krea2DecodedRGB
from kinomlx.reporting import TimingReporter
from tests._reporting import RecordingReporter


def _resources(
    variant: Krea2Variant,
    *,
    vae_decoder: Krea2VAEDecoder = Krea2VAEDecoder.QWEN,
) -> SimpleNamespace:
    return SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        variant=variant,
        revision=f"{variant.value}-revision",
        recipe_identity=f"{variant.value}-recipe",
        reference_denoise=False,
        check_adapter_weights=True,
        scheduler_config=default_scheduler_config(),
        capabilities=SimpleNamespace(
            spatial_compression=8,
            transformer_patch_size=2,
            latent_channels=16,
        ),
        transformer=SimpleNamespace(dtype_policy="bfloat16"),
        vae=SimpleNamespace(dtype_policy="bfloat16"),
        vae_decoder=vae_decoder,
        vae_config=SimpleNamespace(
            spatial_compression=8,
            base_dim=96,
            dim_mult=(1, 2, 4, 4),
            input_channels=3,
        ),
    )


def _prompt_conditioning(
    prompts: tuple[str, ...],
    *,
    marker: float,
) -> Krea2PromptConditioning:
    batch = len(prompts)
    mask_rows = [[True, *([index % 2 == 0] * 2), True] for index in range(batch)]
    return Krea2PromptConditioning(
        hidden_states=mx.full((batch, 4, 1, 2), marker, dtype=mx.bfloat16),
        attention_mask=mx.array(mask_rows, dtype=mx.bool_),
        position_ids=mx.zeros((3, batch, 4), dtype=mx.int32),
        prompts=prompts,
        token_identity=f"tokens-{marker}",
    )


class _Conditioner:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __call__(
        self,
        prompts,
        _resources,
        *,
        negative_prompts=None,
        max_sequence_length=512,
        system_prompt=None,
        reporter=None,
    ) -> Krea2TextConditioning:
        del reporter
        positive_prompts = tuple(prompts)
        negative = None if negative_prompts is None else tuple(negative_prompts)
        self.events.append(("text", positive_prompts, negative, max_sequence_length))
        provenance = SimpleNamespace(to_metadata=lambda: {"identity": "fixture"})
        return Krea2TextConditioning(
            positive=_prompt_conditioning(positive_prompts, marker=1.0),
            negative=(None if negative is None else _prompt_conditioning(negative, marker=0.0)),
            provenance=provenance,
        )


class _Transformer:
    adapter_receipts = ()

    def __init__(self, events: list[object], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def __call__(
        self,
        image_tokens,
        *,
        encoder_hidden_states,
        timestep,
        position_ids,
        attention_mask,
        observer=None,
    ) -> SimpleNamespace:
        del position_ids, attention_mask, observer
        branch = "positive" if mx.max(encoder_hidden_states).item() else "negative"
        self.events.append(("transformer", branch, float(timestep[0].item())))
        if self.fail:
            raise RuntimeError("transformer stopped")
        marker = 0.25 if branch == "positive" else -0.5
        return SimpleNamespace(sample=mx.full(image_tokens.shape, marker, image_tokens.dtype))


class _Decoder:
    def __init__(self, events: list[object], *, output_scale: int) -> None:
        self.events = events
        self.output_scale = output_scale

    def __call__(self, latent, *, reporter=None, observer=None, tiling=None) -> Krea2DecodedRGB:
        del reporter, observer
        self.events.append(("decode", tuple(latent.shape), tiling is not None))
        height = latent.shape[2] * 8 * self.output_scale
        width = latent.shape[3] * 8 * self.output_scale
        return Krea2DecodedRGB(mx.zeros((1, height, width, 3), dtype=latent.dtype))


class _Components:
    def __init__(self, events: list[object], *, fail_transformer: bool = False) -> None:
        self.events = events
        self.fail_transformer = fail_transformer

    def transformer(self, _resources, *, adapter_plan=None) -> ComponentLease[_Transformer]:
        self.events.append("load-transformer")
        return ComponentLease(
            _Transformer(self.events, fail=self.fail_transformer),
            cleanup=lambda: self.events.append("close-transformer"),
        )

    def image_decoder(self, resources) -> ComponentLease[_Decoder]:
        self.events.append("load-decoder")
        return ComponentLease(
            _Decoder(self.events, output_scale=resources.vae_decoder.output_scale),
            cleanup=lambda: self.events.append("close-decoder"),
        )


def test_variant_defaults_do_not_cross_contaminate() -> None:
    geometry = resolve_geometry(
        Krea2Request("test"),
        spatial_compression=8,
        patch_size=2,
    )
    scheduler = default_scheduler_config()

    turbo = resolve_recipe(Krea2Variant.TURBO, Krea2Request("test"), geometry, scheduler)
    raw = resolve_recipe(Krea2Variant.RAW, Krea2Request("test"), geometry, scheduler)

    assert (turbo.steps, turbo.guidance, turbo.mu) == (8, 0.0, 1.15)
    assert (raw.steps, raw.guidance, raw.mu) == (52, 3.5, None)
    assert turbo.effective_mu == 1.15
    assert raw.effective_mu == pytest.approx(0.90625)


def test_geometry_rounds_up_then_pack_and_unpack_are_inverse() -> None:
    request = Krea2Request("test", width=17, height=31)
    geometry = resolve_geometry(request, spatial_compression=8, patch_size=2)
    latent = mx.arange(1 * 16 * 4 * 4, dtype=mx.float32).reshape(1, 16, 4, 4)

    packed = pack_latents(latent, patch_size=2)
    unpacked = unpack_latents(packed, geometry, latent_channels=16, patch_size=2)

    assert (geometry.width, geometry.height) == (32, 32)
    assert tuple(packed.shape) == (1, 4, 64)
    assert mx.array_equal(unpacked, latent).item()


def test_krea_cfg_uses_conditional_offset_not_standard_cfg() -> None:
    conditional = mx.array([2.0], dtype=mx.bfloat16)
    unconditional = mx.array([1.0], dtype=mx.bfloat16)

    actual = krea_cfg(conditional, unconditional, 3.5)
    standard = unconditional + mx.array(3.5, dtype=mx.bfloat16) * (conditional - unconditional)

    assert float(actual[0].item()) == 5.5
    assert float(standard[0].item()) == 4.5


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_raw_cfg_compacts_different_positive_and_negative_lengths_independently(dtype) -> None:
    request = Krea2Request("test", width=16, height=16, steps=2, guidance=1.0)
    geometry = resolve_geometry(request, spatial_compression=8, patch_size=2)
    image_tokens = mx.zeros((1, 1, 64), dtype=dtype)
    positive = _prompt_conditioning(("test",), marker=1.0)
    negative = Krea2PromptConditioning(
        hidden_states=mx.zeros((1, 4, 1, 2), dtype=dtype),
        attention_mask=mx.array([[True, False, False, True]]),
        position_ids=mx.zeros((3, 1, 4), dtype=mx.int32),
        prompts=("",),
        token_identity="negative-tokens",
    )
    positive_branch = prepare_branch(image_tokens, positive, geometry, activation_dtype=dtype)
    negative_branch = prepare_branch(image_tokens, negative, geometry, activation_dtype=dtype)
    assert positive_branch.encoder_hidden_states.shape[1] == 4
    assert negative_branch.encoder_hidden_states.shape[1] == 2
    assert positive_branch.attention_mask is negative_branch.attention_mask is None
    states = []

    def transformer(current, *, encoder_hidden_states, attention_mask, **kwargs):
        del kwargs
        assert attention_mask is None
        states.append(current)
        velocity = 0.25 if encoder_hidden_states.shape[1] == 4 else -0.5
        return SimpleNamespace(sample=mx.full(current.shape, velocity, dtype=dtype))

    result = denoise_krea2(
        transformer,
        positive_branch,
        resolve_recipe(Krea2Variant.RAW, request, geometry, default_scheduler_config()),
        negative=negative_branch,
    )
    assert result.conditional_calls == result.unconditional_calls == 2
    assert states[0] is states[1]
    assert states[2] is states[3]
    assert mx.all(mx.isfinite(result.packed)).item()
    # CFG velocity is 0.25 + (0.25 - -0.5) = 1, integrated from time 1 to 0.
    assert mx.allclose(result.packed, -mx.ones_like(result.packed), atol=0.005).item()


@pytest.mark.parametrize(
    ("dtype", "conditional", "unconditional", "guidance", "expected"),
    [
        (mx.float16, -1.96875, 0.4921875, 3.7, -11.078125),
        (mx.bfloat16, -1.71875, 0.4296875, 3.7, -9.6875),
        (mx.float16, -1.9375, 0.484375, 7.1, -19.125),
        (mx.bfloat16, -1.9375, 0.484375, 7.1, -19.25),
        (mx.float32, 2.0, 1.0, 3.7, 5.7),
    ],
)
def test_cfg_preserves_python_scalar_opmath(dtype, conditional, unconditional, guidance, expected):
    # Literal outputs from Torch's conditional + guidance * (conditional - unconditional).
    actual = krea_cfg(mx.array([conditional], dtype), mx.array([unconditional], dtype), guidance)
    assert actual.dtype == dtype
    assert mx.array_equal(actual, mx.array([expected], dtype)).item()


@pytest.mark.parametrize(("dtype", "axis_tokens"), [(mx.bfloat16, 258), (mx.float16, 2050)])
@pytest.mark.parametrize("vertical", [False, True])
def test_image_coordinates_remain_exact_beyond_activation_integer_precision(
    dtype, axis_tokens, vertical
) -> None:
    width, height = (16, axis_tokens * 16) if vertical else (axis_tokens * 16, 16)
    geometry = resolve_geometry(
        Krea2Request("test", width=width, height=height), spatial_compression=8, patch_size=2
    )
    branch = prepare_branch(
        mx.zeros((2, axis_tokens, 64), dtype),
        _prompt_conditioning(("one", "two"), marker=1.0),
        geometry,
        activation_dtype=dtype,
    )
    image_positions = branch.position_ids[:, 4:, :]
    varying_axis = 1 if vertical else 2
    assert branch.position_ids.dtype == mx.float32
    assert branch.encoder_hidden_states.dtype == dtype
    assert mx.array_equal(
        image_positions[:, :, varying_axis],
        mx.broadcast_to(mx.arange(axis_tokens, dtype=mx.float32), (2, axis_tokens)),
    ).item()
    assert mx.all(branch.position_ids[:, :4] == 0).item()
    assert mx.all(image_positions[:, :, 0] == 0).item()
    assert mx.all(image_positions[:, :, 3 - varying_axis] == 0).item()


def test_fixed_and_dynamic_shift_schedules_remain_distinct() -> None:
    turbo = Krea2Recipe(Krea2Variant.TURBO, 8, 0.0, 1.15, 1.15, False, True)
    raw = Krea2Recipe(Krea2Variant.RAW, 8, 3.5, None, 0.5, True, False)

    turbo_schedule = timestep_schedule(turbo)
    raw_schedule = timestep_schedule(raw)
    mx.eval(turbo_schedule, raw_schedule)

    assert tuple(turbo_schedule.shape) == tuple(raw_schedule.shape) == (9,)
    assert float(turbo_schedule[0].item()) == float(raw_schedule[0].item()) == 1.0
    assert float(turbo_schedule[-1].item()) == float(raw_schedule[-1].item()) == 0.0
    assert not mx.array_equal(turbo_schedule, raw_schedule).item()


@pytest.mark.requires_metal
@pytest.mark.parametrize("cfg_enabled", [False, True])
def test_denoise_reports_one_evaluated_interval_per_step(monkeypatch, cfg_enabled) -> None:
    import kinomlx.models.krea2.sampling as sampling

    now = [0.0]
    original_eval = mx.eval

    def evaluate(*values):
        original_eval(*values)
        if len(values) == 1:
            now[0] += 2.0

    monkeypatch.setattr(sampling.mx, "eval", evaluate)
    events = RecordingReporter()
    reporter = TimingReporter(events, clock=lambda: now[0])
    geometry = resolve_geometry(
        Krea2Request("test", width=16, height=16), spatial_compression=8, patch_size=2
    )
    branch = prepare_branch(
        mx.zeros((1, 1, 64), dtype=mx.bfloat16),
        _prompt_conditioning(("test",), marker=1.0),
        geometry,
        activation_dtype=mx.bfloat16,
    )
    recipe = Krea2Recipe(Krea2Variant.RAW, 3, 3.5, 1.15, 1.15, cfg_enabled, False)
    result = denoise_krea2(
        _Transformer([]),
        branch,
        recipe,
        negative=branch if cfg_enabled else None,
        reporter=reporter,
    )

    assert result.conditional_calls == 3
    assert result.unconditional_calls == (3 if cfg_enabled else 0)
    (phase,) = reporter.to_dict()["phases"]
    assert phase["phase"] == "denoise"
    assert phase["duration_seconds"] == 6.0
    assert [advance["duration_seconds"] for advance in phase["advances"]] == [2.0] * 3
    assert events.events == [
        ("start", "denoise", {"total": 3, "unit": "step"}),
        *[("advance", "denoise", {"advance": 1.0})] * 3,
        ("end", "denoise", {}),
    ]


def test_turbo_multi_prompt_generation_skips_every_unconditional_operation() -> None:
    events: list[object] = []
    request = Krea2Request(
        ("short", "a longer prompt"),
        width=17,
        height=19,
        seed=5,
        images_per_prompt=2,
        noise_backend="mlx",
        vae_tiling="single",
    )

    output = generate_krea2(
        request,
        _resources(Krea2Variant.TURBO),
        components=_Components(events),
        text_conditioner=_Conditioner(events),
    )

    text_events = [event for event in events if isinstance(event, tuple) and event[0] == "text"]
    # Every prompt row is encoded under one text-encoder lifetime, not one per prompt.
    assert [event[1] for event in text_events] == [("short", "a longer prompt")]
    assert text_events[0][2] is None
    transformer_events = [
        event for event in events if isinstance(event, tuple) and event[0] == "transformer"
    ]
    assert len(transformer_events) == 32
    assert {event[1] for event in transformer_events} == {"positive"}
    assert [image.seed for image in output.images] == [5, 6, 5, 6]
    assert [tuple(image.pixels.shape) for image in output.images] == [(19, 17, 3)] * 4
    assert events.index("close-transformer") < events.index("load-decoder")
    assert events[-1] == "close-decoder"
    output.close()
    assert output.closed
    assert output.images == ()


def test_saved_text_conditioning_bypasses_the_qwen_station(tmp_path, monkeypatch) -> None:
    sidecar = tmp_path / "prompt_text.safetensors"
    mx.save_safetensors(
        str(sidecar),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": '["saved prompt"]',
            "negative_prompts": "null",
        },
    )
    events: list[object] = []

    def load_saved(path, _resources, **kwargs):
        events.append(("load-saved-text", path, kwargs["metadata_policy"]))
        return _Conditioner(events)(("saved prompt",), _resources)

    def unexpected_qwen(*_args, **_kwargs):
        pytest.fail("saved text conditioning must bypass the Qwen station")

    monkeypatch.setattr(text_to_image, "load_text_conditioning", load_saved)
    with generate_krea2(
        Krea2Request(
            "saved prompt",
            text_conditioning=sidecar,
            width=16,
            height=16,
            steps=1,
            vae_tiling="single",
        ),
        _resources(Krea2Variant.TURBO),
        components=_Components(events),
        text_conditioner=unexpected_qwen,
    ) as output:
        assert output.text_conditioning is not None
        assert output.metadata["text_conditioning"] == {"identity": "fixture"}

    assert events[0] == ("load-saved-text", sidecar.absolute(), "require")


def test_upscale2x_decoder_doubles_requested_output_crop() -> None:
    request = Krea2Request(
        "test",
        width=17,
        height=19,
        steps=1,
        noise_backend="mlx",
        vae_tiling="single",
    )

    with generate_krea2(
        request,
        _resources(Krea2Variant.TURBO, vae_decoder=Krea2VAEDecoder.UPSCALE2X),
        components=_Components([]),
        text_conditioner=_Conditioner([]),
    ) as output:
        assert tuple(output.images[0].pixels.shape) == (38, 34, 3)
        assert output.metadata["decode"] == {
            "vae_decoder": "upscale2x",
            "output_scale": 2,
            "width": 34,
            "height": 38,
        }


def test_raw_cfg_broadcasts_blank_negative_and_uses_shared_current_state() -> None:
    events: list[object] = []
    request = Krea2Request(
        ("one", "two"),
        width=16,
        height=16,
        steps=2,
        noise_backend="mlx",
        vae_tiling="single",
    )

    with generate_krea2(
        request,
        _resources(Krea2Variant.RAW),
        components=_Components(events),
        text_conditioner=_Conditioner(events),
    ) as output:
        # Call counts are request totals: two prompts, two steps, both branches.
        denoise = output.metadata["denoise"]
        sigmas = denoise.pop("sigmas")
        assert denoise == {
            "mode": "compact_unmasked_pretransposed",
            "dtype": "bfloat16",
            "conditional_calls": 4,
            "unconditional_calls": 4,
            "euler": "FP32 scalar multiply narrowed before low-precision add",
        }
        # The recorded schedule drives the two steps the call counts report.
        assert len(sigmas) == 3
        assert (sigmas[0], sigmas[-1]) == (1.0, 0.0)

    text_event = next(event for event in events if isinstance(event, tuple) and event[0] == "text")
    assert text_event[2] == ("", "")
    branches = [
        event[1] for event in events if isinstance(event, tuple) and event[0] == "transformer"
    ]
    assert branches == ["positive", "negative", "positive", "negative"] * 2


def test_guidance_zero_rejects_nonempty_unused_negative_prompt() -> None:
    with pytest.raises(ValueError, match="negative_prompt is unused"):
        generate_krea2(
            Krea2Request("test", negative_prompt="unused", steps=1, noise_backend="mlx"),
            _resources(Krea2Variant.TURBO),
            components=_Components([]),
            text_conditioner=_Conditioner([]),
        )


def test_adapter_preflight_failure_precedes_text_conditioning(tmp_path, monkeypatch) -> None:
    events: list[object] = []
    resources = _resources(Krea2Variant.TURBO)
    resources.transformer_config = object()

    def fail_preflight(_config, _selections):
        events.append("validate-adapters")
        raise ValueError("malformed adapter")

    monkeypatch.setattr(text_to_image, "prepare_krea2_adapters", fail_preflight)
    with pytest.raises(ValueError, match="malformed adapter"):
        generate_krea2(
            Krea2Request(
                "test",
                adapters=({"path": tmp_path / "malformed.safetensors"},),
            ),
            resources,
            components=_Components(events),
            text_conditioner=_Conditioner(events),
        )
    assert events == ["validate-adapters"]


def test_failed_transformer_lease_closes_before_error_escapes() -> None:
    events: list[object] = []
    with pytest.raises(RuntimeError, match="transformer stopped"):
        generate_krea2(
            Krea2Request("test", steps=1, noise_backend="mlx", vae_tiling="single"),
            _resources(Krea2Variant.TURBO),
            components=_Components(events, fail_transformer=True),
            text_conditioner=_Conditioner(events),
        )
    assert "close-transformer" in events
    assert "load-decoder" not in events


@pytest.mark.requires_metal
def test_component_objects_die_before_cleanup_and_next_station() -> None:
    events: list[object] = []
    references: dict[str, weakref.ReferenceType] = {}

    def lease(name, model):
        references[name] = weakref.ref(model)

        def cleanup():
            assert references[name]() is None, f"{name} still alive at cleanup"
            events.append(f"released-{name}")

        return ComponentLease(model, cleanup=cleanup)

    class Provider:
        def transformer(self, _resources, *, adapter_plan=None):
            return lease("transformer", _Transformer(events))

        def image_decoder(self, resources):
            assert references["transformer"]() is None
            return lease(
                "decoder",
                _Decoder(events, output_scale=resources.vae_decoder.output_scale),
            )

    with generate_krea2(
        Krea2Request("test", width=16, height=16, steps=1, noise_backend="mlx"),
        _resources(Krea2Variant.TURBO),
        components=Provider(),
        text_conditioner=_Conditioner(events),
    ) as output:
        assert output.images
        assert all(reference() is None for reference in references.values())
    assert events.index("released-transformer") < events.index("released-decoder")


def test_checkpoint_cancellation_after_transformer_prevents_decoder_load() -> None:
    events: list[object] = []
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        generate_krea2(
            Krea2Request("test", steps=1, noise_backend="mlx", vae_tiling="single"),
            _resources(Krea2Variant.TURBO),
            components=_Components(events),
            text_conditioner=_Conditioner(events),
            checkpoint=checkpoint,
        )
    assert "close-transformer" in events
    assert "load-decoder" not in events


@pytest.mark.parametrize(
    ("dtype", "hint"),
    [
        (mx.float16, "FP16 activations overflowed"),
        (mx.bfloat16, "Check the transformer weights"),
    ],
)
def test_denoise_stops_at_nonfinite_step_after_closing_the_phase(dtype, hint) -> None:
    request = Krea2Request("test", width=16, height=16, steps=3, guidance=0.0)
    geometry = resolve_geometry(request, spatial_compression=8, patch_size=2)
    branch = prepare_branch(
        mx.zeros((1, 1, 64), dtype=dtype),
        _prompt_conditioning(("test",), marker=1.0),
        geometry,
        activation_dtype=dtype,
    )
    recipe = resolve_recipe(Krea2Variant.TURBO, request, geometry, default_scheduler_config())
    assert not recipe.cfg_enabled
    calls = 0

    def transformer(current, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        # One nonfinite prediction in the middle poisons every later state.
        value = float("nan") if calls == 2 else 0.25
        return SimpleNamespace(sample=mx.full(current.shape, value, dtype=dtype))

    events = RecordingReporter()
    name = str(dtype).rsplit(".", 1)[-1]
    with pytest.raises(Krea2DenoiseError, match=f"at step 2 in {name}") as info:
        denoise_krea2(transformer, branch, recipe, reporter=events)
    assert hint in str(info.value)
    assert calls == 2
    assert events.events[-1] == ("end", "denoise", {})


@pytest.mark.parametrize("invalid", [False, True])
def test_denoise_checks_adapter_destination_only_on_first_use(invalid):
    from kinomlx.adapters.validation import AdapterWeightCheck, AdapterWeightError
    from kinomlx.modeling.loading import WeightTarget

    request = Krea2Request("test", width=16, height=16, steps=3, guidance=0.0)
    geometry = resolve_geometry(request, spatial_compression=8, patch_size=2)
    branch = prepare_branch(
        mx.zeros((1, 1, 64)),
        _prompt_conditioning(("test",), marker=1.0),
        geometry,
        activation_dtype=mx.float32,
    )
    recipe = resolve_recipe(Krea2Variant.TURBO, request, geometry, default_scheduler_config())

    class Model:
        def __init__(self):
            self.weight = mx.array(float("inf") if invalid else 0.25)
            self.adapter_weight_check = AdapterWeightCheck()
            self.adapter_weight_check.add("weight", WeightTarget(self, "weight", ()))
            self.calls = 0

        def __call__(self, current, **kwargs):
            self.calls += 1
            return SimpleNamespace(sample=mx.ones_like(current) * self.weight)

    model = Model()
    reporter = RecordingReporter()
    if invalid:
        with pytest.raises(AdapterWeightError, match="step 1"):
            denoise_krea2(model, branch, recipe, reporter=reporter)
        assert model.calls == 1
    else:
        denoise_krea2(model, branch, recipe, reporter=reporter)
        assert model.calls == 3
        assert not model.adapter_weight_check.targets
    checks = [event for event in reporter.events if event[:2] == ("start", "check adapter weights")]
    assert len(checks) == 1
    assert reporter.events[-1][:2] == ("end", "denoise")


@pytest.mark.parametrize(("mode", "loads"), [("sequential", 3), ("station", 1), ("resident", 1)])
def test_seed_modes_reuse_conditioning_and_preserve_noise(mode, loads):
    events = []
    request = Krea2Request(
        "test", seeds=(9, 2, 9), seed_mode=mode, width=16, height=16, steps=1, vae_tiling="single"
    )
    with generate_krea2(
        request,
        _resources(Krea2Variant.TURBO),
        components=_Components(events),
        text_conditioner=_Conditioner(events),
    ) as output:
        assert [image.seed for image in output.images] == [9, 2, 9]
        assert mx.array_equal(output.initial_noise[0], output.initial_noise[2]).item()
        assert not mx.array_equal(output.initial_noise[0], output.initial_noise[1]).item()
        assert output.metadata["noise"]["per_image_seeds"] == [9, 2, 9]
        assert len(output.metadata["noise"]["streams"]) == 3
    assert sum(isinstance(event, tuple) and event[0] == "text" for event in events) == 1
    assert events.count("load-transformer") == events.count("close-transformer") == loads
    assert events.count("load-decoder") == events.count("close-decoder") == loads
    decode_events = [event for event in events if isinstance(event, tuple) and event[0] == "decode"]
    assert all(event[1][0] == 1 for event in decode_events)
    if mode == "resident":
        assert events.index("load-decoder") < events.index("close-transformer")
    else:
        assert events.index("close-transformer") < events.index("load-decoder")
    if mode == "station":
        assert max(
            index
            for index, event in enumerate(events)
            if isinstance(event, tuple) and event[0] == "transformer"
        ) < events.index("load-decoder")


def test_multi_prompt_seed_list_encodes_every_prompt_under_one_text_lifetime():
    events = []
    request = Krea2Request(
        ("first", "second", "third"),
        seeds=(9, 2),
        seed_mode="station",
        width=16,
        height=16,
        steps=1,
        vae_tiling="single",
    )
    with generate_krea2(
        request,
        _resources(Krea2Variant.TURBO),
        components=_Components(events),
        text_conditioner=_Conditioner(events),
    ) as output:
        assert [(image.prompt, image.seed) for image in output.images] == [
            ("first", 9),
            ("first", 2),
            ("second", 9),
            ("second", 2),
            ("third", 9),
            ("third", 2),
        ]
        assert output.metadata["denoise"]["conditional_calls"] == 6
    text_events = [event for event in events if isinstance(event, tuple) and event[0] == "text"]
    assert [event[1] for event in text_events] == [("first", "second", "third")]


def test_joined_warnings_reports_every_seed_once_in_first_seen_order():
    outputs = [
        text_to_image.Krea2GenerationOutput((), {"warnings": ["shared", "first only"]}),
        text_to_image.Krea2GenerationOutput((), {"warnings": ["shared"]}),
        text_to_image.Krea2GenerationOutput((), {"warnings": ["last only", "shared"]}),
    ]
    assert text_to_image._joined_warnings(outputs) == ["shared", "first only", "last only"]


def test_joined_warnings_rejects_malformed_seed_metadata():
    outputs = [text_to_image.Krea2GenerationOutput((), {"warnings": "not a list"})]
    with pytest.raises(RuntimeError, match="malformed warnings"):
        text_to_image._joined_warnings(outputs)
