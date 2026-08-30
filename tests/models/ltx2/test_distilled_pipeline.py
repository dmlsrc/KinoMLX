"""Phase C contracts for the public two-stage distilled recipe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.conditioning.preparation as condition_preparation
import kinomlx.models.ltx2.decode as decode_module
import kinomlx.models.ltx2.pipelines.distilled as distilled
from kinomlx.components import ComponentLease
from kinomlx.lora.loading import LoRAConfig
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.artifacts import (
    STAGE_1_CONDITIONING,
    STAGE_2_CONDITIONING,
    TEXT_CONDITIONING,
)
from kinomlx.models.ltx2.precision import LTX2DTypePolicy
from kinomlx.models.ltx2.resources import ComponentKind, LTX2Resources
from kinomlx.models.ltx2.text_conditioning import (
    EncodedTextConditioning,
    TextConditioningProvenance,
)
from kinomlx.models.ltx2.types import (
    DistilledRequest,
    ImageConditioningConfig,
    VideoVAETilingConfig,
)
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG
from kinomlx.reporting import RecordingReporter
from kinomlx.samplers.noise import NoiseStreamState
from kinomlx.types import VideoPixelShape


@dataclass(frozen=True)
class _MemoryReceipt:
    station: str
    active_components: frozenset[str]
    active_bytes: int


class _Resources:
    def __init__(
        self,
        *,
        generation: str = "2.3",
        dtype_policy: LTX2DTypePolicy | None = None,
    ) -> None:
        self.dtype_policy = dtype_policy or LTX2DTypePolicy(
            transformer=mx.float32,
            latent=mx.float32,
            video_vae=mx.float32,
            spatial_upscaler=mx.float32,
            audio_vae=mx.float32,
        )
        self.capabilities = SimpleNamespace(
            model_generation=generation,
            recipe_families=("distilled",),
            condition_families=("text", "image", "keyframe"),
            generates_audio=True,
            sampler_policy=(
                "deterministic-euler-two-stage"
                if generation == "2.3"
                else "ancestral-stage1-deterministic-stage2"
            ),
            generated_keyframes=generation == "2.5",
            native_hdr=generation == "2.5",
            duration_available=generation == "2.5",
            video_compression=LTX23_VIDEO_VAE_CONFIG.encoder_scale,
        )
        self.checkpoint = SimpleNamespace(
            model_generation=generation,
            model_version=f"{generation}.0",
            source_path=Path("checkpoint.safetensors"),
            source_fingerprint="checkpoint-fingerprint",
        )
        self.components = ()
        self.execution_policy = SimpleNamespace(mlx_cache_limit_bytes=None)
        self.gemma_path = Path("gemma")
        self.weights_path = Path("checkpoint.safetensors")

    def require(self, kind: ComponentKind) -> SimpleNamespace:
        return SimpleNamespace(cache_path=Path(f"{kind.value}.safetensors"))


class _IdentityStatistics:
    def normalize(self, value: mx.array) -> mx.array:
        return value

    def denormalize(self, value: mx.array) -> mx.array:
        return value


class _X0:
    velocity_model = SimpleNamespace(close_streamer=lambda: None)
    lora_receipts = ()

    def __init__(self, owner: _RecordingComponents, load_index: int, *, fail: bool) -> None:
        self._owner = owner
        self._load_index = load_index
        self._fail = fail
        self.video_token_counts: list[int] = []

    def __call__(self, video, audio=None):
        station = (
            "stage 2" if self._load_index > 0 or len(self.video_token_counts) >= 8 else "stage 1"
        )
        self._owner._use("transformer", station)
        if self._owner.fail_at == station:
            raise RuntimeError(f"{station} failed")
        if self._fail:
            raise RuntimeError(f"transformer-{self._load_index} failed")
        self.video_token_counts.append(video.latent.shape[1])
        return (
            mx.zeros_like(video.latent),
            None if audio is None else mx.zeros_like(audio.latent),
        )


class _Upscaler:
    per_channel_statistics = _IdentityStatistics()

    def __init__(self, owner: _RecordingComponents) -> None:
        self._owner = owner

    def __call__(self, latent, *, reporter=None):
        del reporter
        self._owner._use("spatial_upscaler", "upscale")
        if self._owner.fail_at == "upscale":
            raise RuntimeError("upscale failed")
        return mx.tile(latent, (1, 1, 1, 2, 2))


class _AudioDecoder:
    def __init__(self, owner: _RecordingComponents) -> None:
        self._owner = owner

    def __call__(self, latent, *, reporter=None):
        del reporter
        self._owner._use("audio_decoder", "audio decode")
        if self._owner.fail_at == "audio decode":
            raise RuntimeError("audio decode failed")
        return mx.zeros((1, 2, latent.shape[2], 64), dtype=latent.dtype)


class _Vocoder:
    output_sample_rate = 48_000

    def __init__(self, owner: _RecordingComponents) -> None:
        self._owner = owner

    def __call__(self, mel, *, reporter=None):
        del reporter
        self._owner._use("vocoder", "vocoder")
        if self._owner.fail_at == "vocoder":
            raise RuntimeError("vocoder failed")
        return mx.zeros((1, 2, mel.shape[2] * 640), dtype=mel.dtype)


class _VideoDecoder:
    def __init__(self, owner: _RecordingComponents) -> None:
        self.owner = owner


class _DurationPredictor:
    def __init__(self, owner: _RecordingComponents, frames: int = 17) -> None:
        self._owner = owner
        self._frames = frames

    def predict_num_frames(self, *_args, **_kwargs) -> int:
        self._owner._use("duration_predictor", "auto duration")
        return self._frames


class _RecordingComponents:
    def __init__(
        self,
        *,
        fail_at: str | None = None,
        fail_transformer_loads: frozenset[int] = frozenset(),
    ) -> None:
        self.fail_at = fail_at
        self.fail_transformer_loads = fail_transformer_loads
        self.active: set[str] = set()
        self.events: list[tuple[str, str, frozenset[str]]] = []
        self.memory_receipts: list[_MemoryReceipt] = []
        self.transformer_profiles: list[tuple[LoRAConfig, ...]] = []
        self.transformers: list[_X0] = []
        self.video_decoder_dtypes: list[mx.Dtype] = []
        self.close_counts: dict[str, int] = {}

    def _load(self, name: str) -> None:
        if name in self.active:
            raise AssertionError(f"duplicate live component: {name}")
        self.active.add(name)
        self.events.append(("load", name, frozenset(self.active)))

    def _use(self, name: str, station: str) -> None:
        if name not in self.active:
            raise AssertionError(f"use of inactive component: {name}")
        self.events.append(("use", station, frozenset(self.active)))

    def _close(self, name: str) -> None:
        if name not in self.active:
            raise AssertionError(f"close of inactive component: {name}")
        self.active.remove(name)
        self.close_counts[name] = self.close_counts.get(name, 0) + 1
        self.events.append(("close", name, frozenset(self.active)))

    def _lease(self, name: str, component):
        self._load(name)
        return ComponentLease(
            component,
            close_component=lambda _component: self._close(name),
        )

    def transformer(self, resources, profile):
        del resources
        load_index = len(self.transformer_profiles)
        self.transformer_profiles.append(tuple(profile))
        transformer = _X0(
            self,
            load_index,
            fail=load_index in self.fail_transformer_loads,
        )
        self.transformers.append(transformer)
        return self._lease("transformer", transformer)

    def video_encoder(self, resources):
        del resources
        return self._lease("video_encoder", SimpleNamespace())

    def spatial_upscaler(self, resources):
        del resources
        self.memory_receipts.append(
            _MemoryReceipt(
                station="before spatial upscaler load",
                active_components=frozenset(self.active),
                active_bytes=mx.get_active_memory(),
            )
        )
        return self._lease("spatial_upscaler", _Upscaler(self))

    def duration_predictor(self, resources):
        del resources
        return self._lease("duration_predictor", _DurationPredictor(self))

    def audio_decoder(self, resources):
        del resources
        return self._lease("audio_decoder", _AudioDecoder(self))

    def vocoder(self, resources):
        del resources
        return self._lease("vocoder", _Vocoder(self))

    def video_decoder(self, resources):
        self.video_decoder_dtypes.append(resources.dtype_policy.video_vae)
        return self._lease("video_decoder", _VideoDecoder(self))


def _text(prompt: str = "test", tokens: int = 1) -> EncodedTextConditioning:
    return EncodedTextConditioning(
        video_encoding=mx.zeros((1, tokens, 4096)),
        audio_encoding=mx.zeros((1, tokens, 2048)),
        attention_mask=mx.ones((1, tokens)),
        prompt=prompt,
        provenance=TextConditioningProvenance(
            model_generation="ltx-2.3",
            text_encoder_identity="gemma-3-12b-it",
            projection_identity="connector:test",
        ),
    )


@pytest.mark.parametrize(
    ("attention_mask", "expect_elided"),
    [
        (mx.array([[0, 1]], dtype=mx.int32), False),
        (mx.ones((1, 2), dtype=mx.int32), True),
    ],
)
def test_run_stage_elides_only_an_all_valid_connector_mask(
    monkeypatch,
    attention_mask: mx.array,
    expect_elided: bool,
) -> None:
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        generate_audio=True,
    )
    stage = distilled.prepare_stage(
        request,
        VideoPixelShape(batch=1, frames=9, height=64, width=64),
        dtype_policy=_Resources().dtype_policy,
    )
    text = replace(
        _text(tokens=2),
        attention_mask=attention_mask,
    )
    captured = {}

    class _Captured(RuntimeError):
        pass

    def capture_loop(*_args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(distilled, "denoise_loop", capture_loop)
    with pytest.raises(_Captured):
        distilled.run_stage(
            stage,
            _X0(_RecordingComponents(), 0, fail=False),
            text,
            (1.0, 0.0),
            noiser=lambda state, *, scale: state,
        )

    if expect_elided:
        assert captured["video_context_mask"] is None
        assert captured["audio_context_mask"] is None
    else:
        assert mx.array_equal(captured["video_context_mask"], text.attention_mask).item()
        assert mx.array_equal(captured["audio_context_mask"], text.attention_mask).item()


def test_prepare_stage_can_use_reference_audio_latent_length() -> None:
    stage = distilled.prepare_stage(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=121,
            generate_audio=True,
            reference_aligned_audio=True,
        ),
        VideoPixelShape(batch=1, frames=121, height=64, width=64),
        dtype_policy=_Resources().dtype_policy,
    )

    assert stage.audio_tools is not None
    assert stage.audio_tools.target_shape.frames == 126


class _RecordingTextConditioner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, request, resources, **kwargs):
        self.calls.append({"request": request, "resources": resources, **kwargs})
        return _text(prompt=request.prompt)


def _patch_operations(monkeypatch) -> None:
    def decode_chunks(latent, decoder, **_kwargs):
        frames = (int(latent.shape[2]) - 1) * 8 + 1
        height = int(latent.shape[3]) * 32
        width = int(latent.shape[4]) * 32
        decoder.owner._use("video_decoder", "video iteration")
        yield mx.zeros((1, 3, 1, height, width), dtype=mx.float32)
        if decoder.owner.fail_at == "video iteration":
            raise RuntimeError("video iteration failed")
        if frames > 1:
            yield mx.zeros((1, 3, frames - 1, height, width), dtype=mx.float32)

    monkeypatch.setattr(decode_module, "decode_streaming", decode_chunks)
    monkeypatch.setattr(distilled, "release_stage_temporaries", lambda: None)


def _generate(
    monkeypatch,
    request: DistilledRequest,
    *,
    components: _RecordingComponents | None = None,
    resources: _Resources | None = None,
    text_conditioner=None,
    artifact_sink=None,
    reporter=None,
):
    _patch_operations(monkeypatch)
    components = components or _RecordingComponents()
    output = distilled.generate_distilled(
        request,
        resources or _Resources(),
        components=components,
        text_conditioner=text_conditioner or _RecordingTextConditioner(),
        artifact_sink=artifact_sink,
        reporter=reporter,
    )
    return output, components


def _profile_signature(
    profile: tuple[LoRAConfig, ...],
) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    return tuple((item.path.name, item.strength, item.exclude) for item in profile)


def test_public_recipe_runs_eight_plus_three_steps_and_closes_every_lease(monkeypatch) -> None:
    output, components = _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
    )

    assert output.frames.frame_count == 9
    assert (output.frames.spec.height, output.frames.spec.width) == (64, 64)
    assert output.audio_waveform is None
    assert output.audio_sample_rate is None
    assert len(components.transformers) == 1
    assert components.transformers[0].video_token_counts == [2] * 8 + [8] * 3
    assert components.close_counts == {
        "transformer": 1,
        "spatial_upscaler": 1,
    }
    assert "video_decoder" not in components.close_counts
    assert len(list(output.frames)) == 9
    assert components.close_counts["video_decoder"] == 1
    assert components.active == set()


def test_default_and_injected_providers_follow_the_same_station_graph(monkeypatch) -> None:
    request = DistilledRequest(prompt="test", width=64, height=64, frames=9)
    _injected_output, injected = _generate(monkeypatch, request)
    native = _RecordingComponents()
    monkeypatch.setattr(
        distilled,
        "NativeLTX2Components",
        lambda *, reporter: native,
    )
    _patch_operations(monkeypatch)

    default_output = distilled.generate_distilled(
        request,
        _Resources(),
        text_conditioner=_RecordingTextConditioner(),
    )

    assert default_output.frames.frame_count == 9
    assert [(action, name) for action, name, _active in native.events] == [
        (action, name) for action, name, _active in injected.events
    ]
    _injected_output.close()
    default_output.close()


def test_public_recipe_applies_cache_limit_before_prompt(monkeypatch) -> None:
    events = []
    resources = _Resources()
    resources.execution_policy.mlx_cache_limit_bytes = 1234
    _patch_operations(monkeypatch)
    monkeypatch.setattr(
        distilled.mx,
        "set_cache_limit",
        lambda value: events.append(("cache", value)),
    )

    def condition_text(request, _resources, **_kwargs):
        events.append(("prompt", None))
        return _text(prompt=request.prompt)

    distilled.generate_distilled(
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
        resources,
        components=_RecordingComponents(),
        text_conditioner=condition_text,
    )

    assert events == [("cache", 1234), ("prompt", None)]


def test_recipe_offers_materialized_artifacts_at_boundaries(monkeypatch) -> None:
    events = []

    class _Artifacts:
        def save(self, artifact):
            tensors = dict(artifact.tensors)
            metadata = dict(artifact.metadata)
            if artifact.name == TEXT_CONDITIONING:
                events.append(("text", metadata["prompt"], tuple(tensors["video_encoding"].shape)))
                return
            audio_latent = tensors.get("audio_latent")
            events.append(
                (
                    "latents",
                    int(metadata["stage"]),
                    metadata["final"] == "true",
                    tuple(tensors["video_latent"].shape),
                    None if audio_latent is None else tuple(audio_latent.shape),
                )
            )

    _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
        artifact_sink=_Artifacts(),
    )

    assert events == [
        ("text", "test", (1, 1, 4096)),
        ("latents", 1, False, (1, 128, 2, 1, 1), None),
        ("latents", 2, True, (1, 128, 2, 2, 2), None),
    ]


def test_recipe_forwards_saved_text_conditioning_to_the_public_station(
    tmp_path,
    monkeypatch,
) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    calls = []

    def condition_text(request, resources, **kwargs):
        calls.append((request, resources, kwargs))
        return _text(prompt="saved prompt", tokens=4)

    saved_prompts = []

    class _Artifacts:
        def save(self, artifact):
            if artifact.name == TEXT_CONDITIONING:
                saved_prompts.append(dict(artifact.metadata)["prompt"])

    _patch_operations(monkeypatch)
    resources = _Resources()
    distilled.generate_distilled(
        DistilledRequest(width=64, height=64, frames=9, text_conditioning=sidecar),
        resources,
        components=_RecordingComponents(),
        text_conditioner=condition_text,
        artifact_sink=_Artifacts(),
    )
    assert calls[0][0].text_conditioning == sidecar
    assert calls[0][1] is resources
    assert saved_prompts == ["saved prompt"]


def test_recipe_forwards_the_vae_tiling_policy(monkeypatch) -> None:
    received = []
    _patch_operations(monkeypatch)

    def decode_frames(_latent, _decoder_provider, *, spec, frame_count, **kwargs):
        received.append(kwargs["tiling_config"])
        return VideoFrameStream(
            lambda: iter(()),
            spec=spec,
            frame_count=frame_count,
        )

    monkeypatch.setattr(distilled, "decode_ltx23_sdr_frames", decode_frames)
    output = distilled.generate_distilled(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            vae_tiling=VideoVAETilingConfig(mode="single"),
        ),
        _Resources(),
        components=_RecordingComponents(),
        text_conditioner=_RecordingTextConditioner(),
    )
    assert len(received) == 1
    assert received[0] is not None
    assert received[0].temporal_config is None
    assert received[0].spatial_config is None
    output.close()


@pytest.mark.parametrize(
    ("hdr", "override", "expected"),
    [
        (None, "auto", mx.bfloat16),
        (None, "float32", mx.float32),
        ("ACESCG", "auto", mx.float32),
        ("ACESCG", "bfloat16", mx.bfloat16),
    ],
)
def test_terminal_resolves_recipe_aware_vae_decode_dtype(
    monkeypatch,
    hdr: str | None,
    override: str,
    expected: mx.Dtype,
) -> None:
    _patch_operations(monkeypatch)
    policy = LTX2DTypePolicy.reference(transformer=mx.bfloat16)
    base = _Resources(generation="2.5", dtype_policy=policy)
    resources = LTX2Resources(
        checkpoint=base.checkpoint,
        components=(),
        capabilities=base.capabilities,
        dtype_policy=policy,
        cache_policy=SimpleNamespace(),
        execution_policy=base.execution_policy,
        video_vae_config=LTX23_VIDEO_VAE_CONFIG,
    )
    components = _RecordingComponents()
    image = ImageConditioningConfig(Path("condition.exr")) if hdr is not None else None
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        hdr=hdr,  # type: ignore[arg-type]
        image=image,
        vae_decode_dtype=override,  # type: ignore[arg-type]
    )

    output = distilled.decode_stage_latents(
        request,
        resources,
        distilled.StageLatents(
            video=mx.zeros((1, 128, 2, 2, 2), dtype=mx.bfloat16),
        ),
        geometry=VideoPixelShape(batch=1, frames=9, height=64, width=64),
        components=components,
        reporter=RecordingReporter(),
    )
    assert len(list(output.frames)) == 9
    assert components.video_decoder_dtypes == [expected]
    assert output.metadata["dtype_policy"]["video_vae"] == str(expected).removeprefix("mlx.core.")


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
@pytest.mark.parametrize("frame_index", [0, 4])
def test_image_conditions_use_two_bounded_encoder_leases_and_stage_geometry(
    generation: str,
    frame_index: int,
    tmp_path,
    monkeypatch,
) -> None:
    image = tmp_path / "conditioning.png"
    image.touch()
    encoded_sizes = []

    def encode_image(_path, _encoder, *, width, height, compute_dtype, **_kwargs):
        encoded_sizes.append((width, height, compute_dtype))
        return mx.ones((1, 128, 1, height // 32, width // 32), dtype=compute_dtype)

    monkeypatch.setattr(condition_preparation, "encode_image", encode_image)
    policy = LTX2DTypePolicy.reference(transformer=mx.float16)
    saved_conditions = []

    class _Artifacts:
        def save(self, artifact):
            if artifact.name in {STAGE_1_CONDITIONING, STAGE_2_CONDITIONING}:
                saved_conditions.append(artifact)

    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            image=ImageConditioningConfig(path=image, frame_index=frame_index),
        ),
        resources=_Resources(generation=generation, dtype_policy=policy),
        artifact_sink=_Artifacts(),
    )

    assert output.frames.frame_count == 9
    assert encoded_sizes == [(32, 32, mx.bfloat16), (64, 64, mx.bfloat16)]
    assert [artifact.name for artifact in saved_conditions] == [
        STAGE_1_CONDITIONING,
        STAGE_2_CONDITIONING,
    ]
    assert [
        tuple(dict(artifact.tensors)["condition_0_latent"].shape) for artifact in saved_conditions
    ] == [
        (1, 128, 1, 1, 1),
        (1, 128, 1, 2, 2),
    ]
    assert [dict(artifact.metadata)["condition_0_family"] for artifact in saved_conditions] == [
        "image" if frame_index == 0 else "keyframe",
    ] * 2
    expected_tokens = [2] * 8 + [8] * 3 if frame_index == 0 else [3] * 8 + [12] * 3
    assert components.transformers[0].video_token_counts == expected_tokens
    video_events = [event[:2] for event in components.events if "video_encoder" in event[1]]
    assert video_events == [
        ("load", "video_encoder"),
        ("close", "video_encoder"),
        ("load", "video_encoder"),
        ("close", "video_encoder"),
    ]
    first_transformer_load = components.events.index(
        next(event for event in components.events if event[:2] == ("load", "transformer"))
    )
    first_encoder_close = components.events.index(
        next(event for event in components.events if event[:2] == ("close", "video_encoder"))
    )
    assert first_encoder_close < first_transformer_load
    encoder_close_indexes = [
        index
        for index, event in enumerate(components.events)
        if event[:2] == ("close", "video_encoder")
    ]
    transformer_use_indexes = [
        index
        for index, event in enumerate(components.events)
        if event[0] == "use" and event[1] in {"stage 1", "stage 2"}
    ]
    assert encoder_close_indexes[1] < transformer_use_indexes[8]
    assert components.active == set()
    output.close()


@pytest.mark.parametrize(
    ("reference_aligned_audio", "expected_policy", "expected_audio_tokens"),
    [
        (False, "coverage-ceil", 10),
        (True, "reference-round", 9),
    ],
)
def test_audio_decode_vocode_and_video_decode_are_sequential_leases(
    monkeypatch,
    reference_aligned_audio: bool,
    expected_policy: str,
    expected_audio_tokens: int,
) -> None:
    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            generate_audio=True,
            reference_aligned_audio=reference_aligned_audio,
        ),
    )

    assert output.audio_waveform is not None
    assert output.audio_waveform.shape[:2] == (1, 2)
    assert output.audio_sample_rate == 48_000
    assert output.metadata["audio_latent_length_policy"] == expected_policy
    assert output.metadata["audio_latent_shape"] == (1, 8, expected_audio_tokens, 16)
    lifecycle = [
        (action, name)
        for action, name, _active in components.events
        if action in {"load", "close"} and name in {"audio_decoder", "vocoder", "video_decoder"}
    ]
    assert lifecycle == [
        ("load", "audio_decoder"),
        ("close", "audio_decoder"),
        ("load", "vocoder"),
        ("close", "vocoder"),
    ]
    assert len(list(output.frames)) == 9
    lifecycle = [
        (action, name)
        for action, name, _active in components.events
        if action in {"load", "close"} and name in {"audio_decoder", "vocoder", "video_decoder"}
    ]
    assert lifecycle == [
        ("load", "audio_decoder"),
        ("close", "audio_decoder"),
        ("load", "vocoder"),
        ("close", "vocoder"),
        ("load", "video_decoder"),
        ("close", "video_decoder"),
    ]
    decoder_load = next(
        active
        for action, name, active in components.events
        if action == "load" and name == "video_decoder"
    )
    assert decoder_load == frozenset({"video_decoder"})


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (None, None, "no waveform"),
        (mx.zeros((2, 100)), 48_000, "shape"),
        (mx.zeros((1, 2, 100)), 0, "sample rate"),
    ],
)
def test_generation_output_rejects_invalid_audio_and_closes_frames(
    waveform,
    sample_rate,
    message: str,
) -> None:
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        generate_audio=True,
    )
    spec = distilled.ltx23_sdr_signal(width=64, height=64, fps=request.fps)
    frames = VideoFrameStream(
        lambda: iter(()),
        spec=spec,
        frame_count=request.frames,
    )

    with pytest.raises(RuntimeError, match=message):
        distilled.build_generation_output(
            request,
            _Resources(),
            frames=frames,
            waveform=waveform,
            sample_rate=sample_rate,
        )

    assert frames.closed


@pytest.mark.parametrize(
    ("name", "stage_1", "stage_2", "expected"),
    [
        ("identical nonempty", 0.5, 0.5, [(0.5,)]),
        ("different strengths", 0.25, 0.75, [(0.25,), (0.75,)]),
        ("stage 1 only", 0.5, 0.0, [(0.5,), ()]),
        ("stage 2 only", 0.0, 0.5, [(), (0.5,)]),
    ],
)
def test_complete_profile_truth_table_for_one_adapter(
    name,
    stage_1,
    stage_2,
    expected,
    tmp_path,
    monkeypatch,
) -> None:
    del name
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(adapter,),
            lora_stage1_strengths=(stage_1,),
            lora_stage2_strengths=(stage_2,),
        ),
    )

    assert [
        tuple(item.strength for item in profile) for profile in components.transformer_profiles
    ] == expected
    assert components.close_counts["transformer"] == len(expected)


def test_empty_profiles_reuse_one_pristine_transformer(monkeypatch) -> None:
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
    )
    assert components.transformer_profiles == [()]
    assert components.close_counts["transformer"] == 1


def test_equal_strengths_with_different_adapter_paths_reload_pristine(
    tmp_path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.touch()
    second.touch()
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(first, second),
            lora_stage1_strengths=(0.5, 0.0),
            lora_stage2_strengths=(0.0, 0.5),
        ),
    )
    assert [_profile_signature(profile) for profile in components.transformer_profiles] == [
        (("first.safetensors", 0.5, ()),),
        (("second.safetensors", 0.5, ()),),
    ]


def test_equal_path_and_strength_with_different_exclusions_reloads_pristine(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    stage_1 = (LoRAConfig(adapter, strength=0.5, exclude=("audio",)),)
    stage_2 = (LoRAConfig(adapter, strength=0.5, exclude=("video",)),)
    monkeypatch.setattr(distilled, "resolve_lora_profiles", lambda _request: (stage_1, stage_2))
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
    )
    assert [_profile_signature(profile) for profile in components.transformer_profiles] == [
        (("adapter.safetensors", 0.5, ("audio",)),),
        (("adapter.safetensors", 0.5, ("video",)),),
    ]


def test_distinct_profile_memory_receipt_has_no_transformer_at_upscaler_entry(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(adapter,),
            lora_stage1_strengths=(0.25,),
            lora_stage2_strengths=(0.75,),
        ),
    )
    receipt = components.memory_receipts[0]
    assert receipt.station == "before spatial upscaler load"
    assert receipt.active_bytes >= 0
    assert "transformer" not in receipt.active_components
    close_index = components.events.index(
        next(event for event in components.events if event[:2] == ("close", "transformer"))
    )
    upscale_index = components.events.index(
        next(event for event in components.events if event[:2] == ("load", "spatial_upscaler"))
    )
    assert close_index < upscale_index


def test_identical_profile_intentionally_keeps_transformer_through_upscale(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    _output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(adapter,),
            lora_strengths=(0.5,),
        ),
    )
    assert "transformer" in components.memory_receipts[0].active_components
    assert len(components.transformer_profiles) == 1


@pytest.mark.parametrize("failed_load", [0, 1])
def test_transformer_failure_closes_each_active_lease_once(
    failed_load,
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    components = _RecordingComponents(fail_transformer_loads=frozenset({failed_load}))
    _patch_operations(monkeypatch)
    with pytest.raises(RuntimeError, match=rf"transformer-{failed_load} failed"):
        distilled.generate_distilled(
            DistilledRequest(
                prompt="test",
                width=64,
                height=64,
                frames=9,
                lora_paths=(adapter,),
                lora_stage1_strengths=(0.25,),
                lora_stage2_strengths=(0.75,),
            ),
            _Resources(),
            components=components,
            text_conditioner=_RecordingTextConditioner(),
        )
    assert components.close_counts["transformer"] == failed_load + 1
    assert components.active == set()


@pytest.mark.parametrize(
    "station",
    [
        "prompt",
        "stage 1",
        "upscale",
        "stage 2",
        "audio decode",
        "vocoder",
        "video iteration",
    ],
)
def test_product_recipe_station_failure_closes_every_loaded_component_once(
    station: str,
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    components = _RecordingComponents(fail_at=station)
    _patch_operations(monkeypatch)

    def condition_text(request, _resources, **_kwargs):
        if station == "prompt":
            raise RuntimeError("prompt failed")
        return _text(prompt=request.prompt)

    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        generate_audio=True,
        lora_paths=(adapter,),
        lora_stage1_strengths=(0.25,),
        lora_stage2_strengths=(0.75,),
    )
    if station == "video iteration":
        output = distilled.generate_distilled(
            request,
            _Resources(),
            components=components,
            text_conditioner=condition_text,
        )
        assert next(output.frames).shape == (64, 64, 3)
        with pytest.raises(RuntimeError, match="video iteration failed"):
            next(output.frames)
    else:
        with pytest.raises(RuntimeError, match=rf"{station} failed"):
            distilled.generate_distilled(
                request,
                _Resources(),
                components=components,
                text_conditioner=condition_text,
            )

    loaded = [name for action, name, _active in components.events if action == "load"]
    for name in set(loaded):
        assert components.close_counts[name] == loaded.count(name)
    assert components.active == set()


def test_reporter_records_only_real_transformer_loads(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    reporter = RecordingReporter()
    _output, _components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(adapter,),
            lora_stage1_strengths=(0.25,),
            lora_stage2_strengths=(0.75,),
        ),
        reporter=reporter,
    )
    assert [
        (action, phase)
        for action, phase, _details in reporter.events
        if "load stage" in phase and "transformer" in phase
    ] == [
        ("start", "load stage 1 transformer"),
        ("end", "load stage 1 transformer"),
        ("start", "load stage 2 transformer"),
        ("end", "load stage 2 transformer"),
    ]


def test_unsupported_condition_fails_before_text_or_components(tmp_path, monkeypatch) -> None:
    image = tmp_path / "condition.png"
    image.touch()
    resources = _Resources()
    resources.capabilities.condition_families = ("text", "image")
    components = _RecordingComponents()

    def fail_text(*_args, **_kwargs):
        pytest.fail("capability failure must precede text loading")

    with pytest.raises(ValueError, match="keyframe conditioning"):
        distilled.generate_distilled(
            DistilledRequest(
                prompt="test",
                width=64,
                height=64,
                frames=9,
                image=ImageConditioningConfig(path=image, frame_index=4),
            ),
            resources,
            components=components,
            text_conditioner=fail_text,
        )
    assert components.events == []


def test_resolved_profiles_use_canonical_file_identity_and_omit_zero(
    tmp_path,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    alias = tmp_path / "alias.safetensors"
    adapter.touch()
    alias.symlink_to(adapter)
    stage_1, stage_2 = distilled.resolve_lora_profiles(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(alias,),
            lora_stage1_strengths=(0.0,),
            lora_stage2_strengths=(0.5,),
        )
    )
    assert stage_1 == ()
    assert len(stage_2) == 1
    assert stage_2[0].path == adapter.resolve()
    assert stage_2[0].strength == 0.5


class _RecordingTransitionNoise:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str, tuple[int, ...], mx.Dtype]] = []

    @property
    def state(self) -> NoiseStreamState:
        return NoiseStreamState(
            backend="mlx",
            compatibility_profile="mlx-native",
            seed=10_042,
            draws=len(self.calls),
            elements=0,
            philox_blocks=0,
        )

    def __call__(
        self,
        *,
        stage: int,
        transition: int,
        modality: str,
        shape: tuple[int, ...],
        dtype: mx.Dtype,
    ) -> mx.array:
        self.calls.append((stage, transition, modality, shape, dtype))
        return mx.zeros(shape, dtype=dtype)


@pytest.mark.parametrize("generate_audio", [False, True])
def test_ltx25_recipe_uses_seed_plus_10000_only_for_stage1_ancestral_noise(
    generate_audio: bool,
    monkeypatch,
) -> None:
    seeds = []
    provider = _RecordingTransitionNoise()

    def noise_stream(seed: int, **_kwargs):
        seeds.append(seed)
        return provider

    monkeypatch.setattr(distilled, "SeededGaussianNoise", noise_stream)
    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            seed=42,
            generate_audio=generate_audio,
        ),
        resources=_Resources(generation="2.5"),
    )

    modalities = ("video", "audio") if generate_audio else ("video",)
    assert seeds == [10_042]
    assert [
        (stage, transition, modality)
        for stage, transition, modality, _shape, _dtype in provider.calls
    ] == [(1, transition, modality) for transition in range(7) for modality in modalities]
    assert output.metadata["model_generation"] == "2.5"
    assert output.metadata["sampler_policy"] == "ancestral-stage1-deterministic-stage2"
    assert output.metadata["ancestral_noise_seed"] == 10_042
    assert components.transformer_profiles == [()]
    assert len(list(output.frames)) == 9
    assert components.active == set()


@pytest.mark.parametrize(
    ("generation", "override", "policy", "stage_1", "ancestral_seed"),
    [
        ("2.5", "deterministic", "deterministic-euler-two-stage", "deterministic-euler", None),
        (
            "2.3",
            "ancestral",
            "ancestral-stage1-deterministic-stage2",
            "ancestral-rf",
            10_042,
        ),
    ],
)
def test_sampler_override_replaces_only_the_generation_default(
    generation: str,
    override: str,
    policy: str,
    stage_1: str,
    ancestral_seed: int | None,
) -> None:
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        seed=42,
        sampler=override,  # type: ignore[arg-type]
    )

    plan = distilled.resolve_distilled_sampler_plan(request, _Resources(generation=generation))

    assert plan.policy == policy
    assert plan.stage_1 == stage_1
    assert plan.stage_2 == "deterministic-euler"
    assert plan.ancestral_seed == ancestral_seed


def test_ltx25_deterministic_override_is_recorded_without_an_ancestral_seed(
    monkeypatch,
) -> None:
    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            seed=42,
            sampler="deterministic",
        ),
        resources=_Resources(generation="2.5"),
    )

    assert output.metadata["sampler_override"] == "deterministic"
    assert output.metadata["sampler_policy"] == "deterministic-euler-two-stage"
    assert output.metadata["ancestral_noise_seed"] is None
    assert len(list(output.frames)) == 9
    assert components.active == set()


def test_ltx25_saved_conditioning_and_audio_off_keep_the_public_station_graph(
    tmp_path,
    monkeypatch,
) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    provider = _RecordingTransitionNoise()
    conditioner = _RecordingTextConditioner()
    monkeypatch.setattr(
        distilled,
        "SeededGaussianNoise",
        lambda _seed, **_kwargs: provider,
    )

    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            width=64,
            height=64,
            frames=9,
            text_conditioning=sidecar,
            generate_audio=False,
        ),
        resources=_Resources(generation="2.5"),
        text_conditioner=conditioner,
    )

    assert conditioner.calls[0]["request"].text_conditioning == sidecar
    assert {modality for _stage, _transition, modality, _shape, _dtype in provider.calls} == {
        "video"
    }
    assert not any(
        name in {"audio_decoder", "vocoder"} for _action, name, _active in components.events
    )
    output.close()
    assert components.active == set()


def _assert_rejected_before_any_provider(
    request: DistilledRequest,
    message: str,
) -> None:
    components = _RecordingComponents()

    def fail_text(*_args, **_kwargs):
        raise AssertionError("request validation must precede text loading")

    with pytest.raises(ValueError, match=message):
        distilled.generate_distilled(
            request,
            _Resources(generation="2.5"),
            components=components,
            text_conditioner=fail_text,
        )
    assert components.events == []


def test_ltx25_auto_duration_runs_after_text_and_closes_before_transformer(
    monkeypatch,
) -> None:
    output, components = _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=None),
        resources=_Resources(generation="2.5"),
    )
    assert output.frames.frame_count == 17
    duration_load = next(
        index
        for index, event in enumerate(components.events)
        if event[0:2] == ("load", "duration_predictor")
    )
    duration_close = next(
        index
        for index, event in enumerate(components.events)
        if event[0:2] == ("close", "duration_predictor")
    )
    transformer_load = next(
        index
        for index, event in enumerate(components.events)
        if event[0:2] == ("load", "transformer")
    )
    assert duration_load < duration_close < transformer_load
    output.close()


def test_ltx25_generated_keyframe_slots_run_only_in_stage_one(monkeypatch) -> None:
    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            generated_keyframes=2,
        ),
        resources=_Resources(generation="2.5"),
    )
    counts = components.transformers[0].video_token_counts
    assert counts[:8] == [4] * 8
    assert counts[8:] == [8] * 3
    output.close()


def test_ltx25_impossible_generated_keyframes_reject_before_any_provider() -> None:
    _assert_rejected_before_any_provider(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            generated_keyframes=8,
        ),
        "9 frames provide only 7 interior keyframe slots",
    )


def test_ltx25_nonempty_lora_profile_reaches_transformer_provider(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    output, components = _generate(
        monkeypatch,
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            lora_paths=(adapter,),
        ),
        resources=_Resources(generation="2.5"),
    )
    assert len(components.transformer_profiles) == 1
    assert components.transformer_profiles[0][0].path == adapter.resolve()
    output.close()


def test_ltx25_ancestral_seed_overflow_rejects_before_any_provider_opens() -> None:
    _assert_rejected_before_any_provider(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            seed=2**64 - 10_000,
        ),
        r"seed \+ 10000.*unsigned 64-bit",
    )


def test_ltx25_cancellation_inside_ancestral_noise_closes_the_transformer_once(
    monkeypatch,
) -> None:
    components = _RecordingComponents()

    class _CancelNoise:
        def __call__(
            self,
            *,
            transition: int,
            shape: tuple[int, ...],
            dtype: mx.Dtype,
            **_kwargs,
        ):
            if transition == 2:
                raise RuntimeError("cancelled")
            return mx.zeros(shape, dtype=dtype)

    monkeypatch.setattr(
        distilled,
        "SeededGaussianNoise",
        lambda _seed, **_kwargs: _CancelNoise(),
    )
    _patch_operations(monkeypatch)
    with pytest.raises(RuntimeError, match="cancelled"):
        distilled.generate_distilled(
            DistilledRequest(prompt="test", width=64, height=64, frames=9),
            _Resources(generation="2.5"),
            components=components,
            text_conditioner=_RecordingTextConditioner(),
        )

    assert components.close_counts == {"transformer": 1}
    assert components.active == set()


@pytest.mark.parametrize(
    "station",
    [
        "prompt",
        "stage 1",
        "upscale",
        "stage 2",
        "audio decode",
        "vocoder",
        "video iteration",
    ],
)
def test_ltx25_station_failure_closes_every_loaded_component_once(
    station: str,
    monkeypatch,
) -> None:
    components = _RecordingComponents(fail_at=station)
    _patch_operations(monkeypatch)

    def condition_text(request, _resources, **_kwargs):
        if station == "prompt":
            raise RuntimeError("prompt failed")
        return _text(prompt=request.prompt)

    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        generate_audio=True,
    )
    if station == "video iteration":
        output = distilled.generate_distilled(
            request,
            _Resources(generation="2.5"),
            components=components,
            text_conditioner=condition_text,
        )
        assert next(output.frames).shape == (64, 64, 3)
        with pytest.raises(RuntimeError, match="video iteration failed"):
            next(output.frames)
    else:
        with pytest.raises(RuntimeError, match=rf"{station} failed"):
            distilled.generate_distilled(
                request,
                _Resources(generation="2.5"),
                components=components,
                text_conditioner=condition_text,
            )

    loaded = [name for action, name, _active in components.events if action == "load"]
    for name in set(loaded):
        assert components.close_counts[name] == loaded.count(name)
    assert components.active == set()


def test_ltx23_recipe_never_constructs_the_ancestral_noise_stream(monkeypatch) -> None:
    monkeypatch.setattr(
        distilled,
        "SeededGaussianNoise",
        lambda _seed, **_kwargs: pytest.fail("LTX-2.3 must remain deterministic Euler"),
    )
    output, components = _generate(
        monkeypatch,
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
        resources=_Resources(generation="2.3"),
    )
    output.close()
    assert components.active == set()
