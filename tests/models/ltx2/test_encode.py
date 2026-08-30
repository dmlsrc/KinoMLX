"""Prompt and conditioning-input boundary tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.encode as encode
from kinomlx.media.hdr import scene_linear_to_acescct
from kinomlx.models.ltx2.types import HDRAuthoring
from kinomlx.reporting import RecordingReporter


def _conditioning_arrays() -> dict[str, mx.array]:
    return {
        "video_encoding": mx.zeros((1, 8, 4096)),
        "audio_encoding": mx.zeros((1, 8, 2048)),
        "attention_mask": mx.ones((1, 8)),
    }


def test_load_text_conditioning_accepts_the_saved_sidecar_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            _conditioning_arrays(),
            {
                "schema_version": "2",
                "artifact": "ltx2_text_conditioning",
                "prompt": "original prompt",
            },
        ),
    )
    reporter = RecordingReporter()
    output, metadata = encode.load_text_conditioning(
        "conditioning.safetensors",
        reporter=reporter,
    )
    assert output.video_encoding.shape == (1, 8, 4096)
    assert output.audio_encoding.shape == (1, 8, 2048)
    assert output.attention_mask.shape == (1, 8)
    assert metadata["prompt"] == "original prompt"
    assert reporter.events[0][1] == "load text conditioning"
    assert reporter.events[-1][0] == "end"


def test_structural_text_replay_ignores_metadata_and_unconsumed_baggage(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    arrays = _conditioning_arrays()
    arrays["community_notes"] = mx.ones((3,), dtype=mx.float32)
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            arrays,
            {"schema_version": "community", "artifact": "renamed"},
        ),
    )

    with caplog.at_level("WARNING"):
        output, metadata = encode.load_text_conditioning(
            "conditioning.safetensors",
            metadata_policy="observe",
        )

    assert output.video_encoding.shape == (1, 8, 4096)
    assert metadata["artifact"] == "renamed"
    assert "ignoring 1 unconsumed" in caplog.text
    assert "advisory text-conditioning metadata differs" in caplog.text


def test_load_text_conditioning_rejects_wrong_schema_and_geometry(monkeypatch) -> None:
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            _conditioning_arrays(),
            {"schema_version": "0", "artifact": "ltx2_text_conditioning"},
        ),
    )
    with pytest.raises(ValueError, match="unsupported text-conditioning schema"):
        encode.load_text_conditioning("conditioning.safetensors")

    arrays = _conditioning_arrays()
    arrays["attention_mask"] = mx.ones((1, 7))
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            arrays,
            {"schema_version": "2", "artifact": "ltx2_text_conditioning"},
        ),
    )
    with pytest.raises(ValueError, match="token counts do not match"):
        encode.load_text_conditioning("conditioning.safetensors")


@pytest.mark.parametrize(
    ("tensor", "replacement", "message"),
    [
        ("video_encoding", mx.zeros((1, 8, 4096), dtype=mx.int32), "floating dtype"),
        (
            "audio_encoding",
            mx.full((1, 8, 2048), float("nan"), dtype=mx.float32),
            "finite values",
        ),
        ("attention_mask", mx.full((1, 8), 2.0), "binary"),
        (
            "attention_mask",
            mx.full((1, 8), float("nan"), dtype=mx.float32),
            "finite values",
        ),
    ],
)
def test_load_text_conditioning_rejects_invalid_tensor_values(
    monkeypatch,
    tensor: str,
    replacement: mx.array,
    message: str,
) -> None:
    arrays = _conditioning_arrays()
    arrays[tensor] = replacement
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            arrays,
            {"schema_version": "2", "artifact": "ltx2_text_conditioning"},
        ),
    )

    with pytest.raises(ValueError, match=rf"{tensor}.*{message}"):
        encode.load_text_conditioning("conditioning.safetensors")


def test_load_text_conditioning_rejects_zero_tokens(monkeypatch) -> None:
    arrays = {
        "video_encoding": mx.zeros((1, 0, 4096)),
        "audio_encoding": mx.zeros((1, 0, 2048)),
        "attention_mask": mx.ones((1, 0)),
    }
    monkeypatch.setattr(
        encode,
        "load_weights_with_metadata",
        lambda _path: (
            arrays,
            {"schema_version": "2", "artifact": "ltx2_text_conditioning"},
        ),
    )

    with pytest.raises(ValueError, match="positive token count"):
        encode.load_text_conditioning("conditioning.safetensors")


@pytest.mark.parametrize("authoring", ["SRGB_LINEAR", "ACESCG", "ACESCCT"])
def test_encode_image_converts_explicit_exr_signal_to_acescct_working_codes(
    monkeypatch: pytest.MonkeyPatch,
    authoring: HDRAuthoring,
) -> None:
    if authoring == "ACESCCT":
        source = mx.array([[[0.1, 0.5, 0.9]]], dtype=mx.float32)
        expected = source
    else:
        source = mx.array([[[0.18, 1.0, 4.0]]], dtype=mx.float32)
        if authoring == "SRGB_LINEAR":
            from kinomlx.media.hdr import convert_scene_linear_primaries
            from kinomlx.media.signals import ColorPrimaries

            linear = convert_scene_linear_primaries(
                source,
                source=ColorPrimaries.REC709,
                target=ColorPrimaries.ACESCG,
            )
        else:
            linear = source
        expected = scene_linear_to_acescct(linear)
    monkeypatch.setattr(encode, "load_raw_exr", lambda *_args, **_kwargs: source)
    seen: list[mx.array] = []

    def video_encoder(video: mx.array, *, reporter=None) -> mx.array:
        del reporter
        seen.append(video)
        return mx.zeros((1, 128, 1, 1, 1), dtype=mx.float32)

    latent = encode.encode_image(
        "condition.exr",
        video_encoder,
        width=1,
        height=1,
        compute_dtype=mx.float32,
        hdr_authoring=authoring,
    )

    assert tuple(latent.shape) == (1, 128, 1, 1, 1)
    actual_codes = (seen[0][0, :, 0, 0, 0] + 1.0) / 2.0
    assert mx.allclose(actual_codes, expected[0, 0], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (mx.array([[[float("nan"), 0.5, 0.5]]]), "finite"),
        (mx.array([[[-0.1, 0.5, 0.5]]]), r"in \[0, 1\]"),
        (mx.array([[[0.5, 0.5, 1.1]]]), r"in \[0, 1\]"),
    ],
)
def test_encode_image_rejects_invalid_acescct_exr_codes(
    monkeypatch: pytest.MonkeyPatch,
    source: mx.array,
    message: str,
) -> None:
    monkeypatch.setattr(encode, "load_raw_exr", lambda *_args, **_kwargs: source)
    with pytest.raises(ValueError, match=message):
        encode.encode_image(
            "condition.exr",
            lambda *_args, **_kwargs: pytest.fail("invalid EXR must not reach the VAE"),
            width=1,
            height=1,
            compute_dtype=mx.float32,
            hdr_authoring="ACESCCT",
        )


def test_encode_image_requires_exr_interpretation_and_rejects_it_for_sdr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        encode,
        "load_raw_exr",
        lambda *_args, **_kwargs: mx.zeros((1, 1, 3), dtype=mx.float32),
    )
    with pytest.raises(ValueError, match="explicit HDR signal interpretation"):
        encode.encode_image(
            "condition.exr",
            lambda *_args, **_kwargs: None,
            width=1,
            height=1,
            compute_dtype=mx.float32,
        )
    with pytest.raises(ValueError, match="applies only to EXR"):
        encode.encode_image(
            "condition.png",
            lambda *_args, **_kwargs: None,
            width=1,
            height=1,
            compute_dtype=mx.float32,
            hdr_authoring="ACESCG",
        )


def _stub_video_encoder(video: mx.array, *, reporter=None) -> mx.array:
    del video, reporter
    return mx.zeros((1, 128, 1, 1, 1), dtype=mx.float32)


@pytest.mark.parametrize(
    ("authoring", "source"),
    [
        ("SRGB_LINEAR", mx.array([[[0.18, 0.5, 1.0]]], dtype=mx.float32)),
        ("ACESCG", mx.array([[[0.18, 0.5, 0.999]]], dtype=mx.float32)),
        ("ACESCCT", mx.array([[[0.1, 0.3, 0.55]]], dtype=mx.float32)),
    ],
)
def test_encode_image_warns_when_the_exr_condition_carries_no_hdr_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    authoring: HDRAuthoring,
    source: mx.array,
) -> None:
    # Native HDR preserves the condition's exposure distribution; a plate
    # with nothing above SDR reference white yields effectively SDR output.
    monkeypatch.setattr(encode, "load_raw_exr", lambda *_args, **_kwargs: source)
    with caplog.at_level("WARNING", logger="kinomlx.models.ltx2.encode"):
        encode.encode_image(
            "condition.exr",
            _stub_video_encoder,
            width=1,
            height=1,
            compute_dtype=mx.float32,
            hdr_authoring=authoring,
        )
    assert any("effectively SDR" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("authoring", "source"),
    [
        ("SRGB_LINEAR", mx.array([[[0.18, 1.0, 4.0]]], dtype=mx.float32)),
        ("ACESCG", mx.array([[[0.18, 1.0, 3.2]]], dtype=mx.float32)),
        ("ACESCCT", mx.array([[[0.1, 0.5, 0.7]]], dtype=mx.float32)),
    ],
)
def test_encode_image_stays_quiet_for_genuine_hdr_exr_conditions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    authoring: HDRAuthoring,
    source: mx.array,
) -> None:
    monkeypatch.setattr(encode, "load_raw_exr", lambda *_args, **_kwargs: source)
    with caplog.at_level("WARNING", logger="kinomlx.models.ltx2.encode"):
        encode.encode_image(
            "condition.exr",
            _stub_video_encoder,
            width=1,
            height=1,
            compute_dtype=mx.float32,
            hdr_authoring=authoring,
        )
    assert not [record for record in caplog.records if "effectively SDR" in record.message]
