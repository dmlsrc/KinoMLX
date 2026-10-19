"""Human-readable LTX-2 run logging helpers."""

import logging
from pathlib import Path

import pytest

from kinomlx.models.ltx2.pipelines.distilled import sampling_facts, stage_sampler_choices
from kinomlx.models.ltx2.run_logging import (
    _gemma_variant_label,
    _log_sampling,
    log_effective_sampling,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import LTX2Request
from kinomlx.models.ltx2.vocabularies import LTX2StageSampler
from kinomlx.samplers.noise import noise_compatibility_profile

_LTX23 = (LTX2StageSampler.DETERMINISTIC, LTX2StageSampler.DETERMINISTIC)
_LTX25 = (LTX2StageSampler.ANCESTRAL, LTX2StageSampler.ANCESTRAL)


def test_gemma_variant_label_names_the_discovered_gemma3_variant() -> None:
    assert _gemma_variant_label("gemma3-12b-it", LTX2Settings()) == "qat"
    assert _gemma_variant_label("gemma3-12b-it", LTX2Settings(gemma_variant="plain")) == "plain"


def test_gemma_variant_label_reports_explicit_paths_and_skips_other_encoders(
    tmp_path: Path,
) -> None:
    explicit = LTX2Settings(text_encoder_path=tmp_path / "gemma")
    assert _gemma_variant_label("gemma3-12b-it", explicit) == (
        f"explicit text_encoder_path ({tmp_path / 'gemma'})"
    )
    assert _gemma_variant_label("gemma4-12b-it", LTX2Settings()) is None
    assert _gemma_variant_label("gemma3-12b-it", None) is None


def test_sampling_facts_resolve_each_stage_against_the_published_samplers() -> None:
    facts = sampling_facts(LTX2Request(prompt="p"), _LTX25)
    assert facts["samplers"] == {
        "stage_1": {"requested": "auto", "effective": "ancestral"},
        "stage_2": {"requested": "auto", "effective": "ancestral"},
    }
    assert facts["noise_compatibility_profile"] == noise_compatibility_profile(
        LTX2Request(prompt="p").noise_backend
    )
    mixed = sampling_facts(LTX2Request(prompt="p", stage_2_sampler="ancestral"), _LTX23)
    assert mixed["samplers"] == {
        "stage_1": {"requested": "auto", "effective": "deterministic"},
        "stage_2": {"requested": "ancestral", "effective": "ancestral"},
    }


def test_effective_sampling_only_warns_when_the_run_diverged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("kinomlx.test.effective_sampling")
    request = LTX2Request(prompt="p")
    facts = sampling_facts(request, _LTX25)
    recorded = {
        **facts,
        "samplers": {
            stage: {**choice, "noise_seed": seed, "noise_state": None}
            for (stage, choice), seed in zip(
                stage_sampler_choices(request, _LTX25).items(), (10_042, 20_042), strict=True
            )
        },
    }
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_effective_sampling(logger, recorded, request=request, published=_LTX25)
    assert caplog.records == []

    diverged = {
        **recorded,
        "samplers": {
            **recorded["samplers"],
            "stage_2": {**recorded["samplers"]["stage_2"], "effective": "deterministic"},
        },
    }
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_effective_sampling(logger, diverged, request=request, published=_LTX25)
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    message = caplog.records[0].getMessage()
    assert "Sampling changed at run time" in message
    assert "Stage 2 sampler: announced 'ancestral (auto)', recorded 'deterministic (auto)'" in (
        message
    )
    assert "Stage 1 sampler" not in message


def test_sampling_block_announces_each_stage_sampler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("kinomlx.test.sampling_block")
    request = LTX2Request(prompt="p", stage_1_sampler="ancestral")
    with caplog.at_level(logging.INFO, logger=logger.name):
        _log_sampling(logger, request, None)
        _log_sampling(logger, request, _LTX23)
        _log_sampling(logger, None, _LTX25)
    text = "\n".join(record.getMessage() for record in caplog.records)
    blocks = text.split("Sampling:")
    assert len(blocks) == 4
    assert "Stage 1 steps: 8" in blocks[1]
    assert "Stage 2 steps: 3" in blocks[1]
    assert "Stage 1 sampler: ancestral" in blocks[1]
    assert "Stage 2 sampler: auto" in blocks[1]
    assert "Stage 1 sampler: ancestral (requested)" in blocks[2]
    assert "Stage 2 sampler: deterministic (auto)" in blocks[2]
    assert "Noise compatibility:" in blocks[2]
    assert "Stage 1 sampler: ancestral" in blocks[3]
    assert "Stage 2 sampler: ancestral" in blocks[3]


@pytest.mark.parametrize(
    ("audio", "target_fps", "active"),
    [(False, None, False), (True, 48.0, True), (False, 24.0, False)],
)
def test_delivery_logs_effective_controls(caplog, audio, target_fps, active):
    from types import SimpleNamespace

    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request

    invocation = SimpleNamespace(
        model="ltx2",
        model_settings=LTX2Settings(),
        restart=None,
        request=LTX2Request(prompt="example", generate_audio=audio, fps=24),
        output=OutputConfig(target_fps=target_fps),
    )
    logger = logging.getLogger("kinomlx.test.delivery")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation)
    assert "Distilled (generation auto)" in caplog.text
    assert ("inactive (audio disabled)" in caplog.text) == (not audio)
    assert ("inactive (no frame-rate conversion)" in caplog.text) == (not active)


def test_conditioning_lines_name_every_image_in_author_order(caplog):
    from types import SimpleNamespace

    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request
    from kinomlx.models.ltx2.types import LTX2ImageConditioningConfig

    def invocation(images):
        return SimpleNamespace(
            model="ltx2",
            model_settings=LTX2Settings(),
            restart=None,
            request=LTX2Request(prompt="example", frames=17, images=images),
            output=OutputConfig(),
        )

    logger = logging.getLogger("kinomlx.test.conditioning_lines")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(
            logger,
            invocation(
                (
                    LTX2ImageConditioningConfig(Path("last.png"), frame_index=16, strength=0.8),
                    LTX2ImageConditioningConfig(Path("first.png")),
                )
            ),
        )
    text = caplog.text
    assert "Input image 1: last.png; frame=16; strength=0.8" in text
    assert "Input image 2: first.png; frame=first (0); strength=1" in text
    assert text.index("Input image 1") < text.index("Input image 2")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation((LTX2ImageConditioningConfig(Path("only.png")),)))
    assert "Input image: only.png; frame=first (0); strength=1" in caplog.text
    assert "Input image 1" not in caplog.text


@pytest.mark.parametrize("mode", ["anchor", "guide"])
def test_the_conditioning_section_always_names_the_frame_zero_mode(mode, caplog):
    from types import SimpleNamespace

    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request

    invocation = SimpleNamespace(
        model="ltx2",
        model_settings=LTX2Settings(),
        restart=None,
        request=LTX2Request(prompt="example", frames=17, first_frame_mode=mode),
        output=OutputConfig(),
    )
    logger = logging.getLogger(f"kinomlx.test.first_frame_mode_{mode}")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation)
    text = caplog.text
    assert f"First frame mode: {mode}" in text
    assert text.index("Generated keyframes") < text.index("First frame mode")


def test_an_image_line_names_both_stages_only_when_they_differ(caplog):
    from types import SimpleNamespace

    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request
    from kinomlx.models.ltx2.types import LTX2ImageConditioningConfig

    invocation = SimpleNamespace(
        model="ltx2",
        model_settings=LTX2Settings(),
        restart=None,
        request=LTX2Request(
            prompt="example",
            frames=17,
            images=(
                LTX2ImageConditioningConfig(
                    Path("split.png"), stage1_strength=0.9, stage2_strength=0.5
                ),
                LTX2ImageConditioningConfig(
                    Path("even.png"), frame_index=16, strength=0.8, stage2_strength=0.8
                ),
            ),
        ),
        output=OutputConfig(),
    )
    logger = logging.getLogger("kinomlx.test.image_stage_lines")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation)
    text = caplog.text
    assert (
        "Input image 1: split.png; frame=first (0); strength=0.9 (stage 1) / 0.5 (stage 2)" in text
    )
    # Both stages resolve to 0.8, so the line stays a single number.
    assert "Input image 2: even.png; frame=16; strength=0.8" in text
    assert "0.8 (stage 1)" not in text


def test_adapter_lines_name_what_each_stage_applies(caplog):
    from types import SimpleNamespace

    from kinomlx.models.ltx2 import LTX2AdapterConfig
    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request

    invocation = SimpleNamespace(
        model="ltx2",
        model_settings=LTX2Settings(),
        restart=None,
        request=LTX2Request(
            prompt="example",
            adapters=(
                LTX2AdapterConfig(Path("style.safetensors"), exclude=("audio",)),
                LTX2AdapterConfig(
                    Path("distill.safetensors"), stage1_strength=0.25, stage2_strength=0.5
                ),
                LTX2AdapterConfig(Path("motion.safetensors"), strength=0.8, stage2_strength=0),
            ),
        ),
        output=OutputConfig(),
    )
    logger = logging.getLogger("kinomlx.test.adapter_lines")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation)
    assert "Adapter 1: style.safetensors; strength=1; exclude=audio" in caplog.text
    assert (
        "Adapter 2: distill.safetensors; stage 1 strength=0.25; stage 2 strength=0.5; exclude=none"
        in caplog.text
    )
    assert (
        "Adapter 3: motion.safetensors; stage 1 strength=0.8; stage 2 strength=0; exclude=none"
        in caplog.text
    )
    # The common scalar is never shown beside stage values it does not feed.
    assert "; strength=1; stage" not in caplog.text
    assert "; strength=0.8;" not in caplog.text
