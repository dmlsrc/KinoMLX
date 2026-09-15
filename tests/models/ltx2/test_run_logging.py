"""Human-readable LTX-2 run logging helpers."""

import logging
from pathlib import Path

import pytest

from kinomlx.models.ltx2.pipelines.distilled import sampling_facts
from kinomlx.models.ltx2.run_logging import (
    _gemma_variant_label,
    _log_sampling,
    log_effective_sampling,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.samplers.noise import noise_compatibility_profile

_DECLARED = "deterministic-euler-two-stage"


def test_gemma_variant_label_names_the_discovered_gemma3_variant() -> None:
    assert _gemma_variant_label("gemma3-12b-it", LTX2Settings()) == "qat"
    assert _gemma_variant_label("gemma3-12b-it", LTX2Settings(gemma_variant="plain")) == "plain"


def test_gemma_variant_label_reports_explicit_paths_and_skips_other_encoders(
    tmp_path: Path,
) -> None:
    explicit = LTX2Settings(gemma_path=tmp_path / "gemma")
    assert _gemma_variant_label("gemma3-12b-it", explicit) == (
        f"explicit gemma_path ({tmp_path / 'gemma'})"
    )
    assert _gemma_variant_label("gemma4-12b-it", LTX2Settings()) is None
    assert _gemma_variant_label("gemma3-12b-it", None) is None


def test_sampling_facts_resolve_the_announced_policy() -> None:
    facts = sampling_facts(DistilledRequest(prompt="p"), _DECLARED)
    assert facts["sampler_policy"] == _DECLARED
    assert facts["sampler_override"] is None
    assert facts["noise_compatibility_profile"] == noise_compatibility_profile(
        DistilledRequest(prompt="p").noise_backend
    )
    ancestral = sampling_facts(DistilledRequest(prompt="p", sampler="ancestral"), _DECLARED)
    assert ancestral["sampler_policy"] == "ancestral-stage1-deterministic-stage2"
    assert ancestral["sampler_override"] == "ancestral"


def test_effective_sampling_only_warns_when_the_run_diverged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("kinomlx.test.effective_sampling")
    request = DistilledRequest(prompt="p")
    facts = sampling_facts(request, _DECLARED)
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_effective_sampling(logger, facts, request=request, declared_policy=_DECLARED)
    assert caplog.records == []

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_effective_sampling(
            logger,
            {**facts, "sampler_policy": "other"},
            request=request,
            declared_policy=_DECLARED,
        )
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "Sampling changed at run time" in caplog.records[0].getMessage()


def test_sampling_block_announces_request_facts_and_the_declared_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("kinomlx.test.sampling_block")
    request = DistilledRequest(prompt="p", sampler="ancestral")
    with caplog.at_level(logging.INFO, logger=logger.name):
        _log_sampling(logger, request, None)
        _log_sampling(logger, request, _DECLARED)
        _log_sampling(logger, None, _DECLARED)
    text = "\n".join(record.getMessage() for record in caplog.records)
    blocks = text.split("Sampling:")
    assert len(blocks) == 4
    assert "Stage 1 steps: 8" in blocks[1]
    assert "Stage 2 steps: 3" in blocks[1]
    assert "Requested sampler: ancestral" in blocks[1]
    assert "Policy:" not in blocks[1]
    assert "Policy: ancestral-stage1-deterministic-stage2" in blocks[2]
    assert "Override: ancestral" in blocks[2]
    assert "Noise compatibility:" in blocks[2]
    assert blocks[3].strip() == f"Policy: {_DECLARED}"


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
        request=DistilledRequest(prompt="example", generate_audio=audio, fps=24),
        output=OutputConfig(target_fps=target_fps),
    )
    logger = logging.getLogger("kinomlx.test.delivery")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(logger, invocation)
    assert "Distilled (generation auto)" in caplog.text
    assert ("inactive (audio disabled)" in caplog.text) == (not audio)
    assert ("inactive (no frame-rate conversion)" in caplog.text) == (not active)


def test_adapter_lines_name_what_each_stage_applies(caplog):
    from types import SimpleNamespace

    from kinomlx.models.ltx2 import LTX2AdapterConfig
    from kinomlx.models.ltx2.config import OutputConfig
    from kinomlx.models.ltx2.run_logging import log_request

    invocation = SimpleNamespace(
        model="ltx2",
        model_settings=LTX2Settings(),
        restart=None,
        request=DistilledRequest(
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
