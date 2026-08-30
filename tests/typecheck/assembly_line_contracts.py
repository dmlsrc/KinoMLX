"""Focused static lane for public assembly-line structural contracts.

The native component provider, recipe, signal, and delivery surfaces are
checked without widening this into a whole-repository strict-typing migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import mlx.core as mx

from kinomlx.components import ComponentLease
from kinomlx.lora.loading import LoRAProfile
from kinomlx.media.signals import (
    BT709_SDR_DELIVERY,
    OutputColorPlan,
    validate_sdr_output_plan,
)
from kinomlx.models.ltx2.cache import LoRAAdapterReceipt
from kinomlx.models.ltx2.components import (
    AudioDecoderPort,
    DurationPredictorPort,
    LatentStatisticsPort,
    NativeLTX2Components,
    SpatialUpscalerPort,
    TransformerPort,
    VideoDecoderPort,
    VideoEncoderPort,
    VocoderPort,
)
from kinomlx.models.ltx2.components import (
    DistilledComponents as ProductDistilledComponents,
)
from kinomlx.models.ltx2.conditioning import EncodedCondition
from kinomlx.models.ltx2.conditioning.tools import VideoLatentTools
from kinomlx.models.ltx2.pipelines.distilled import (
    DistilledRequest as ProductDistilledRequest,
)
from kinomlx.models.ltx2.pipelines.distilled import generate_distilled
from kinomlx.models.ltx2.resources import LTX2Resources
from kinomlx.models.ltx2.runner import Recipe
from kinomlx.models.ltx2.signals import LTX23_SDR_SIGNAL
from kinomlx.output import ArtifactSet, Generation, GenerationSink, VideoToolboxGenerationSink
from kinomlx.reporting import Reporter
from kinomlx.types import LatentState
from tests.models.ltx2.assembly_line._contracts import (
    ContractOutput,
    ContractResources,
    DistilledComponents,
    DistilledRequest,
    NativeContractComponents,
    OneStageComponents,
    OneStageRequest,
    ProgressReporter,
    RecordingComponents,
    RecordingProgressReporter,
    RecordingTextConditioner,
    TextConditioner,
    run_distilled_contract,
    run_one_stage_contract,
)


class DistilledRecipe(Protocol):
    def __call__(
        self,
        request: DistilledRequest,
        resources: ContractResources,
        *,
        components: DistilledComponents | None = None,
        text_conditioner: TextConditioner | None = None,
        reporter: ProgressReporter | None = None,
    ) -> ContractOutput: ...


class OneStageRecipe(Protocol):
    def __call__(
        self,
        request: OneStageRequest,
        resources: ContractResources,
        *,
        components: OneStageComponents | None = None,
        text_conditioner: TextConditioner | None = None,
        reporter: ProgressReporter | None = None,
    ) -> ContractOutput: ...


class SyntheticEncodedCondition:
    """Test double for the existing public encoded-condition protocol."""

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        return latent_state


class SyntheticStatistics:
    def normalize(self, latent: mx.array) -> mx.array:
        return latent

    def denormalize(self, latent: mx.array) -> mx.array:
        return latent


class SyntheticTransformer:
    lora_receipts: tuple[LoRAAdapterReceipt, ...] = ()

    def __call__(
        self,
        video: object,
        audio: object | None = None,
    ) -> tuple[mx.array | None, mx.array | None]:
        del video, audio
        raise NotImplementedError


class SyntheticVideoEncoder:
    per_channel_statistics: LatentStatisticsPort = SyntheticStatistics()

    def __call__(self, video: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        del reporter
        return video


class SyntheticSpatialUpscaler:
    per_channel_statistics: LatentStatisticsPort = SyntheticStatistics()

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        del reporter
        return latent


class SyntheticAudioDecoder:
    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        del reporter
        return latent


class SyntheticDurationPredictor:
    def predict_num_frames(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        del (
            video_tokens,
            audio_tokens,
            frame_rate,
            temporal_compression_ratio,
            min_seconds,
            max_seconds,
        )
        return 121


class SyntheticVocoder:
    output_sample_rate = 48_000

    def __call__(self, mel: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        del reporter
        return mel


class SyntheticVideoDecoder:
    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        causal: bool | None = None,
        reporter: Reporter | None = None,
    ) -> mx.array:
        del timestep, causal, reporter
        return latent


class SyntheticProductComponents:
    def transformer(
        self,
        resources: LTX2Resources,
        profile: LoRAProfile = (),
    ) -> ComponentLease[TransformerPort]:
        del resources, profile
        return ComponentLease(SyntheticTransformer())

    def video_encoder(self, resources: LTX2Resources) -> ComponentLease[VideoEncoderPort]:
        del resources
        return ComponentLease(SyntheticVideoEncoder())

    def spatial_upscaler(self, resources: LTX2Resources) -> ComponentLease[SpatialUpscalerPort]:
        del resources
        return ComponentLease(SyntheticSpatialUpscaler())

    def duration_predictor(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[DurationPredictorPort]:
        del resources
        return ComponentLease(SyntheticDurationPredictor())

    def audio_decoder(self, resources: LTX2Resources) -> ComponentLease[AudioDecoderPort]:
        del resources
        return ComponentLease(SyntheticAudioDecoder())

    def vocoder(self, resources: LTX2Resources) -> ComponentLease[VocoderPort]:
        del resources
        return ComponentLease(SyntheticVocoder())

    def video_decoder(self, resources: LTX2Resources) -> ComponentLease[VideoDecoderPort]:
        del resources
        return ComponentLease(SyntheticVideoDecoder())


class SyntheticGenerationSink:
    def write(self, generation: Generation, plan: OutputColorPlan) -> ArtifactSet:
        del generation, plan
        return ArtifactSet(video=Path("synthetic.mp4"))


recording_components: DistilledComponents = RecordingComponents()
native_components: DistilledComponents = NativeContractComponents()
product_native_components: ProductDistilledComponents = NativeLTX2Components()
product_test_components: ProductDistilledComponents = SyntheticProductComponents()
product_distilled_recipe: Recipe[ProductDistilledRequest] = generate_distilled
one_stage_components: OneStageComponents = RecordingComponents()
reporter: ProgressReporter = RecordingProgressReporter()
text_conditioner: TextConditioner = RecordingTextConditioner()
product_reporter: Reporter = RecordingProgressReporter()
encoded_condition: EncodedCondition = SyntheticEncodedCondition()
generation_sink: GenerationSink = SyntheticGenerationSink()
native_generation_sink: GenerationSink = VideoToolboxGenerationSink(
    path=Path("synthetic.mp4"),
    fps=24.0,
)
distilled_recipe: DistilledRecipe = run_distilled_contract
one_stage_recipe: OneStageRecipe = run_one_stage_contract


def external_example(
    request: DistilledRequest,
    resources: ContractResources,
) -> tuple[str, ...]:
    """Type-check the ordinary external recipe -> sink composition."""
    with distilled_recipe(
        request,
        resources,
        components=native_components,
        reporter=reporter,
    ) as output:
        plan = OutputColorPlan(
            source=LTX23_SDR_SIGNAL,
            deliveries=(BT709_SDR_DELIVERY,),
        )
        validate_sdr_output_plan(plan)
        return tuple(output.frames)
