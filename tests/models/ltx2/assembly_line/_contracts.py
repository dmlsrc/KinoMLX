"""Pure contract harness for the assembly-line architecture.

The harness is independent of MLX and model weights. It verifies stage ordering
and ownership rules with injected providers and reusable trace assertions.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol

type ComponentName = Literal[
    "video_encoder",
    "transformer",
    "spatial_upscaler",
    "audio_decoder",
    "vocoder",
    "video_decoder",
]


@dataclass(frozen=True, order=True)
class AdapterSpec:
    """One canonical adapter entry in an ordered effective profile."""

    identity: str
    strength: float
    exclusions: tuple[str, ...] = ()


type LoraProfile = tuple[AdapterSpec, ...]
EMPTY_PROFILE: LoraProfile = ()


def effective_profile(*adapters: AdapterSpec) -> LoraProfile:
    """Return the ordered profile after zero-strength entries are omitted."""
    return tuple(adapter for adapter in adapters if adapter.strength != 0.0)


@dataclass(frozen=True)
class ContractResources:
    """Immutable, weight-free stand-in for the future prepared resource plan."""

    checkpoint: str = "synthetic-ltx-2.3"
    component_inventory: tuple[ComponentName, ...] = (
        "video_encoder",
        "transformer",
        "spatial_upscaler",
        "audio_decoder",
        "vocoder",
        "video_decoder",
    )


@dataclass(frozen=True)
class ComponentEvent:
    """One component event plus the active set immediately after the event."""

    action: Literal["load", "use", "close"]
    component: ComponentName
    stage: str
    active: frozenset[ComponentName]
    profile: LoraProfile = EMPTY_PROFILE


class InjectedStageFailure(RuntimeError):
    """Failure raised by the recording provider at a selected stage."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"injected failure at {stage}")
        self.stage = stage


class ComponentPort(Protocol):
    """Operation shared by the synthetic component ports."""

    def use(self, stage: str) -> None: ...


class LTX2VideoEncoderPort(ComponentPort, Protocol):
    pass


class LTX2TransformerPort(ComponentPort, Protocol):
    pass


class LTX2SpatialUpscalerPort(ComponentPort, Protocol):
    pass


class LTX2AudioDecoderPort(ComponentPort, Protocol):
    pass


class LTX2VocoderPort(ComponentPort, Protocol):
    pass


class LTX2VideoDecoderPort(ComponentPort, Protocol):
    pass


class LTX2VideoEncoderProvider(Protocol):
    def video_encoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[LTX2VideoEncoderPort]: ...


class LTX2TransformerProvider(Protocol):
    def transformer(
        self,
        resources: ContractResources,
        profile: LoraProfile,
    ) -> AbstractContextManager[LTX2TransformerPort]: ...


class LTX2SpatialUpscalerProvider(Protocol):
    def spatial_upscaler(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[LTX2SpatialUpscalerPort]: ...


class LTX2AudioDecoderProvider(Protocol):
    def audio_decoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[LTX2AudioDecoderPort]: ...


class LTX2VocoderProvider(Protocol):
    def vocoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[LTX2VocoderPort]: ...


class LTX2VideoDecoderProvider(Protocol):
    def video_decoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[LTX2VideoDecoderPort]: ...


class LTX2ComponentsProvider(
    LTX2VideoEncoderProvider,
    LTX2TransformerProvider,
    LTX2SpatialUpscalerProvider,
    LTX2AudioDecoderProvider,
    LTX2VocoderProvider,
    LTX2VideoDecoderProvider,
    Protocol,
):
    pass


class OneStageComponents(
    LTX2VideoEncoderProvider,
    LTX2TransformerProvider,
    LTX2VideoDecoderProvider,
    Protocol,
):
    pass


class ProgressReporter(Protocol):
    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
        aggregate_total: float | None = None,
    ) -> None: ...

    def phase_advance(self, phase: str, advance: float = 1.0) -> None: ...

    def phase_end(self, phase: str) -> None: ...


@dataclass(frozen=True)
class LTX2EncodedTextConditioning:
    """Pure synthetic prompt product with no component ownership."""

    value: str


class LTX2TextConditionerPort(Protocol):
    """Prompt stage callable, deliberately not a component provider."""

    def __call__(
        self,
        request: LTX2Request | OneStageRequest,
        resources: ContractResources,
        *,
        reporter: ProgressReporter | None = None,
    ) -> LTX2EncodedTextConditioning: ...


@dataclass
class RecordingTextConditioner:
    """Record prompt-stage calls and optionally fail before producing text."""

    fail: bool = False
    events: list[tuple[str, str]] = field(default_factory=list)

    def __call__(
        self,
        request: LTX2Request | OneStageRequest,
        resources: ContractResources,
        *,
        reporter: ProgressReporter | None = None,
    ) -> LTX2EncodedTextConditioning:
        del request, resources, reporter
        self.events.append(("start", "prompt"))
        if self.fail:
            self.events.append(("fail", "prompt"))
            raise InjectedStageFailure("prompt")
        product = LTX2EncodedTextConditioning("encoded prompt")
        self.events.append(("return", product.value))
        return product


class NullProgressReporter:
    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
        aggregate_total: float | None = None,
    ) -> None:
        pass

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        pass

    def phase_end(self, phase: str) -> None:
        pass


@dataclass
class RecordingProgressReporter:
    events: list[tuple[str, str]] = field(default_factory=list)

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
        aggregate_total: float | None = None,
    ) -> None:
        self.events.append(("start", phase))

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        self.events.append(("advance", phase))

    def phase_end(self, phase: str) -> None:
        self.events.append(("end", phase))


@contextmanager
def _reported(reporter: ProgressReporter, stage: str) -> Iterator[None]:
    reporter.phase_start(stage)
    try:
        yield
    finally:
        reporter.phase_end(stage)


class RecordingLease:
    """Idempotent component lease owned by one lexical stage scope."""

    def __init__(
        self,
        factory: RecordingComponents,
        component: ComponentName,
        *,
        load_stage: str,
        profile: LoraProfile = EMPTY_PROFILE,
    ) -> None:
        self._factory = factory
        self.component = component
        self.profile = profile
        self._load_stage = load_stage
        self._entered = False
        self._closed = False

    def __enter__(self) -> RecordingLease:
        if self._closed:
            raise RuntimeError(f"cannot enter closed {self.component} lease")
        if self._entered:
            raise RuntimeError(f"cannot enter {self.component} lease twice")
        self._factory._load(self.component, self._load_stage, self.profile)
        self._entered = True
        return self

    def use(self, stage: str) -> None:
        if not self._entered or self._closed:
            raise RuntimeError(f"cannot use inactive {self.component} lease")
        self._factory._use(self.component, stage, self.profile)

    def close(self, stage: str | None = None) -> None:
        if not self._entered or self._closed:
            return
        self._closed = True
        self._factory._close(
            self.component,
            stage if stage is not None else f"{self._load_stage} exit",
            self.profile,
        )

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass
class RecordingComponents:
    """Injected provider recording loads, uses, closes, and active sets."""

    fail_at: str | None = None
    events: list[ComponentEvent] = field(default_factory=list)
    _active: set[ComponentName] = field(default_factory=set, init=False)

    @property
    def active(self) -> frozenset[ComponentName]:
        return frozenset(self._active)

    def video_encoder(self, resources: ContractResources) -> RecordingLease:
        return self._lease("video_encoder", "condition preparation")

    def transformer(
        self,
        resources: ContractResources,
        profile: LoraProfile,
    ) -> RecordingLease:
        return self._lease("transformer", "denoise", profile)

    def spatial_upscaler(self, resources: ContractResources) -> RecordingLease:
        return self._lease("spatial_upscaler", "upscale")

    def audio_decoder(self, resources: ContractResources) -> RecordingLease:
        return self._lease("audio_decoder", "audio decode")

    def vocoder(self, resources: ContractResources) -> RecordingLease:
        return self._lease("vocoder", "vocoder")

    def video_decoder(self, resources: ContractResources) -> RecordingLease:
        return self._lease("video_decoder", "video iteration")

    def _lease(
        self,
        component: ComponentName,
        stage: str,
        profile: LoraProfile = EMPTY_PROFILE,
    ) -> RecordingLease:
        return RecordingLease(self, component, load_stage=stage, profile=profile)

    def _load(
        self,
        component: ComponentName,
        stage: str,
        profile: LoraProfile,
    ) -> None:
        if component in self._active:
            raise AssertionError(f"duplicate live {component} component")
        self._active.add(component)
        self._record("load", component, stage, profile)

    def _use(
        self,
        component: ComponentName,
        stage: str,
        profile: LoraProfile,
    ) -> None:
        self._record("use", component, stage, profile)
        if self.fail_at == stage:
            raise InjectedStageFailure(stage)

    def _close(
        self,
        component: ComponentName,
        stage: str,
        profile: LoraProfile,
    ) -> None:
        if component not in self._active:
            raise AssertionError(f"closing inactive {component} component")
        self._active.remove(component)
        self._record("close", component, stage, profile)

    def _record(
        self,
        action: Literal["load", "use", "close"],
        component: ComponentName,
        stage: str,
        profile: LoraProfile,
    ) -> None:
        self.events.append(
            ComponentEvent(
                action=action,
                component=component,
                stage=stage,
                active=self.active,
                profile=profile,
            )
        )


class NativeContractComponents(RecordingComponents):
    """Default-provider stand-in for static and call-graph parity checks."""


class DecodedFrameStream(Iterator[str]):
    """Closeable lazy stream that owns its video-decoder lease."""

    def __init__(
        self,
        resources: ContractResources,
        components: LTX2VideoDecoderProvider,
        *,
        frame_count: int = 3,
    ) -> None:
        self._resources = resources
        self._components = components
        self._frame_count = frame_count
        self._index = 0
        self._manager: AbstractContextManager[LTX2VideoDecoderPort] | None = None
        self._decoder: LTX2VideoDecoderPort | None = None
        self._closed = False

    def __iter__(self) -> DecodedFrameStream:
        return self

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration
        if self._index == self._frame_count:
            self.close()
            raise StopIteration
        if self._manager is None:
            self._manager = self._components.video_decoder(self._resources)
            self._decoder = self._manager.__enter__()
        if self._decoder is None:
            raise AssertionError("video decoder manager entered without a decoder")
        try:
            self._decoder.use("video iteration")
        except BaseException:
            self.close()
            raise
        frame = f"frame-{self._index}"
        self._index += 1
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        manager = self._manager
        self._manager = None
        self._decoder = None
        if manager is not None:
            manager.__exit__(None, None, None)


@dataclass
class ContractOutput:
    """Synthetic generation product owning any unconsumed frame stream."""

    frames: DecodedFrameStream
    audio_waveform: str | None
    text_conditioning: LTX2EncodedTextConditioning
    recording: RecordingComponents | None = None
    recording_text_conditioner: RecordingTextConditioner | None = None

    def close(self) -> None:
        self.frames.close()

    def __enter__(self) -> ContractOutput:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass(frozen=True)
class LTX2Request:
    stage_1_profile: LoraProfile = EMPTY_PROFILE
    stage_2_profile: LoraProfile = EMPTY_PROFILE
    image_conditioned: bool = True
    generate_audio: bool = True
    frame_count: int = 3


def run_distilled_contract(
    request: LTX2Request,
    resources: ContractResources,
    *,
    components: LTX2ComponentsProvider | None = None,
    text_conditioner: LTX2TextConditionerPort | None = None,
    reporter: ProgressReporter | None = None,
) -> ContractOutput:
    """Execute the intended distilled ownership schedule with injected ports."""
    selected = components if components is not None else NativeContractComponents()
    selected_text = text_conditioner if text_conditioner is not None else RecordingTextConditioner()
    progress = reporter if reporter is not None else NullProgressReporter()

    with _reported(progress, "prompt"):
        text = selected_text(request, resources, reporter=progress)

    if request.image_conditioned:
        with (
            _reported(progress, "stage 1 condition"),
            selected.video_encoder(resources) as video_encoder,
        ):
            video_encoder.use("stage 1 condition")

    transformer_manager = selected.transformer(resources, request.stage_1_profile)
    stage_1_transformer = transformer_manager.__enter__()
    transformer: LTX2TransformerPort | None = stage_1_transformer
    try:
        with _reported(progress, "stage 1"):
            stage_1_transformer.use("stage 1")

        if request.stage_1_profile != request.stage_2_profile:
            transformer_manager.__exit__(None, None, None)
            transformer = None

        with _reported(progress, "upscale"), selected.spatial_upscaler(resources) as upscaler:
            upscaler.use("upscale")

        if request.image_conditioned:
            with (
                _reported(progress, "stage 2 condition"),
                selected.video_encoder(resources) as video_encoder,
            ):
                video_encoder.use("stage 2 condition")

        if transformer is None:
            transformer_manager = selected.transformer(resources, request.stage_2_profile)
            transformer = transformer_manager.__enter__()
        with _reported(progress, "stage 2"):
            transformer.use("stage 2")
    finally:
        if transformer is not None:
            transformer_manager.__exit__(None, None, None)

    waveform = None
    if request.generate_audio:
        with (
            _reported(progress, "audio decode"),
            selected.audio_decoder(resources) as audio_decoder,
        ):
            audio_decoder.use("audio decode")
        with _reported(progress, "vocoder"), selected.vocoder(resources) as vocoder:
            vocoder.use("vocoder")
        waveform = "waveform"

    recording = selected if isinstance(selected, RecordingComponents) else None
    return ContractOutput(
        frames=DecodedFrameStream(
            resources,
            selected,
            frame_count=request.frame_count,
        ),
        audio_waveform=waveform,
        text_conditioning=text,
        recording=recording,
        recording_text_conditioner=(
            selected_text if isinstance(selected_text, RecordingTextConditioner) else None
        ),
    )


@dataclass(frozen=True)
class RawVideoCondition:
    """Caller-owned media intent with no encoder or model ownership."""

    kind: Literal["image", "keyframe", "video", "mask"]
    source_id: str


@dataclass(frozen=True)
class OneStageRequest:
    condition: RawVideoCondition
    profile: LoraProfile = EMPTY_PROFILE
    frame_count: int = 3


def run_one_stage_contract(
    request: OneStageRequest,
    resources: ContractResources,
    *,
    components: OneStageComponents | None = None,
    text_conditioner: LTX2TextConditionerPort | None = None,
    reporter: ProgressReporter | None = None,
) -> ContractOutput:
    """Compose a conditioned one-stage recipe without the distilled runner."""
    selected = components if components is not None else NativeContractComponents()
    selected_text = text_conditioner if text_conditioner is not None else RecordingTextConditioner()
    progress = reporter if reporter is not None else NullProgressReporter()

    with _reported(progress, "prompt"):
        text = selected_text(request, resources, reporter=progress)
    with (
        _reported(progress, "condition preparation"),
        selected.video_encoder(resources) as video_encoder,
    ):
        video_encoder.use("condition preparation")
    with (
        _reported(progress, "one-stage denoise"),
        selected.transformer(resources, request.profile) as transformer,
    ):
        transformer.use("one-stage denoise")

    recording = selected if isinstance(selected, RecordingComponents) else None
    return ContractOutput(
        frames=DecodedFrameStream(
            resources,
            selected,
            frame_count=request.frame_count,
        ),
        audio_waveform=None,
        text_conditioning=text,
        recording=recording,
        recording_text_conditioner=(
            selected_text if isinstance(selected_text, RecordingTextConditioner) else None
        ),
    )


def event_signature(event: ComponentEvent) -> tuple[str, ComponentName, str, LoraProfile]:
    """Return the stable event fields used for call-graph comparisons."""
    return event.action, event.component, event.stage, event.profile


def assert_balanced(factory: RecordingComponents) -> None:
    """Assert every loaded component was closed exactly once."""
    loads: dict[ComponentName, int] = {}
    closes: dict[ComponentName, int] = {}
    for event in factory.events:
        if event.action == "load":
            loads[event.component] = loads.get(event.component, 0) + 1
        elif event.action == "close":
            closes[event.component] = closes.get(event.component, 0) + 1
    assert factory.active == frozenset()
    assert closes == loads
