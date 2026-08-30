"""Pure contract harness for the assembly-line architecture.

The harness is deliberately independent of MLX and model weights. Phase A uses
it to make the desired station ordering and ownership rules executable before
the product exposes component providers. Later phases can run the same trace
assertions against the public recipes and native providers.
"""

from __future__ import annotations

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
    station: str
    active: frozenset[ComponentName]
    profile: LoraProfile = EMPTY_PROFILE


class InjectedStationFailure(RuntimeError):
    """Failure raised by the recording provider at a selected station."""

    def __init__(self, station: str) -> None:
        super().__init__(f"injected failure at {station}")
        self.station = station


class ComponentPort(Protocol):
    """Operation shared by the synthetic component ports."""

    def use(self, station: str) -> None: ...


class VideoEncoderPort(ComponentPort, Protocol):
    pass


class TransformerPort(ComponentPort, Protocol):
    pass


class SpatialUpscalerPort(ComponentPort, Protocol):
    pass


class AudioDecoderPort(ComponentPort, Protocol):
    pass


class VocoderPort(ComponentPort, Protocol):
    pass


class VideoDecoderPort(ComponentPort, Protocol):
    pass


class VideoEncoderProvider(Protocol):
    def video_encoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[VideoEncoderPort]: ...


class TransformerProvider(Protocol):
    def transformer(
        self,
        resources: ContractResources,
        profile: LoraProfile,
    ) -> AbstractContextManager[TransformerPort]: ...


class SpatialUpscalerProvider(Protocol):
    def spatial_upscaler(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[SpatialUpscalerPort]: ...


class AudioDecoderProvider(Protocol):
    def audio_decoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[AudioDecoderPort]: ...


class VocoderProvider(Protocol):
    def vocoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[VocoderPort]: ...


class VideoDecoderProvider(Protocol):
    def video_decoder(
        self,
        resources: ContractResources,
    ) -> AbstractContextManager[VideoDecoderPort]: ...


class DistilledComponents(
    VideoEncoderProvider,
    TransformerProvider,
    SpatialUpscalerProvider,
    AudioDecoderProvider,
    VocoderProvider,
    VideoDecoderProvider,
    Protocol,
):
    pass


class OneStageComponents(
    VideoEncoderProvider,
    TransformerProvider,
    VideoDecoderProvider,
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
    ) -> None: ...

    def phase_advance(self, phase: str, advance: float = 1.0) -> None: ...

    def phase_end(self, phase: str) -> None: ...


@dataclass(frozen=True)
class EncodedTextConditioning:
    """Pure synthetic prompt product with no component ownership."""

    value: str


class TextConditioner(Protocol):
    """Prompt station callable, deliberately not a component provider."""

    def __call__(
        self,
        request: DistilledRequest | OneStageRequest,
        resources: ContractResources,
        *,
        reporter: ProgressReporter | None = None,
    ) -> EncodedTextConditioning: ...


@dataclass
class RecordingTextConditioner:
    """Record prompt-station calls and optionally fail before producing text."""

    fail: bool = False
    events: list[tuple[str, str]] = field(default_factory=list)

    def __call__(
        self,
        request: DistilledRequest | OneStageRequest,
        resources: ContractResources,
        *,
        reporter: ProgressReporter | None = None,
    ) -> EncodedTextConditioning:
        del request, resources, reporter
        self.events.append(("start", "prompt"))
        if self.fail:
            self.events.append(("fail", "prompt"))
            raise InjectedStationFailure("prompt")
        product = EncodedTextConditioning("encoded prompt")
        self.events.append(("return", product.value))
        return product


class NullProgressReporter:
    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
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
    ) -> None:
        self.events.append(("start", phase))

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        self.events.append(("advance", phase))

    def phase_end(self, phase: str) -> None:
        self.events.append(("end", phase))


@contextmanager
def _reported(reporter: ProgressReporter, station: str) -> Iterator[None]:
    reporter.phase_start(station)
    try:
        yield
    finally:
        reporter.phase_end(station)


class RecordingLease:
    """Idempotent component lease owned by one lexical station scope."""

    def __init__(
        self,
        factory: RecordingComponents,
        component: ComponentName,
        *,
        load_station: str,
        profile: LoraProfile = EMPTY_PROFILE,
    ) -> None:
        self._factory = factory
        self.component = component
        self.profile = profile
        self._load_station = load_station
        self._entered = False
        self._closed = False

    def __enter__(self) -> RecordingLease:
        if self._closed:
            raise RuntimeError(f"cannot enter closed {self.component} lease")
        if self._entered:
            raise RuntimeError(f"cannot enter {self.component} lease twice")
        self._factory._load(self.component, self._load_station, self.profile)
        self._entered = True
        return self

    def use(self, station: str) -> None:
        if not self._entered or self._closed:
            raise RuntimeError(f"cannot use inactive {self.component} lease")
        self._factory._use(self.component, station, self.profile)

    def close(self, station: str | None = None) -> None:
        if not self._entered or self._closed:
            return
        self._closed = True
        self._factory._close(
            self.component,
            station if station is not None else f"{self._load_station} exit",
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
        station: str,
        profile: LoraProfile = EMPTY_PROFILE,
    ) -> RecordingLease:
        return RecordingLease(self, component, load_station=station, profile=profile)

    def _load(
        self,
        component: ComponentName,
        station: str,
        profile: LoraProfile,
    ) -> None:
        if component in self._active:
            raise AssertionError(f"duplicate live {component} component")
        self._active.add(component)
        self._record("load", component, station, profile)

    def _use(
        self,
        component: ComponentName,
        station: str,
        profile: LoraProfile,
    ) -> None:
        self._record("use", component, station, profile)
        if self.fail_at == station:
            raise InjectedStationFailure(station)

    def _close(
        self,
        component: ComponentName,
        station: str,
        profile: LoraProfile,
    ) -> None:
        if component not in self._active:
            raise AssertionError(f"closing inactive {component} component")
        self._active.remove(component)
        self._record("close", component, station, profile)

    def _record(
        self,
        action: Literal["load", "use", "close"],
        component: ComponentName,
        station: str,
        profile: LoraProfile,
    ) -> None:
        self.events.append(
            ComponentEvent(
                action=action,
                component=component,
                station=station,
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
        components: VideoDecoderProvider,
        *,
        frame_count: int = 3,
    ) -> None:
        self._resources = resources
        self._components = components
        self._frame_count = frame_count
        self._index = 0
        self._manager: AbstractContextManager[VideoDecoderPort] | None = None
        self._decoder: VideoDecoderPort | None = None
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
    text_conditioning: EncodedTextConditioning
    recording: RecordingComponents | None = None
    recording_text_conditioner: RecordingTextConditioner | None = None

    def close(self) -> None:
        self.frames.close()

    def __enter__(self) -> ContractOutput:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass(frozen=True)
class DistilledRequest:
    stage_1_profile: LoraProfile = EMPTY_PROFILE
    stage_2_profile: LoraProfile = EMPTY_PROFILE
    image_conditioned: bool = True
    generate_audio: bool = True
    frame_count: int = 3


def run_distilled_contract(
    request: DistilledRequest,
    resources: ContractResources,
    *,
    components: DistilledComponents | None = None,
    text_conditioner: TextConditioner | None = None,
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
    transformer: TransformerPort | None = stage_1_transformer
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
    text_conditioner: TextConditioner | None = None,
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
    return event.action, event.component, event.station, event.profile


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
