"""Closeable, single-consumer frame streams."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .signals import VideoSignalSpec

if TYPE_CHECKING:
    import mlx.core as mx


@runtime_checkable
class CloseableVideoFrameStream(Protocol):
    """An owned stream whose producer resources survive until close."""

    spec: VideoSignalSpec
    frame_count: int

    def __iter__(self) -> Iterator[mx.array]: ...

    def close(self) -> None: ...


class VideoFrameStream(Iterator["mx.array"]):
    """Lazily open a producer and validate its public frame contract.

    The factory is discarded on early close, so resources captured for a
    decoder that has not opened yet can be reclaimed without entering it.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[mx.array]],
        *,
        spec: VideoSignalSpec,
        frame_count: int,
        receipts: dict[str, object] | None = None,
    ) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.spec = spec
        self.frame_count = frame_count
        self._factory: Callable[[], Iterator[mx.array]] | None = factory
        self._iterator: Iterator[mx.array] | None = None
        self._receipts = receipts if receipts is not None else {}
        self._consumed = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def receipts(self) -> Mapping[str, object]:
        return MappingProxyType(self._receipts)

    def __iter__(self) -> VideoFrameStream:
        return self

    def __next__(self) -> mx.array:
        if self._closed:
            raise StopIteration
        if self._iterator is None:
            factory = self._factory
            if factory is None:
                self.close()
                raise StopIteration
            self._factory = None
            try:
                self._iterator = iter(factory())
            except BaseException:
                self.close()
                raise

        try:
            frame = next(self._iterator)
        except StopIteration:
            actual = self._consumed
            self.close()
            if actual != self.frame_count:
                raise RuntimeError(
                    f"decoded frame count {actual} does not match expected {self.frame_count}"
                ) from None
            raise
        except BaseException:
            self.close()
            raise

        if self._consumed >= self.frame_count:
            self.close()
            raise RuntimeError(f"decoded frame stream produced more than {self.frame_count} frames")
        self._validate_frame(frame)
        self._consumed += 1
        return frame

    def _validate_frame(self, frame: mx.array) -> None:
        shape = tuple(getattr(frame, "shape", ()))
        expected = (self.spec.height, self.spec.width, 3)
        if shape != expected:
            self.close()
            raise RuntimeError(f"decoded frame shape {shape} does not match {expected}")
        dtype = str(getattr(frame, "dtype", "")).split(".")[-1]
        if dtype != self.spec.dtype:
            self.close()
            raise RuntimeError(f"decoded frame dtype {dtype!r} does not match {self.spec.dtype!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._factory = None
        iterator = self._iterator
        self._iterator = None
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> VideoFrameStream:
        if self._closed:
            raise RuntimeError("cannot enter a closed video frame stream")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = ["CloseableVideoFrameStream", "VideoFrameStream"]
