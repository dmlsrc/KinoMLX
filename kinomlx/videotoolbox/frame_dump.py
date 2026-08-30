"""Lossless diagnostic PNG sequence at the decoded-frame boundary."""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from kinomlx.io.atomic import write_text_atomic
from kinomlx.media.frames import CloseableVideoFrameStream

if TYPE_CHECKING:
    import mlx.core as mx

_log = logging.getLogger(__name__)


class FrameDumpError(RuntimeError):
    """A requested decoded-frame sequence could not be materialized."""


class PNGFrameDumpStream:
    """Tee one decoded frame stream into an 8-bit lossless PNG sequence.

    The source stream remains the object returned to the media pipeline. PNG
    writing therefore occurs before VSR, frame-rate conversion, and video
    encoding. The directory is retained only after :meth:`commit`; failures
    remove the directory created by this stream without touching any
    preexisting path.
    """

    def __init__(
        self,
        source: CloseableVideoFrameStream,
        directory: Path | str,
    ) -> None:
        self.spec = source.spec
        self.frame_count = source.frame_count
        self.directory = Path(directory)
        self._source = source
        self._iterator = iter(source)
        self._written = 0
        self._exhausted = False
        self._committed = False
        self._closed = False
        self._owns_resources = False
        self._digits = max(6, len(str(self.frame_count - 1)))
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise FrameDumpError(
                f"cannot create VAE frame directory {self.directory}: {exc}"
            ) from exc
        self._owns_resources = True

    @property
    def written(self) -> int:
        """Number of PNG frames successfully written."""
        return self._written

    def __iter__(self) -> PNGFrameDumpStream:
        return self

    def __next__(self) -> mx.array:
        if self._closed:
            raise StopIteration
        if self._exhausted:
            raise StopIteration
        try:
            frame = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            raise
        except BaseException:
            self.close()
            raise

        path = self.directory / f"frame_{self._written:0{self._digits}d}.png"
        try:
            import mlx.core as mx

            from kinomlx.videotoolbox.images import save_image

            pixels = (mx.clip(frame.astype(mx.float32), 0.0, 1.0) * 255.0 + 0.5).astype(mx.uint8)
            save_image(pixels, path)
        except Exception as exc:
            self.close()
            raise FrameDumpError(f"cannot write VAE frame {path}: {exc}") from exc
        self._written += 1
        return frame

    def commit(self) -> None:
        """Publish the complete sequence by retaining its private directory."""
        if self._closed:
            raise FrameDumpError("cannot commit a closed VAE frame dump")
        if self._written != self.frame_count:
            raise FrameDumpError(
                f"VAE frame dump wrote {self._written} of {self.frame_count} frames"
            )
        signal = self.spec
        manifest = {
            "schema_version": 1,
            "format": "png",
            "file_pattern": f"frame_%0{self._digits}d.png",
            "frame_count": self.frame_count,
            "stored_dtype": "uint8",
            "value_mapping": "round(clamp(source, 0, 1) * 255)",
            "source_signal": {
                "layout": signal.layout.value,
                "dtype": signal.dtype,
                "value_domain": signal.value_domain.value,
                "primaries": signal.primaries.value,
                "transfer": signal.transfer.value,
                "matrix": signal.matrix.value,
                "range": signal.range.value,
                "width": signal.width,
                "height": signal.height,
                "cadence": f"{signal.cadence.numerator}/{signal.cadence.denominator}",
            },
        }
        try:
            write_text_atomic(
                self.directory / "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        except OSError as exc:
            raise FrameDumpError(f"cannot finalize VAE frame dump {self.directory}: {exc}") from exc
        self._committed = True

    def close(self) -> None:
        """Close the source and remove an uncommitted sequence."""
        if self._closed:
            return
        self._closed = True
        if not self._owns_resources:
            return
        self._source.close()
        if self._committed:
            return
        try:
            shutil.rmtree(self.directory)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _log.warning(
                "Could not remove incomplete VAE frame dump %s: %s",
                self.directory,
                exc,
            )

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = ["FrameDumpError", "PNGFrameDumpStream"]
