"""Minimal unfiltered PNG fixtures independent of the product image writer."""

import struct
import zlib
from pathlib import Path


def write_png(path: Path, width: int, height: int, pixels: bytes, *, channels: int = 3) -> None:
    """Write explicit 8-bit gray, RGB, or straight-alpha RGBA samples."""
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    assert len(pixels) == width * height * channels

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    stride = width * channels
    rows = b"".join(
        b"\x00" + pixels[start : start + stride] for start in range(0, len(pixels), stride)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def solid_png(path: Path, width: int, height: int, color: tuple[int, ...]) -> None:
    write_png(path, width, height, bytes(color) * (width * height), channels=len(color))
