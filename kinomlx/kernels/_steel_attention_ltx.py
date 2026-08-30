"""BF16 MLX STEEL attention resources for LTX-2.3 attention."""

from pathlib import Path

_RESOURCE_DIR = Path(__file__).with_name("metal")
__all__ = ["HEADER", "SOURCE"]


def _read_metal(name: str) -> str:
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


HEADER = _read_metal("steel_attention_ltx_header.metal")
SOURCE = _read_metal("steel_attention_ltx_body.metal")
