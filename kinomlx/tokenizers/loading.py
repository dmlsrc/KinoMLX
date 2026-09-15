"""Construct native Qwen tokenizer tables directly from source assets."""

from pathlib import Path

from kinomlx._json import load_json_object

from .byte_bpe import ByteBPETables


def load_tokenizer_tables(
    tokenizer_json: Path | str, tokenizer_config: Path | str
) -> ByteBPETables:
    """Validate source JSON and derive runtime tables without persistence."""
    return ByteBPETables.from_source(
        load_json_object(tokenizer_json), load_json_object(tokenizer_config)
    )
