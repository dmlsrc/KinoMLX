"""Minimal Gemma-style tokenizer JSON shared by tokenizer and conditioning tests."""

import json


def tokenizer_document(*, prepends_bos: bool = True) -> dict[str, object]:
    pieces = ["<pad>", "<eos>", "<bos>", "<unk>"]
    pieces.extend(f"<0x{value:02X}>" for value in range(256))
    pieces.extend(["a", "b", "ab", "\u2581", "<unused0>", "<|video|>"])
    added_tokens = [
        {"id": index, "content": pieces[index], "special": True}
        for index in (0, 1, 2, 3, len(pieces) - 2, len(pieces) - 1)
    ]
    post_processor: dict[str, object] | None = None
    if prepends_bos:
        post_processor = {
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ]
        }
    return {
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "byte_fallback": True,
            "vocab": {piece: index for index, piece in enumerate(pieces)},
            "merges": [["a", "b"]],
        },
        "added_tokens": added_tokens,
        "post_processor": post_processor,
    }


def tokenizer_json(*, prepends_bos: bool = True) -> bytes:
    return json.dumps(tokenizer_document(prepends_bos=prepends_bos), sort_keys=True).encode("utf-8")


__all__ = ["tokenizer_document", "tokenizer_json"]
