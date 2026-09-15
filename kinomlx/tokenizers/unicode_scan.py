"""First-party scanner equivalent to the pinned Qwen Unicode split regex."""

import unicodedata

QWEN_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
    r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_ASCII_UNICODE_WHITESPACE = "\t\n\x0b\x0c\r\x85"
_UNICODE_SEPARATOR_CATEGORIES = frozenset(("Zs", "Zl", "Zp"))


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_newline(character: str) -> bool:
    return character in "\r\n"


def _is_whitespace(character: str) -> bool:
    """Match oniguruma's Unicode-aware ``\\s`` without Python bidi extras."""
    return (
        character in _ASCII_UNICODE_WHITESPACE
        or unicodedata.category(character) in _UNICODE_SEPARATOR_CATEGORIES
    )


def split_qwen_unicode(text: str) -> tuple[str, ...]:
    """Split normalized text according to the selected Qwen tokenizer contract."""
    pieces: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        contraction = (
            next(
                (
                    item
                    for item in _CONTRACTIONS
                    if text[index : index + len(item)].casefold() == item
                ),
                None,
            )
            if character == "'"
            else None
        )
        if contraction is not None:
            pieces.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue

        if _is_letter(character):
            end = index + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if (
            not _is_newline(character)
            and not _is_letter(character)
            and not _is_number(character)
            and index + 1 < len(text)
            and _is_letter(text[index + 1])
        ):
            end = index + 2
            while end < len(text) and _is_letter(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if _is_number(character):
            pieces.append(character)
            index += 1
            continue

        punctuation_start = index
        punctuation_index = index
        if character == " " and index + 1 < len(text):
            next_character = text[index + 1]
            if (
                not _is_whitespace(next_character)
                and not _is_letter(next_character)
                and not _is_number(next_character)
            ):
                punctuation_index += 1
        if punctuation_index < len(text):
            candidate = text[punctuation_index]
            if (
                not _is_whitespace(candidate)
                and not _is_letter(candidate)
                and not _is_number(candidate)
            ):
                end = punctuation_index + 1
                while end < len(text):
                    candidate = text[end]
                    if _is_whitespace(candidate) or _is_letter(candidate) or _is_number(candidate):
                        break
                    end += 1
                while end < len(text) and _is_newline(text[end]):
                    end += 1
                pieces.append(text[punctuation_start:end])
                index = end
                continue

        if _is_whitespace(character):
            end = index + 1
            while end < len(text) and _is_whitespace(text[end]):
                end += 1
            last_newline = max(
                (position for position in range(index, end) if _is_newline(text[position])),
                default=-1,
            )
            if last_newline >= index:
                pieces.append(text[index : last_newline + 1])
                index = last_newline + 1
                continue
            if end < len(text) and end - index > 1:
                pieces.append(text[index : end - 1])
                index = end - 1
                continue
            pieces.append(text[index:end])
            index = end
            continue
        raise ValueError(f"Qwen tokenizer scanner could not classify character at index {index}")
    return tuple(pieces)


__all__ = ["QWEN_SPLIT_PATTERN", "split_qwen_unicode"]
