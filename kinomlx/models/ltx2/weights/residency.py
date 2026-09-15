"""Selection of persistent block tensors independently of ring residency."""

from fnmatch import fnmatchcase


def retains_tensor(key: str, shape: tuple[int, ...], selectors: tuple[str, ...]) -> bool:
    """Match classes or block-relative logical tensor glob patterns."""
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "transformer_blocks":
        return False
    suffix = parts[2]
    family = suffix.split(".", 1)[0]
    for selector in selectors:
        if selector == "small":
            if len(shape) <= 1 or "scale_shift_table" in suffix:
                return True
        elif selector == "audio-attention":
            if "attn" in family and "audio" in family:
                return True
        elif selector == "video-attention":
            if family in ("attn1", "attn2"):
                return True
        elif selector == "audio-ff":
            if family == "audio_ff":
                return True
        elif selector == "video-ff":
            if family == "ff":
                return True
        elif fnmatchcase(suffix, selector):
            return True
    return False
