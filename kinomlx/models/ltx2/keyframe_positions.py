"""Pure generated-keyframe position policy shared by admission and execution."""


def generated_keyframe_indices(pixel_frames: int, count: int) -> tuple[int, ...]:
    """Choose evenly spaced, rounded interior pixel-frame indices."""
    if isinstance(pixel_frames, bool) or not isinstance(pixel_frames, int) or pixel_frames < 1:
        raise ValueError("pixel_frames must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("generated keyframe count must be a non-negative integer")
    if count == 0:
        return ()
    available = pixel_frames - 2
    if available < count:
        raise ValueError(
            f"{pixel_frames} frames provide only {max(0, available)} interior keyframe slots"
        )
    denominator = count + 1
    indices = tuple(
        round(index * (pixel_frames - 1) / denominator) for index in range(1, count + 1)
    )
    if (
        any(index <= 0 or index >= pixel_frames - 1 for index in indices)
        or len(set(indices)) != count
    ):
        raise ValueError("generated keyframe request does not map to unique interior frames")
    return indices
