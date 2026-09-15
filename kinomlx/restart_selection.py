"""Select one image of a multi-image source run for a saved-station restart.

A run that produced several images records one artifact path per image and keys
each digest by output index. Every image model resolves that selection the same
way through this module: count the source images, map a one-based selection onto
the recorded lists, and key the matching fingerprint.
"""

from collections.abc import Mapping, Sequence

ARTIFACT_NAMES = ("images", "initial_noise", "final_latents", "text_conditioning")
PLANNED_IMAGE_COUNT = "planned_image_count"
SEED_OUTPUTS = "seed_outputs"

# A run record's own status words. A canceled run stopped between images and
# published everything before that point, so its finished images are as
# restartable as a completed run's. A failed run stopped for a reason it does
# not understand, so nothing it left behind is trusted.
COMPLETED_RUN_STATUS = "completed"
CANCELED_RUN_STATUS = "aborted"
RESTARTABLE_RUN_STATUSES = frozenset({COMPLETED_RUN_STATUS, CANCELED_RUN_STATUS})


def unpublished_image(index: int, count: int) -> str:
    """Name one image a canceled run planned but stopped before publishing."""
    return f"image {index + 1} of {count} was not published before cancellation"


def source_image_count(outputs: Mapping[str, object], *, planned: object = None) -> int:
    """Count the images a source run set out to write.

    Every image writer keys its per-image digests by the planned count, so a run
    that recorded that count is read through it. A run canceled after its first
    image then still resolves the digests it did write.

    A record without the planned count is read through whichever artifacts it
    kept. Records that omit images, and older records that kept one combined
    tensor product, both read as a single image, so only a genuinely per-image
    record selects an index.
    """
    if type(planned) is int and planned >= 1:
        return planned
    count = 1
    for name in ARTIFACT_NAMES:
        value = outputs.get(name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            count = max(count, len(value))
    return count


def resolve_restart_index(index: int | None, count: int) -> int:
    """Map a one-based selection onto a zero-based output index."""
    if count < 1:
        raise ValueError("a restart source run must record at least one image")
    if index is None:
        if count == 1:
            return 0
        raise ValueError(
            f"this run recorded {count} images; select one with a restart index "
            f"between 1 and {count}"
        )
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("restart index must be an integer")
    if not 1 <= index <= count:
        raise ValueError(f"restart index must be between 1 and {count}; this run has {count}")
    return index - 1


def selected_artifact(value: object, *, index: int, label: str) -> object:
    """Take one image's entry from an artifact recorded once per image.

    A single-image run records a bare path, so it is returned unchanged. A
    multi-image run records one path per image and is indexed.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = list(value)
        if not all(isinstance(item, str) for item in entries):
            raise ValueError(f"{label} must hold path strings")
        if index >= len(entries):
            raise ValueError(f"{label} has {len(entries)} entries; output {index + 1} is missing")
        return entries[index]
    raise ValueError(f"{label} must be a path string or an array of path strings")


def fingerprint_key(name: str, *, index: int, count: int) -> str:
    """Name the digest a run record uses for one artifact of one image."""
    return name if count == 1 else f"{name}.{index}"


__all__ = [
    "CANCELED_RUN_STATUS",
    "COMPLETED_RUN_STATUS",
    "PLANNED_IMAGE_COUNT",
    "RESTARTABLE_RUN_STATUSES",
    "SEED_OUTPUTS",
    "fingerprint_key",
    "resolve_restart_index",
    "selected_artifact",
    "source_image_count",
    "unpublished_image",
]
