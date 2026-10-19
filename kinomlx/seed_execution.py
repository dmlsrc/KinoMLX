"""Request-scoped model ownership and sequential execution at image stages."""

from collections.abc import Callable, Generator, Iterator, Sequence
from typing import cast

from kinomlx.components import ComponentLease
from kinomlx.seeds import SeedMode


class SeedModels:
    """Own loaded models, lending short leases to existing single-seed stages."""

    def __init__(self) -> None:
        self._leases: dict[str, ComponentLease[object]] = {}

    def borrow[T](self, key: str, load: Callable[[], ComponentLease[T]]) -> ComponentLease[T]:
        if key not in self._leases:
            self._leases[key] = cast(ComponentLease[object], load())
        return ComponentLease(cast(T, self._leases[key].value))

    def close(self) -> None:
        leases, self._leases = self._leases, {}
        try:
            while leases:
                _, lease = leases.popitem()
                lease.close()
        finally:
            for lease in leases.values():
                lease.close()


def execute_seed_stages[C, R](
    jobs: Sequence[Generator[C | None, None, R]],
    models: SeedModels,
    mode: SeedMode,
) -> Iterator[R]:
    """Run already-conditioned jobs: denoise yield, then decode and return.

    Callers prime jobs through their conditioning yield before entering this
    scheduler. Each model call remains batch one in every execution mode.
    """
    try:
        if mode == "stage":
            for job in jobs:
                next(job)
            models.close()
        for job in jobs:
            if mode != "stage":
                next(job)
                if mode == "sequential":
                    models.close()
            try:
                next(job)
            except StopIteration as finished:
                result = cast(R, finished.value)
            else:
                raise RuntimeError("Image recipe yielded an unexpected stage")
            if mode == "sequential":
                models.close()
            yield result
    finally:
        try:
            for job in jobs:
                job.close()
        finally:
            models.close()
