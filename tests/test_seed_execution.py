"""Seed modes preserve stage ownership, ordering, and failure cleanup."""

import pytest

from kinomlx.components import ComponentLease
from kinomlx.seed_execution import SeedModels, execute_seed_stages
from kinomlx.seeds import resolve_seeds


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "sequential",
            [
                "load-t",
                "t-9",
                "close-t",
                "load-v",
                "v-9",
                "close-v",
                "load-t",
                "t-2",
                "close-t",
                "load-v",
                "v-2",
                "close-v",
            ],
        ),
        ("stage", ["load-t", "t-9", "t-2", "close-t", "load-v", "v-9", "v-2", "close-v"]),
        ("resident", ["load-t", "t-9", "load-v", "v-9", "t-2", "v-2", "close-v", "close-t"]),
    ],
)
def test_stage_order(mode, expected):
    models = SeedModels()
    events = []

    def load(name):
        events.append(f"load-{name}")
        return ComponentLease(object(), cleanup=lambda: events.append(f"close-{name}"))

    def job(seed):
        yield "conditioning"
        with models.borrow("t", lambda: load("t")):
            events.append(f"t-{seed}")
        yield None
        with models.borrow("v", lambda: load("v")):
            events.append(f"v-{seed}")
        return seed

    jobs = [job(seed) for seed in (9, 2)]
    for item in jobs:
        next(item)
    assert list(execute_seed_stages(jobs, models, mode)) == [9, 2]
    assert events == expected


@pytest.mark.parametrize("mode", ["sequential", "stage", "resident"])
@pytest.mark.parametrize("phase", ["denoise", "decode"])
def test_failure_releases_every_loaded_model(mode, phase):
    models = SeedModels()
    loaded, closed = [], []

    def load(name):
        loaded.append(name)
        return ComponentLease(object(), cleanup=lambda: closed.append(name))

    def job():
        yield "conditioning"
        with models.borrow("t", lambda: load("t")):
            if phase == "denoise":
                raise KeyboardInterrupt
        yield None
        with models.borrow("v", lambda: load("v")):
            raise KeyboardInterrupt

    item = job()
    next(item)
    with pytest.raises(KeyboardInterrupt):
        list(execute_seed_stages([item], models, mode))
    assert sorted(loaded) == sorted(closed)


def test_selection_preserves_duplicates_and_checks_overflow():
    assert resolve_seeds(0, (9, 2, 9)) == (9, 2, 9)
    assert resolve_seeds(9, None, 3) == (9, 10, 11)
    for seeds in [(), (True,), (-1,), (2**64,), (1.5,)]:
        with pytest.raises(ValueError, match="seeds"):
            resolve_seeds(0, seeds)
    with pytest.raises(ValueError, match="images_per_prompt"):
        resolve_seeds(0, (9, 2), 2)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        resolve_seeds(2**64 - 1, None, 2)
