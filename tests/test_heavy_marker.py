"""A heavy test refuses to start inside an xdist worker.

``pytest_configure`` clears the worker list for a ``--run-heavy`` run, so the
ordinary commands never reach this guard. It exists for any other path that
could hand a heavy item to a worker, and it must refuse before the test body
reads a checkpoint. The stub item below is the only way to exercise it without
starting a second pytest process.
"""

import pytest

from tests.conftest import HEAVY_IN_WORKER, pytest_runtest_setup


class _StubConfig:
    """An xdist worker's config carries ``workerinput``; a controller's does not."""

    def __init__(self, *, worker: bool) -> None:
        if worker:
            self.workerinput = {"workerid": "gw0"}


class _StubItem:
    def __init__(self, *, marker: str, worker: bool) -> None:
        self._marker = marker
        self.config = _StubConfig(worker=worker)

    def get_closest_marker(self, name: str) -> object | None:
        return pytest.mark.heavy if name == self._marker else None


def test_a_heavy_item_handed_to_a_worker_fails_before_it_runs() -> None:
    with pytest.raises(pytest.fail.Exception) as refusal:
        pytest_runtest_setup(_StubItem(marker="heavy", worker=True))
    assert str(refusal.value) == HEAVY_IN_WORKER
    assert "-n 0" in HEAVY_IN_WORKER

    # The guard is silent everywhere else: an ordinary test in a worker, and a
    # heavy test in the controller process that --run-heavy left serial.
    assert pytest_runtest_setup(_StubItem(marker="other", worker=True)) is None
    assert pytest_runtest_setup(_StubItem(marker="heavy", worker=False)) is None
