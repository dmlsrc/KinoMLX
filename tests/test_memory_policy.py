"""Invocation ordering and process-wide MLX memory settings."""

from types import SimpleNamespace

import pytest

from kinomlx import MLXMemoryPolicy, Settings, apply_memory_policy
from kinomlx import memory_policy as policy_module

GIB = 1024**3


@pytest.fixture
def setters(monkeypatch):
    calls = []
    monkeypatch.setattr(policy_module.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        policy_module.mx, "device_info", lambda: {"max_recommended_working_set_size": 60 * GIB}
    )
    monkeypatch.setattr(policy_module.mx, "set_cache_limit", lambda n: calls.append(("cache", n)))
    monkeypatch.setattr(policy_module.mx, "set_wired_limit", lambda n: calls.append(("wired", n)))
    return calls


def test_defaults_apply_both_controls(setters):
    apply_memory_policy(Settings())
    assert setters == [("cache", GIB), ("wired", 60 * GIB)]


def test_sequential_invocations_reapply_changed_limits_and_disable_wiring(setters):
    for policy in [MLXMemoryPolicy(2, 48), MLXMemoryPolicy(0, 0), MLXMemoryPolicy(None, None)]:
        with policy_module.memory_policy_scope(policy):
            pass
    assert setters == [
        ("cache", 2 * GIB),
        ("wired", 48 * GIB),
        ("cache", 0),
        ("wired", 0),
        ("wired", 60 * GIB),
    ]


@pytest.mark.parametrize(
    "policy",
    [
        MLXMemoryPolicy(2, 61),
        MLXMemoryPolicy(-1, 40),
        MLXMemoryPolicy(1, float("nan")),
        MLXMemoryPolicy(float("inf"), 40),
    ],
)
def test_invalid_policy_changes_neither_limit(setters, policy):
    with pytest.raises(ValueError, match="limit"):
        apply_memory_policy(policy)
    assert setters == []


def test_nested_recipe_applies_once_and_preserves_keyword_calling(setters):
    resources = SimpleNamespace(memory_policy=MLXMemoryPolicy())

    @policy_module.memory_managed
    def recipe(request, resources, *, option):
        assert len(setters) == 2
        return request + option

    with policy_module.memory_policy_scope(resources.memory_policy):
        assert recipe(request=2, resources=resources, option=3) == 5
    assert len(setters) == 2


def test_failed_invocation_releases_scope_and_rejects_nested_changes(setters):
    def invoke():
        with policy_module.memory_policy_scope(MLXMemoryPolicy()):
            with (
                pytest.raises(ValueError, match="same memory policy"),
                policy_module.memory_policy_scope(MLXMemoryPolicy(2, 40)),
            ):
                pass
            with pytest.raises(RuntimeError, match="active invocation"):
                apply_memory_policy(Settings())
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        invoke()
    apply_memory_policy(MLXMemoryPolicy(2, 40))
    assert setters[-2:] == [("cache", 2 * GIB), ("wired", 40 * GIB)]


def test_cpu_host_applies_allocator_without_wiring(setters, monkeypatch):
    monkeypatch.setattr(policy_module.mx.metal, "is_available", lambda: False)
    apply_memory_policy(Settings())
    assert setters == [("cache", GIB)]
