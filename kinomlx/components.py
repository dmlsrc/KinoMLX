"""Model-neutral ownership primitive for heavyweight runtime components."""

from __future__ import annotations

from collections.abc import Callable


class ComponentLease[ComponentT]:
    """Own one loaded component until an explicit or contextual close.

    The lease deliberately exposes the component structurally instead of
    defining model-specific operations. Attribute access and calls are proxied
    while the lease is live, and rejected once ownership has ended.
    """

    def __init__(
        self,
        component: ComponentT,
        *,
        close_component: Callable[[ComponentT], None] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._component: ComponentT | None = component
        self._close_component = close_component
        self._cleanup = cleanup

    @property
    def closed(self) -> bool:
        """Whether this lease has relinquished its component."""
        return self._component is None

    @property
    def value(self) -> ComponentT:
        """Return the live component or reject use after close."""
        component = self._component
        if component is None:
            raise RuntimeError("component lease is closed")
        return component

    def __enter__(self) -> ComponentT:
        """Return the owned component while this context owns its lease."""
        return self.value

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __call__(self, *args: object, **kwargs: object) -> object:
        component = self.value
        if not callable(component):
            raise TypeError(f"leased component {type(component).__name__} is not callable")
        return component(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.value, name)

    def close(self) -> None:
        """Release the component exactly once and run post-release cleanup."""
        component = self._component
        if component is None:
            return
        close_component = self._close_component
        cleanup = self._cleanup
        # Clear every lease-owned reference before cleanup. A close callback
        # may itself capture the component (for example, a model-specific
        # teardown closure), so retaining that callback would keep heavyweight
        # arrays live while the cleanup hook collects and clears MLX memory.
        self._component = None
        self._close_component = None
        self._cleanup = None
        try:
            if close_component is not None:
                close_component(component)
        finally:
            component = None
            close_component = None
            if cleanup is not None:
                cleanup()


__all__ = ["ComponentLease"]
