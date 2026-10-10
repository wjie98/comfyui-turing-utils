"""Cycle-free helpers for instance method patches.

Storing ``types.MethodType(function, module)`` back on ``module`` creates a
reference cycle even when ``function`` captures nothing.  ModelPatcher relies
on prompt-scoped clones becoming unreachable without a full cyclic GC, so the
patches here bind through a weak proxy and retain original methods in unbound
form whenever possible.
"""

from __future__ import annotations

import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OriginalMethod:
    """An original callable that does not strongly retain its bound owner."""

    function: Callable
    pass_owner: bool
    owner: weakref.ReferenceType | None = None

    @classmethod
    def capture(cls, original: Callable, owner: object) -> "OriginalMethod":
        function = getattr(original, "__func__", None)
        bound_owner = getattr(original, "__self__", None)
        if callable(function) and bound_owner is not None:
            try:
                same_owner = bound_owner is owner or bound_owner == owner
            except ReferenceError:
                same_owner = False
            if same_owner:
                return cls(function, True)
            try:
                return cls(function, True, weakref.ref(bound_owner))
            except TypeError:
                pass
        return cls(original, False)

    def __call__(self, current_owner: object, /, *args, **kwargs):
        if not self.pass_owner:
            return self.function(*args, **kwargs)
        owner = current_owner if self.owner is None else self.owner()
        if owner is None:
            raise ReferenceError(
                "the owner of the original patched method was released"
            )
        return self.function(owner, *args, **kwargs)


def weak_method(function: Callable, owner: object):
    """Bind an instance patch without making the owner retain itself."""
    return types.MethodType(function, weakref.proxy(owner))


__all__ = ["OriginalMethod", "weak_method"]
