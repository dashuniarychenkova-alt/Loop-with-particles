"""Lightweight singleton decorator used by config and language modules."""
from __future__ import annotations

from threading import Lock
from typing import TypeVar, cast

T = TypeVar("T", bound=type)


def singleton(cls: T) -> T:
    """Return a thread-safe singleton version of ``cls``."""

    class _SingletonWrapper(cls):  # type: ignore[misc,valid-type]
        _instance = None
        _init_lock = Lock()
        _initialized = False

        def __new__(wrapper_cls, *args, **kwargs):  # noqa: N804 (matching PyPI style)
            if wrapper_cls._instance is None:
                with wrapper_cls._init_lock:
                    if wrapper_cls._instance is None:
                        wrapper_cls._instance = super().__new__(wrapper_cls, *args, **kwargs)
            return wrapper_cls._instance

        def __init__(self, *args, **kwargs):
            if not self.__class__._initialized:
                super().__init__(*args, **kwargs)
                self.__class__._initialized = True

        @classmethod
        def instance(cls) -> "_SingletonWrapper":
            """Explicit accessor mirroring ``py_singleton.singleton``."""
            return cls()  # type: ignore[return-value]

    _SingletonWrapper.__name__ = cls.__name__
    _SingletonWrapper.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    _SingletonWrapper.__module__ = cls.__module__
    _SingletonWrapper.__doc__ = cls.__doc__

    return cast(T, _SingletonWrapper)
