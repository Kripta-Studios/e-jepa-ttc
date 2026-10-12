"""Avoid reading non-Python process metadata that every native guard discards."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar
from unittest.mock import patch

import psutil

ORIGINAL_SCAN = psutil.process_iter
P = ParamSpec("P")
T = TypeVar("T")
SUPPORTED = (
    {"name", "cmdline"},
    {"name", "cmdline", "memory_info"},
    {"name", "cmdline", "ppid"},
)


def python_first(attrs: list | None = None, ad_value: object = None) -> Iterator:
    if attrs is None or set(attrs) not in SUPPORTED:
        yield from ORIGINAL_SCAN(attrs, ad_value)
        return
    for process in ORIGINAL_SCAN(["name"], ad_value):
        try:
            if not (process.info["name"] or "").lower().startswith("python"):
                continue
            process.info = process.as_dict(attrs=attrs, ad_value=ad_value)
            yield process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def wrap(original: Callable[P, T]) -> Callable[P, T]:
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        with patch.object(psutil, "process_iter", python_first):
            return original(*args, **kwargs)

    return guarded
