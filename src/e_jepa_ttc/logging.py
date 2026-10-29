"""Consistent logging setup shared by scripts and library entry points."""

from __future__ import annotations

import logging as _logging
from pathlib import Path


def configure_logging(level: str = "INFO", *, log_file: str | Path | None = None) -> None:
    """Configure a console logger and optionally an UTF-8 file handler."""

    numeric_level = getattr(_logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level!r}.")
    handlers: list[_logging.Handler] = [_logging.StreamHandler()]
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(_logging.FileHandler(destination, encoding="utf-8"))
    _logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


__all__ = ["configure_logging"]
