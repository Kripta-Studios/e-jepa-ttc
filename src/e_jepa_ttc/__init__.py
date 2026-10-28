"""PHASE-TTC, retaining the historical e_jepa_ttc import namespace."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phase-ttc")
except PackageNotFoundError:  # pragma: no cover
    try:
        __version__ = version("e-jepa-ttc")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__"]
