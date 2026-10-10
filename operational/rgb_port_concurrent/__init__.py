"""Additive two-producer concurrency supervisor for RGB-PORT."""

from .contracts import build_concurrent_freeze, validate_concurrent_freeze

__all__ = ["build_concurrent_freeze", "validate_concurrent_freeze"]
