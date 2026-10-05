"""Isolated authorized E0–E3 campaign execution."""

from .resource_policy import install

install()

from .fast_owned_scan import install_if_sealed  # noqa: E402

install_if_sealed()
