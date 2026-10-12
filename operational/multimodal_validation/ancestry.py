"""Relocate read-only endpoint validation while retaining exact historical checks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from operational.rgb_port_revision import migration

from .common import OUT, ROOT, RUN, V13, sha, write


@contextmanager
def historical_source_binding() -> Iterator[None]:
    """Require the admitted original bytes and only CRLF/LF-equivalent loaded sources.

    The historical migration binds CRLF source files in V13. The new Git
    checkout uses LF via .gitattributes. No legacy file or freeze is rewritten.
    The original validator still runs, against its original historical root;
    every relocated source must separately match after newline normalization.
    """
    with patch.object(migration, "ROOT", V13):
        validated = migration.validate(RUN)
        sources = {}
        for relative in validated["files"]:
            original, relocated = V13 / relative, ROOT / relative
            old, new = original.read_bytes(), relocated.read_bytes()
            if old.replace(b"\r\n", b"\n") != new.replace(b"\r\n", b"\n"):
                raise ValueError("Relocated source differs beyond CRLF/LF: " + relative)
            sources[relative] = {"original_sha256": sha(original), "loaded_sha256": sha(relocated)}
        write(
            OUT / "SOURCE_RELOCATION.json",
            {
                "original_migration_verified": True,
                "sources": sources,
                "allowed_difference": "CRLF versus LF only",
                "historical_files_modified": False,
                "optimizer_updates": 0,
            },
        )
        yield
