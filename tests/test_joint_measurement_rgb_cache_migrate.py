"""A cache migration must never seal a live writer or discard undurable updates."""

from types import SimpleNamespace

import pytest

from operational.joint_measurement import rgb_cache_migrate as migration
from operational.range_transfer.common import read, sha, write


@pytest.mark.parametrize("mode", ["live", "undurable", "durable"])
def test_seal_requires_dead_writer_and_durable_checkpoint(tmp_path, monkeypatch, mode):
    out, fit, auth = tmp_path / "runtime", tmp_path / "fit", tmp_path / "authority.json"
    monkeypatch.setattr(migration, "OUT", out)
    monkeypatch.setattr(migration, "FIT", fit)
    monkeypatch.setattr(migration, "AUTH", auth)
    out.mkdir()
    original = b'{"status": "APPLIED"}\n'
    (out / "authorization_original.bytes").write_bytes(original)
    held = {"status": "RUNTIME_MIGRATION_HOLD"}
    write(auth, held)
    write(out / "authorization_hold.json", held)
    write(
        out / "MIGRATION.json",
        {
            "old_pid": 99,
            "old_created": 123.0,
            "original_authorization_sha256": sha(out / "authorization_original.bytes"),
        },
    )
    monkeypatch.setattr(
        migration.psutil,
        "Process",
        lambda _: SimpleNamespace(
            is_running=lambda: mode == "live",
            create_time=lambda: 123.0,
        ),
    )
    if mode == "live":
        with pytest.raises(ValueError, match="remains live"):
            migration.seal()
        assert read(auth) == held
        return
    write(
        fit / "UPDATE_JOURNAL.json",
        {
            "completed_updates": 10,
            "durable_updates": 9 if mode == "undurable" else 10,
            "pending_update_upper": 0,
            "recovery_upper": 0,
        },
    )
    versions = fit / "checkpoint_versions"
    versions.mkdir()
    checkpoint = versions / "checkpoint_10.pt"
    checkpoint.write_bytes(b"immutable test checkpoint bytes")
    write(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": 10,
            "version": checkpoint.name,
            "checkpoint_sha256": sha(checkpoint),
        },
    )
    write(fit / "CHECKPOINT_RECEIPT.json", {"completed_updates": 10})
    if mode == "undurable":
        with pytest.raises(ValueError, match="undurable trajectory"):
            migration.seal()
        assert read(auth) == held
        assert not (out / "origin_checkpoint.pt").exists()
    else:
        result = migration.seal()
        assert result["origin_updates"] == 10 and result["lost_updates"] == 0
        assert auth.read_bytes() == original
        assert (out / "origin_checkpoint.pt").read_bytes() == checkpoint.read_bytes()
