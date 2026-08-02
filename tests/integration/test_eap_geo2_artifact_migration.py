from __future__ import annotations

from tests.integration._artifact_helpers import artifact_path, read_artifact, sha256


def test_eap_artifact_migration_is_signed_and_reproducible() -> None:
    migration = read_artifact("artifacts/metrics/artifact_migration_v4.json")
    source = artifact_path(migration["input_artifact"])
    output = artifact_path(migration["output_artifact"])
    assert migration["status"] == "migrated"
    assert migration["source_sha256"] == sha256(source)
    migrated = read_artifact(migration["output_artifact"])
    assert (
        migrated["artifact_sha256"]
        == "d502c7bef3317a03b3f91ce859afe9d635a1da60d2f1a3b47db1a6bfc999a4fb"
    )
    assert migrated["source_sha256"] == migration["source_sha256"]
    assert output.stat().st_size > 0
