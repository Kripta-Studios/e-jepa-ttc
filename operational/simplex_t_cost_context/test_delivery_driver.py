"""Lightweight actual publication/recovery tests with zero model/optimizer work."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from . import deliver


class Admission:
    def __init__(self, fail_at: int = 0) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def check(self) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise InterruptedError("synthetic transient publication pause")


def test_fragmented_zip_resume_retains_committed_members(tmp_path: Path) -> None:
    files = {}
    for i in range(40):
        path = tmp_path / str(i)
        path.write_bytes(("payload" + str(i)).encode())
        files[f"members/{i}"] = path
    manifest = dict(members={k: dict(sha256=deliver.digest(v)) for k, v in files.items()})
    encoded = json.dumps(manifest).encode()
    archive = tmp_path / "bundle.zip"
    with pytest.raises(InterruptedError):
        deliver.fragmented_zip(archive, files, manifest, encoded, Admission(fail_at=2))
    assert not archive.exists()
    with zipfile.ZipFile(archive.with_suffix(".zip.pending")) as z:
        assert len(z.namelist()) == 32
    deliver.fragmented_zip(archive, files, manifest, encoded, Admission())
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())) == 41
        assert z.testzip() is None
        assert z.read("CONTENT_MANIFEST.json") == encoded
        for name, source in files.items():
            assert z.read(name) == source.read_bytes()


def test_zero_checkpoint_and_pending_work_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    night, h16 = tmp_path / "night", tmp_path / "h16"
    night.mkdir()
    h16.mkdir()
    monkeypatch.setattr(deliver, "NIGHT", night)
    monkeypatch.setattr(deliver, "H16", h16)
    key = "H16_REPLICATION_20261003/TPR-D1-H16-C160/fold0/seed13"
    deliver.atomic_json(
        night / "QUEUE_AUTHORIZED.json",
        dict(
            fits=[
                dict(
                    id=key,
                    arm="TPR-D1-H16-C160",
                    fold=0,
                    seed=13,
                )
            ]
        ),
    )
    checkpoint = h16 / "fits" / key / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic initial checkpoint bytes")
    deliver.atomic_json(
        h16 / "PHYSICAL_WORK.json",
        dict(
            fits={key: dict(completed=0, pending=[0, 100], uncertain_lost_upper=0)},
            accounting=dict(scientific_saved_updates=0, scientific_uncertain_lost_upper=0),
        ),
    )
    result = deliver.inventory()
    assert result["scientific_saved_updates"] == 0
    assert result["repeated_or_uncertain_upper"] == 100
    assert result["scientific_physical_upper"] == 100
    assert result["endpoints"] == 0
    assert result["fits"][0]["status"] == "RECOVERABLE_PARTIAL"
