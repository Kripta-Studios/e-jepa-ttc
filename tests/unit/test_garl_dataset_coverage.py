from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_official_garlttc_release import audit_dataset_coverage


def test_dataset_coverage_blocks_train46_claim_without_hiding_missing_ids(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    split = release / "configs" / "splits" / "train.txt"
    split.parent.mkdir(parents=True)
    split.write_text("seq-a\nseq-b\nseq-c\n", encoding="utf-8")

    eap = tmp_path / "eap" / "data"
    garl = tmp_path / "garl" / "data"
    eap.mkdir(parents=True)
    garl.mkdir(parents=True)
    pd.DataFrame({"sequence_id": ["seq-a", "seq-b"]}).to_parquet(eap / "train.parquet", index=False)
    pd.DataFrame({"sequence_id": ["seq-a", "seq-b"]}).to_parquet(
        garl / "train.parquet", index=False
    )

    report = audit_dataset_coverage(
        release,
        eap_root=eap.parent,
        garlttc_root=garl.parent,
    )

    assert report["status"] == "pass"
    assert report["coverage_complete"] is False
    assert report["train_sequence_coverage"] == "2/3"
    assert report["missing_by_root"]["eap"] == ["seq-c"]
    assert report["missing_by_root"]["garlttc"] == ["seq-c"]
    assert report["retraining_claim"] == "public_train40_retraining_only"
    assert report["retraining_claim_allowed"] is False
    assert report["snapshot_check"]["download_authorization_required"] is True
