"""Verify native event tensors and all H history identities before test12 inference."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader
from operational.multimodal_validation.common import RUN, V12, admission, read, sha, write

from . import test12_native_inputs as native
from .common import OUT, authorize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_root = V12 / "artifacts/train40_system_20261005"
    freeze = read(source_root / "PREPARE_FREEZE.json")
    rows = pd.read_parquet(source_root / "TRAIN40_ROWS.parquet", columns=list(native.EVENT_COLUMNS))
    rows["global_index"] = np.arange(len(rows))
    h_tokens = pd.read_parquet(
        Path(read(RUN / "H_MANIFEST.json")["rows_path"]), columns=["sample_token"]
    )
    h = cast(pd.DataFrame, rows[rows.sample_token.isin(h_tokens.sample_token)])
    jobs = [native.native_job(row, args.media_root) for row in h.to_dict("records")]
    histories = native.native_histories(jobs)
    history_path = RUN / "features/H/E_RAW/history.npz"
    with np.load(history_path, allow_pickle=False) as z:
        for token, ids, valid in zip(z["query_id"], z["observation_id"], z["valid"], strict=True):
            assert histories[str(token)] == ids[valid].astype(str).tolist()
        assert len(histories) == len(z["query_id"]) == 18422
    controls = []
    for _, group in h.groupby("sequence_id", sort=True):
        ranked = sorted(
            group.to_dict("records"),
            key=lambda r: hashlib.sha256(r["sample_token"].encode()).hexdigest(),
        )
        controls.extend(("H", row, row["global_index"]) for row in ranked[:2])
    test_path = V12 / "artifacts/sota_evidence_20261010/sources/test_inputs.parquet"
    test = pd.read_parquet(test_path, columns=list(native.EVENT_COLUMNS))
    previous_path = V12 / "artifacts/garlttc_submission_20261010/INPUT_FREEZE.json"
    previous = read(previous_path)
    assert sha(test_path) == previous["input_sha256"]
    old_lookup = {r["sample_token"]: i for i, r in enumerate(previous["rows"])}
    for _, group in test.groupby("sequence_id", sort=True):
        row = min(
            group.to_dict("records"),
            key=lambda r: hashlib.sha256(r["sample_token"].encode()).hexdigest(),
        )
        controls.append(("TEST12", row, old_lookup[row["sample_token"]]))
    records = []
    for split, row, index in controls:
        admission()
        authorization = authorize()
        assert (
            datetime.now(UTC) - datetime.fromisoformat(authorization["utc"])
        ).total_seconds() < authorization["max_elapsed_seconds"]
        job = native.native_job(row, args.media_root)
        reader = CachedEventReader(Path(job["raw_path"]))
        try:
            actual = native.native_event(job, reader).numpy()
        finally:
            reader.close()
        if split == "H":
            path = Path(freeze["cache_root"]) / f"shard_{index // 32:05d}.npz"
            with np.load(path, allow_pickle=False) as z:
                assert str(z["tokens"][index % 32]) == row["sample_token"]
                expected = z["events"][index % 32]
                assert np.array_equal(
                    np.asarray(job["square_xyxy"], np.float32), z["square"][index % 32]
                )
                assert np.float32(job["delta_t_s"]) == z["delta"][index % 32]
        else:
            path = args.test_cache / f"query_{index:05d}.npz"
            receipt = read(path.with_suffix(".json"))
            assert receipt["sample_token"] == row["sample_token"]
            assert receipt["input_freeze_sha256"] == sha(previous_path)
            assert receipt["sha256"] == sha(path)
            with np.load(path, allow_pickle=False) as z:
                assert z["valid"][-1]
                expected = z["events"][-1]
        assert np.array_equal(actual, expected), row["sample_token"]
        records.append(
            dict(
                split=split,
                sample_token=row["sample_token"],
                max_abs_error=float(np.max(np.abs(actual - expected))),
                reference_sha256=sha(path),
            )
        )
        print(split, len(records), "exact", flush=True)
    destination = OUT / "test12_readiness"
    write(
        destination / "NATIVE_EVENT_PARITY.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            status="PASS",
            records=records,
            H_history_queries=len(histories),
            H_history_identities_exact=True,
            H_history_sha256=sha(history_path),
            native_source_sha256=sha(Path(native.__file__)),
            source_sha256=sha(__file__),
            test_targets_read=False,
            optimizer_updates=0,
            scope="Native same-track histories; H8 current observation tensor only is compared, "
            "never the H8 history. No candidate predictions or private labels.",
        ),
    )


if __name__ == "__main__":
    main()
