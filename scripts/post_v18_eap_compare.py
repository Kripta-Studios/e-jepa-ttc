"""Bounded descriptive eAP comparison with the released full Garl checkpoint.

Public Garl TRAIN lists all eight internal V groups. This is not an independent
held-out comparison for Garl, and cannot support a SOTA claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.evttc_rgb_transfer.model import FullGarl
from operational.joint_measurement.garl_route_cost import COLUMNS
from operational.sota_eval import garl_parity
from scripts.post_v18_transfer_compare import OUT, PUBLIC, ROOT, RUN13, V12
from scripts.post_v18_transfer_inputs import read, sha, write

DEST = OUT / "eap"
EAP = Path("E:/eAP_dataset")


def register() -> None:
    """Choose eight evenly spaced sorted query IDs in each group, without labels."""
    if (DEST / "REGISTRATION.json").exists():
        raise ValueError("eAP registration already exists")
    manifest = read(RUN13 / "V_MANIFEST.json")
    rows_path = Path(manifest["rows_path"])
    if sha(rows_path) != manifest["rows_sha256"]:
        raise ValueError("V metadata changed")
    frame = pd.read_parquet(rows_path, columns=["sample_token", "group_id"])
    selected = []
    for _, group in frame.groupby("group_id", sort=True):
        ordered = group.sort_values("sample_token")
        selected.append(ordered.iloc[np.linspace(0, len(ordered) - 1, 8).round().astype(int)])
    keys = pd.concat(selected, ignore_index=True)
    source = V12 / "artifacts/train40_system_20261005/TRAIN40_ROWS.parquet"
    native = pd.read_parquet(source, columns=list(COLUMNS))
    merged = keys.merge(
        native, on="sample_token", validate="one_to_one", how="left", indicator=True
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Missing native metadata")
    serialized = merged.drop(columns="_merge").to_json(orient="records")
    if not isinstance(serialized, str):
        raise TypeError("Expected serialized metadata")
    rows = json.loads(serialized)
    for row in rows:
        for relative in [row["events_path"], *row["rgb_shard_paths"]]:
            if not (EAP / relative).is_file():
                raise FileNotFoundError(relative)
    train = Path("E:/Garl-TTC/configs/splits/train.txt")
    overlap = sorted(set(frame.group_id).intersection(train.read_text().split()))
    write(DEST / "QUERY_MANIFEST.json", {"rows": rows, "targets_read": False})
    write(
        DEST / "REGISTRATION.json",
        {
            "queries": len(rows),
            "groups": 8,
            "selection": "Eight evenly spaced sorted IDs per V group; no label/error selection",
            "manifest_sha256": sha(DEST / "QUERY_MANIFEST.json"),
            "V_manifest_sha256": sha(RUN13 / "V_MANIFEST.json"),
            "native_rows_sha256": sha(source),
            "public_train_sha256": sha(train),
            "V_groups_in_public_train": overlap,
            "independent_generalization_for_public_Garl": False,
            "checkpoint_sha256": sha(PUBLIC / "paper_ours_full.pth"),
            "sources": {
                str(path.relative_to(ROOT)): sha(path)
                for path in (
                    Path(__file__),
                    Path(garl_parity.__file__),
                    ROOT / "operational/evttc_rgb_transfer/model.py",
                )
            },
            "local_helpers": garl_parity._local_source_hashes(),
            "device": "CPU_FP32_two_threads",
            "optimizer_updates": 0,
            "max_wall_seconds": 7200,
            "error_policy": "Keep every query, including missing/nonfinite output; no replacement",
            "preprocessing_parity": (
                "Bit-exact upstream-vs-local sensors on first query of each group before scoring"
            ),
            "own_predictions": (
                "Exact sealed V16 seed7 and V18 seed7 predictions joined only after full Garl seal"
            ),
        },
    )
    print(json.dumps({"registered": len(rows), "groups": 8, "public_train_overlap": overlap}))


def predict() -> None:
    """Execute native full Garl and save all outputs before joining public targets."""
    registration = read(DEST / "REGISTRATION.json")
    for name, expected in registration["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("eAP source changed")
    if registration["local_helpers"] != garl_parity._local_source_hashes():
        raise ValueError("Garl preprocessing helpers changed")
    if sha(DEST / "QUERY_MANIFEST.json") != registration["manifest_sha256"]:
        raise ValueError("eAP manifest changed")
    rows = read(DEST / "QUERY_MANIFEST.json")["rows"]
    torch.set_num_threads(2)
    model = FullGarl(PUBLIC, Path("E:/Garl-TTC"), device="cpu")
    if model.bindings["checkpoint_sha256"] != registration["checkpoint_sha256"]:
        raise ValueError("Garl checkpoint changed")
    write(DEST / "MODEL.json", model.bindings)
    start = time.monotonic()
    first_groups = set()
    for index, row in enumerate(rows):
        first = row["group_id"] not in first_groups
        first_groups.add(row["group_id"])
        path = DEST / f"query_{index:04d}.json"
        if path.exists():
            saved = read(path)
            if saved["sample_token"] != row["sample_token"] or saved["registration_sha256"] != sha(
                DEST / "REGISTRATION.json"
            ):
                raise ValueError("Saved eAP fragment differs")
            continue
        if time.monotonic() - start > registration["max_wall_seconds"]:
            raise TimeoutError("CPU route budget exhausted; fragments retained")
        before = time.perf_counter()
        local, _, _ = garl_parity._local_sensor(row, EAP)
        parity = None
        if first:
            reference, _ = garl_parity._reference_sensor(row, EAP)
            parity = bool(np.array_equal(local, reference))
            if not parity:
                raise ValueError("Upstream/local sensor parity failed")
        prepared = time.perf_counter()
        prediction = model.predict_sensor(local)
        value = prediction["ttc"]
        if not isinstance(value, float):
            raise TypeError("Expected scalar TTC")
        after = time.perf_counter()
        write(
            path,
            {
                "sample_token": row["sample_token"],
                "group_id": row["group_id"],
                "registration_sha256": sha(DEST / "REGISTRATION.json"),
                "Garl_full": value if np.isfinite(value) else None,
                "sensor_sha256": hashlib.sha256(local.tobytes()).hexdigest(),
                "upstream_bit_exact": parity,
                "seconds_model": after - prepared,
                "seconds_preprocess_including_parity": prepared - before,
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )
        print(json.dumps({"route": "eap_garl", "done": index + 1, "total": len(rows)}), flush=True)
    write(
        DEST / "SEALED.json",
        {
            "queries": len(rows),
            "registration_sha256": sha(DEST / "REGISTRATION.json"),
            "files": {
                f"query_{i:04d}.json": sha(DEST / f"query_{i:04d}.json") for i in range(len(rows))
            },
            "targets_read": False,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "predict"))
    arguments = parser.parse_args()
    register() if arguments.action == "register" else predict()
