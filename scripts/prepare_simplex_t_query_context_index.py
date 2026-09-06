"""Build D0 query-conditioned sensor-window metadata, without fitting or event reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.data.event_v4_geometry import shifted_precontext_window
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    """Use only visible identities, supplied crops and input timestamps for membership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--exposure-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous context metadata")
    started = time.perf_counter()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(
        Path("configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = json.loads(Path(ancestry_ref["path"]).read_text(encoding="utf-8"))
    allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    root = Path(ancestry_ref["path"]).parent
    index = json.loads((root / "FROZEN_EXPERT_TABLE_INDEX.json").read_text(encoding="utf-8"))
    sources = {
        str(args.binding): compute_file_hash(str(args.binding)),
        str(args.exposure_audit): compute_file_hash(str(args.exposure_audit)),
    }
    if (
        sources[str(args.binding)]
        != "47b3ee83d61654cf716399a4f564043d2f2289b6785d8c26c456b941e25e70f1"
    ):
        raise ValueError("historical input binding changed")
    if sources[str(args.exposure_audit)] != (
        "d44b1830186d313f2d9d46e9021dc0d155f1f3c01de41d4cf87fa4dc01b35e23"
    ):
        raise ValueError("audited current exposure dependencies changed")
    columns = [
        "sample_token",
        "sequence_id",
        "window_id",
        "window_start_us",
        "window_end_us",
        "roi_x0",
        "roi_y0",
        "roi_x1",
        "roi_y1",
        "events_path_relative",
    ]
    binding = pd.read_csv(args.binding, usecols=columns)
    if not set(binding.sequence_id) <= allowed or len(binding) != 16384:
        raise ValueError("binding includes a closed group or wrong cohort")
    exposure = json.loads(args.exposure_audit.read_text(encoding="utf-8"))
    if exposure["queries"] != 8192 or exposure["targets_read"]:
        raise ValueError("invalid input-only exposure audit")
    times = {r["sample_token"]: r for r in exposure["rows"]}
    tokens = sorted(binding.sample_token.unique())
    if len(tokens) != 8192 or set(tokens) != set(times):
        raise ValueError("query/exposure identity mismatch")
    token_index = {token: i for i, token in enumerate(tokens)}
    assignments = np.full((3, len(tokens)), -1, dtype=np.int16)
    family_records = []
    producers = {(p["outer_fold"], p["role"], p["expert"]): p for p in ancestry["producers"]}
    for outer in range(3):
        for role in ("inner0", "inner1", "inner2", "outer_dev"):
            experts = {
                e: producers[(outer, role, e)]["checkpoint_sha256"] for e in ("A5", "C2F", "PAIR")
            }
            if producers[(outer, role, "PAIR")]["nested_a5_ancestor_sha256"] != experts["A5"]:
                raise ValueError("PAIR teacher family mismatch")
            family_records.append(
                {
                    "outer_fold": outer,
                    "role": role,
                    "experts": experts,
                    "family_sha256": hashlib.sha256(
                        json.dumps(experts, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        for table_role in ("inner_oof", "outer_dev"):
            record = next(r for r in index if r["outer_fold"] == outer and r["role"] == table_role)
            path = root / "tables" / f"outer{outer}_{table_role}.csv"
            sources[str(path)] = compute_file_hash(str(path))
            if sources[str(path)] != record["metadata_sha256"]:
                raise ValueError("current role table changed")
            table = pd.read_csv(
                path, usecols=lambda c: c in {"sample_token", "sequence_id", "inner_fold"}
            )
            for row in table.to_dict("records"):
                role = (
                    "outer_dev" if table_role == "outer_dev" else f"inner{int(row['inner_fold'])}"
                )
                position = token_index[row["sample_token"]]
                if assignments[outer, position] != -1:
                    raise ValueError("duplicate fold assignment")
                for expert in ("A5", "C2F"):
                    split = producers[(outer, role, expert)]["split_validation"]
                    if (
                        not split["ancestry_validated"]
                        or row["sequence_id"] in split["train_sequence_ids"]
                        or row["sequence_id"] not in split["dev_sequence_ids"]
                    ):
                        raise ValueError("query not excluded from producer ancestors")
                assignments[outer, position] = outer * 4 + (
                    3 if role == "outer_dev" else int(role[-1])
                )
    if (assignments < 0).any():
        raise ValueError("incomplete lineage assignments")
    stream_bounds = {}
    raw_root = (Path(paths["eap_root"]) / "data/train").resolve(strict=True)
    for sequence in sorted(allowed):
        relative = binding.loc[binding.sequence_id == sequence, "events_path_relative"].unique()
        if len(relative) != 1:
            raise ValueError("ambiguous raw stream")
        path = (raw_root / relative[0]).resolve(strict=True)
        if not path.is_relative_to(raw_root):
            raise ValueError("source escapes TRAIN root")
        with EAPEventReader(path) as reader:
            stream_bounds[sequence] = (reader.t_start_us, reader.t_end_us)
    base_windows = np.zeros((len(tokens), 3, 2), dtype=np.int64)
    squares = np.zeros((len(tokens), 4), dtype=np.float64)
    roi_available = np.zeros(len(tokens), dtype=np.int64)
    anchors = np.zeros(len(tokens), dtype=np.int64)
    valid = np.zeros((len(tokens), 16), dtype=bool)
    sequences = [""] * len(tokens)
    lag = np.arange(15, -1, -1, dtype=np.int64) * 50_000
    for token, rows in binding.groupby("sample_token", sort=False):
        selected = rows.sort_values("window_id")
        if selected.window_id.tolist() != [0, 1]:
            raise ValueError("ambiguous endpoint windows")
        first, second = selected.to_dict("records")
        i = token_index[token]
        windows = [
            shifted_precontext_window(
                (first["window_start_us"], first["window_end_us"]), shift_s=0.1
            ),
            (first["window_start_us"], first["window_end_us"]),
            (second["window_start_us"], second["window_end_us"]),
        ]
        base_windows[i] = windows
        squares[i] = [first[f"roi_{axis}"] for axis in ("x0", "y0", "x1", "y1")]
        sequences[i] = first["sequence_id"]
        anchors[i] = times[token]["anchor_us"]
        roi_available[i] = times[token]["selected_exposure_end_us"]
        if anchors[i] != windows[-1][-1] or roi_available[i] < anchors[i]:
            raise ValueError("event clock/exposure mismatch")
        start, end = stream_bounds[sequences[i]]
        valid[i] = (base_windows[i].min() - lag >= start) & (base_windows[i].max() - lag <= end)
    if not valid[:, -1].all():
        raise ValueError("current query lacks raw source support")
    args.output.mkdir(parents=True)
    resources = admitted([args.output])
    if not resources["has_headroom"]:
        raise RuntimeError("RESOURCE_PAUSE")
    array_path = args.output / "query_context_index.npz"
    np.savez_compressed(
        array_path,
        tokens=np.asarray(tokens),
        sequences=np.asarray(sequences),
        base_windows_us=base_windows,
        square_xyxy=squares,
        anchor_us=anchors,
        roi_available_us=roi_available,
        lag_us=lag,
        valid=valid,
        producer_family=assignments,
    )
    write_new_json(
        args.output / "INDEX_MANIFEST.json",
        {
            "status": "QUERY_CONTEXT_INDEX_PREPARED_NOT_FEATURE_CACHE",
            "namespace": "SIMPLEX_T_QUERY_CONTEXT_AMENDMENT",
            "queries": len(tokens),
            "families": family_records,
            "stream_bounds_us": stream_bounds,
            "history_count_distribution_h16": {
                str(c): int((valid.sum(1) == c).sum()) for c in range(1, 17)
            },
            "h8_available_queries": int((valid[:, -8:].sum(1) == 8).sum()),
            "index_sha256": compute_file_hash(str(array_path)),
            "sources": sources,
            "ancestry": ancestry_ref,
            "roles": ack["interfaces"]["role_manifest"],
            "target_fields_read_for_membership": False,
            "event_payload_windows_read": False,
            "availability_policy": (
                "At least current supplied ROI exposure end; annotation production latency "
                "is unknown; no online-tracker claim"
            ),
            "optimizer_updates": 0,
            "seconds": time.perf_counter() - started,
            "resources": resources,
        },
    )
    print(
        json.dumps({"queries": len(tokens), "h8_available": int((valid[:, -8:].sum(1) == 8).sum())})
    )


if __name__ == "__main__":
    main()
