"""Build causal ACTIVE eAP geometry inputs from verified TRAIN40 metadata only."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import load_data, read
from operational.phase_campaign.evaluation import metrics
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_startup.geometry import startup_geometry
from scripts.final_train40_transfer import HEAD

HISTORY_COLUMNS = [
    "sample_token",
    "sequence_id",
    "track_id",
    "query_time_us",
    "cutoff_us",
    "producer_times_us",
    "producer_available_us",
    "producer_boxes_xyxy",
    "observation_times_us",
    "observation_available_us",
    "observation_boxes_xyxy",
]


def causal_history(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate one object's supplied history with both time and availability gates.

    Annotation/query clocks share one domain. Sensor absolute timestamps do not.
    Row-local construction avoids joining distinct objects within a sequence.
    """
    query, cutoff = int(row["query_time_us"]), int(row["cutoff_us"])
    if cutoff > query:
        raise ValueError("Cutoff exceeds query time")
    times = [row["producer_times_us"], *row["observation_times_us"]]
    availability = [row["producer_available_us"], *row["observation_available_us"]]
    boxes = [row["producer_boxes_xyxy"], *row["observation_boxes_xyxy"]]
    observations: dict[int, np.ndarray] = {}
    for clocks, arrivals, group in zip(times, availability, boxes, strict=True):
        for clock, arrival, box in zip(clocks, arrivals, group, strict=True):
            clock, arrival = int(clock), int(arrival)
            if arrival < clock:
                raise ValueError("Availability precedes observation")
            if not query - 650000 <= clock <= query or arrival > cutoff:
                continue
            value = np.asarray(box, dtype=np.float64)
            if value.shape != (4,):
                raise ValueError("Expected four bbox coordinates")
            if clock in observations and not np.array_equal(observations[clock], value):
                raise ValueError("Conflicting boxes at one physical timestamp")
            observations[clock] = value
    clocks_out = np.array(sorted(observations), dtype=np.int64)
    boxes_out = np.array([observations[int(t)] for t in clocks_out], dtype=np.float64)
    return boxes_out.reshape(-1, 4), clocks_out


def build(args: argparse.Namespace) -> None:
    """Bind existing source states to causal geometry without fitting any parameters."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    values, _, cache_receipt = load_data(args.data)
    paths = verified_upstream_paths(args.candidate)
    candidate = read(args.candidate / "BUNDLE.json")["parent"]
    if candidate != read(args.data / "BINDING.json")["candidate"]:
        raise ValueError("Candidate differs from the source feature cache")
    audit = read(args.audit)
    source = values["domain"] == "eap"
    tokens = values["sample_token"][source]
    groups = values["group_id"][source]
    count = len(tokens)
    if count != 88744 or len(np.unique(groups)) != 40 or len(np.unique(tokens)) != count:
        raise ValueError("Expected complete unique TRAIN40 source population")
    lookup = {str(token): i for i, token in enumerate(tokens)}
    geometry = np.full(count, np.nan, dtype=np.float32)
    quality = np.zeros((count, 20), dtype=np.float32)
    tracks = np.empty(count, dtype="U128")
    clocks = np.empty(count, dtype=np.int64)
    reasons = np.empty(count, dtype="U64")
    historical_roles = np.empty(count, dtype="U1")
    seen: set[str] = set()
    counts: Counter[int] = Counter()
    role_groups: dict[str, set[str]] = {}
    metadata_inputs = {}
    for role in ("P", "H", "V"):
        entry = audit["train40"]["sources"][role]
        path = args.metadata / f"{role}_ROWS.parquet"
        digest = file_sha(path)
        if digest != entry["rows_file_sha256"]:
            raise ValueError(f"Changed source metadata: {role}")
        metadata_inputs[role] = {"path": str(path.resolve()), "sha256": digest}
        role_groups[role] = set()
        for batch in pq.ParquetFile(path).iter_batches(columns=HISTORY_COLUMNS, batch_size=2048):
            for row in batch.to_pylist():
                token = "eap:" + row["sample_token"]
                if token in seen or token not in lookup:
                    raise ValueError("Duplicated or unmatched source token")
                seen.add(token)
                i = lookup[token]
                if groups[i] != "eap:" + row["sequence_id"]:
                    raise ValueError("Source sequence/group identity differs")
                boxes, timestamps = causal_history(row)
                physical = startup_geometry(boxes, timestamps, row["query_time_us"])
                if physical["valid"]:
                    geometry[i] = np.clip(physical["ttc"], -60, 60)
                quality[i] = physical["quality"]
                reasons[i] = physical["reason"]
                counts[len(timestamps)] += 1
                tracks[i] = row["track_id"]
                clocks[i] = row["query_time_us"]
                historical_roles[i] = role
                role_groups[role].add(row["sequence_id"])
        print(f"Metadata {role} complete; joined {len(seen)}/{count}", flush=True)
    if seen != set(lookup) or any(
        role_groups[a] & role_groups[b] for a, b in (("P", "H"), ("P", "V"), ("H", "V"))
    ):
        raise ValueError("Incomplete token join or overlapping historical groups")
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(
        torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    )
    features = torch.from_numpy(values["features"][source])
    widths = torch.from_numpy(values["rgb_width_logits"][source])
    with torch.inference_mode():
        native = torch.stack(
            [
                baseline(features, widths)["ttc"],
                phase_to_ttc(features[:, 160]),
                phase_to_ttc(features[:, 330]),
            ],
            dim=1,
        ).numpy()
        target = phase_to_ttc(torch.from_numpy(values["target_phase"][source])).numpy()
    arrays = {
        "native": native,
        "geometry": geometry,
        "quality": quality,
        "target": target,
        "target_phase": values["target_phase"][source],
        "group": groups,
        "track": tracks,
        "token": tokens,
        "clocks": clocks,
        "geometry_reason": reasons,
        "historical_role": historical_roles,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "POPULATION.npz"
    save_npz(destination, arrays)
    sources = [
        Path(__file__),
        Path("operational/phase_startup/geometry.py"),
        Path("operational/phase_campaign/data.py"),
        Path("operational/phase_campaign/evaluation.py"),
        Path("operational/phase_robust/bundle.py"),
        Path("operational/final_refit/training.py"),
        Path("src/e_jepa_ttc/simplex_t/phase.py"),
        Path("scripts/final_train40_transfer.py"),
        Path("operational/phase_adapt_cache.py"),
    ]
    active = (quality[:, 9] > 0) & (quality[:, 11] <= 0.15)
    report = {
        "schema": "phase_safe_source_active_audit_v1",
        "status": "COMPLETE",
        "rows": count,
        "groups": 40,
        "tracks": len(np.unique(tracks)),
        "active_history_rows": int(active.sum()),
        "valid_geometry_rows": int(np.isfinite(geometry).sum()),
        "history_counts": {str(k): v for k, v in sorted(counts.items())},
        "geometry_reasons": dict(Counter(reasons.tolist())),
        "exact_token_join": True,
        "historical_groups_disjoint": True,
        "historical_role_groups": {k: sorted(v) for k, v in role_groups.items()},
        "original_metrics": {
            name: metrics(target, native[:, i])
            for i, name in enumerate(("F0", "RGB_context", "event_context"))
        },
        "geometry_metrics": metrics(target, geometry),
        "candidate": candidate,
        "cache_receipt": cache_receipt,
        "metadata_inputs": metadata_inputs,
        "audit_sha256": file_sha(args.audit),
        "upstream_bundle_sha256": file_sha(args.candidate / "BUNDLE.json"),
        "population_sha256": file_sha(destination),
        "sources_sha256": {str(p): file_sha(p) for p in sources},
        "feature_contract": (
            "native=F0,RGB_context,event_context; geometry clipped +/-60s; quality20"
        ),
        "target_contract": "Existing target_phase converted by phase_to_ttc; +/-60s support",
        "scope": (
            "TRAIN40 resubstitution preservation audit; final F0 head fitted all 88744 rows. "
            "Upstream features are FIVE_FOLD_PHYSICAL_OOF. Historical P/H/V are all TRAIN. "
            "This is not held-out eAP generalization or deployment-feature evaluation."
        ),
        "history_contract": (
            "Row-local same-object supplied public bbox geometry; deduplicated annotation clocks; "
            "timestamp in [query-650000,query], availability<=cutoff<=query. No media decoded. "
            "Public annotations are the supplied ROI input, not an evaluated detector."
        ),
        "optimizer_updates": 0,
        "raw_media_read": False,
    }
    atomic_receipt(args.output / "COMPLETE.json", report)
    print(
        f"COMPLETE: {count} source rows; active={active.sum()}; "
        f"valid geometry={np.isfinite(geometry).sum()}"
    )


def main() -> None:
    """Parse explicit local input locations and build the source audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "metadata", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
