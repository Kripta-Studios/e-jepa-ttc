"""Inventory public eAP train sequences and write a signed bounded-pilot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.data.eap import (  # noqa: E402
    build_eap_object_windows,
    load_eap_media_table,
    load_eap_sequence_labels,
    reconstruct_eap_object_states,
)
from e_jepa_ttc.data.eap_pilot import select_eap_pilot_sequences  # noqa: E402
from e_jepa_ttc.utils.io import write_structured  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ttc_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "approaching_0p1_2s": int(np.count_nonzero((values >= 0.1) & (values < 2.0))),
        "approaching_2_4s": int(np.count_nonzero((values >= 2.0) & (values < 4.0))),
        "approaching_4_8s": int(np.count_nonzero((values >= 4.0) & (values < 8.0))),
        "approaching_8_20s": int(np.count_nonzero((values >= 8.0) & (values <= 20.0))),
        "receding_0p1_20s": int(np.count_nonzero((values <= -0.1) & (values >= -20.0))),
        "nonfinite_or_outside_20s": int(
            values.size
            - np.count_nonzero(
                ((values >= 0.1) & (values <= 20.0)) | ((values <= -0.1) & (values >= -20.0))
            )
        ),
    }


def _anchors(path: Path) -> tuple[list[str], dict[str, Any]]:
    if not path.is_file():
        return [], {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    attributes = payload.get("sequence_attributes", {})
    return (
        sorted(key for key, value in attributes.items() if isinstance(value, dict)),
        attributes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--sequences", type=int, default=12)
    parser.add_argument("--validation-sequences", type=int, default=3)
    parser.add_argument("--maximum-event-gib", type=float, default=20.0)
    parser.add_argument(
        "--attributes-config",
        type=Path,
        default=Path("configs/data/eap50_object_jepa.yaml"),
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("data/manifests/eap_train40_inventory_v1.json"),
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("data/splits/eap_pilot12_v1.json"),
    )
    args = parser.parse_args()

    media_path = args.root / "data" / "train.parquet"
    media = load_eap_media_table(args.root, split="train")
    sequence_ids = sorted(str(value) for value in media["sequence_id"].unique())
    rows: list[dict[str, Any]] = []
    for index, sequence_id in enumerate(sequence_ids, start=1):
        labels_path = args.root / "data" / "train" / sequence_id / "labels.parquet"
        events_path = args.root / "data" / "train" / sequence_id / "events.h5"
        labels = load_eap_sequence_labels(args.root, sequence_id, split="train")
        sequence_media = media[media["sequence_id"].astype(str) == sequence_id].copy()
        states = reconstruct_eap_object_states(sequence_media, labels)
        windows = build_eap_object_windows(
            states,
            history_frames=3,
            horizons_ms=(100, 250, 500),
            maximum_slop_ms=25,
            maximum_history_gap_ms=125,
            ttc_range_s=(-20.0, 20.0),
        )
        with h5py.File(events_path, "r") as handle:
            event_count = int(handle["events/t"].shape[0])
            millisecond_count = int(handle["ms_to_idx"].shape[0] - 1)
        ttc = np.asarray([state.ttc_s for state in states], dtype=np.float64)
        row = {
            "sequence_id": sequence_id,
            "media_sample_count": int(sequence_media.shape[0]),
            "label_count": int(labels.shape[0]),
            "track_count": int(labels["track_id"].astype(str).nunique()),
            "projected_state_count": len(states),
            "object_window_count": len(windows),
            "category_counts": dict(sorted(Counter(state.category for state in states).items())),
            "ttc_proxy_counts": _ttc_counts(ttc),
            "event_count": event_count,
            "event_duration_ms": millisecond_count,
            "event_file_bytes": events_path.stat().st_size,
            "event_file_gib": events_path.stat().st_size / 1024**3,
            "labels_sha256": _sha256(labels_path),
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(sequence_ids):02d}] {sequence_id}: "
            f"states={len(states)} windows={len(windows)}",
            flush=True,
        )

    anchor_ids, attributes = _anchors(args.attributes_config)
    selection = select_eap_pilot_sequences(
        rows,
        sequence_count=args.sequences,
        validation_count=args.validation_sequences,
        anchor_sequence_ids=anchor_ids,
        maximum_event_gib=args.maximum_event_gib,
    )
    inventory = {
        "artifact_type": "eap_public_train40_inventory_v1",
        "dataset": "NAIL-HNU/eAP-dataset",
        "source_root_recording": "external_local_root_not_embedded",
        "media_table_sha256": _sha256(media_path),
        "sequence_count": len(rows),
        "event_file_bytes_total": sum(int(row["event_file_bytes"]) for row in rows),
        "rows": rows,
        "official_ttc_labels_available": False,
        "ttc_proxy_definition": "negative_nearest_depth_over_local_depth_derivative",
        "rgb_opened": False,
        "evttc_used_for_selection": False,
        "benchmark10_opened": False,
    }
    write_structured(args.inventory_output, inventory)
    split = {
        "artifact_type": "eap_pilot12_sequence_split_v1",
        "dataset": "NAIL-HNU/eAP-dataset-public-train40",
        "inventory_artifact_sha256": json.loads(args.inventory_output.read_text(encoding="utf-8"))[
            "artifact_sha256"
        ],
        "selection_method": (
            "documented_attribute_anchors_plus_standardized_farthest_point; "
            "validation_medoid_plus_farthest_points"
        ),
        "selection_parameters": {
            "sequence_count": args.sequences,
            "validation_sequence_count": args.validation_sequences,
            "maximum_event_gib": args.maximum_event_gib,
            "anchor_sequence_ids": anchor_ids,
        },
        "assignments": {
            "train": selection["train"],
            "validation": selection["validation"],
        },
        "selected": selection["selected"],
        "excluded_large_outliers": selection["excluded_large_outliers"],
        "documented_sequence_attributes": {
            sequence_id: attributes[sequence_id]
            for sequence_id in selection["selected"]
            if sequence_id in attributes
        },
        "train_validation_disjoint": not bool(
            set(selection["train"]) & set(selection["validation"])
        ),
        "rgb_opened": False,
        "evttc_used_for_selection": False,
        "benchmark10_opened": False,
    }
    write_structured(args.split_output, split)
    print(json.dumps(split, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
