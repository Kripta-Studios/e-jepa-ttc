"""Compare complete original-role input identities without using TTC or geometry labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.original_release import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    charter = json.loads(Path(ack["interfaces"]["time_charter"]["path"]).read_text())
    garl_path = Path(charter["metadata_source"]["path"])
    if sha256(garl_path) != charter["metadata_source"]["sha256"]:
        raise ValueError("Garl metadata changed")
    results = []
    for sequence in sorted(ack["interfaces"]["role_manifest"]["roles"]["original"]):
        root = Path("artifacts/simplex_t/original_annotations") / sequence
        receipt = json.loads((root / "DOWNLOAD_RECEIPT.json").read_text())
        for member in receipt["members"]:
            if sha256(root / Path(member["zip_member"]).name) != member["sha256"]:
                raise ValueError("original annotation bytes changed")
        # Project input fields immediately. Numeric target/3D values are never inspected.
        originals = {
            (r["file_name"], r["instance_id"]): tuple(r["bbox"])
            for r in read_records(root / "annotations.pkl")
        }
        label_path = Path(paths["eap_root"]) / "data/train" / sequence / "labels.parquet"
        hf_rows = pq.read_table(
            label_path, columns=["sequence_id", "frame_name", "instance_id", "track_id"]
        ).to_pylist()
        if any(r["sequence_id"] != sequence for r in hf_rows):
            raise ValueError("unexpected label-file role")
        hf_keys = {(f"rgb/{Path(r['frame_name']).name}", r["instance_id"]) for r in hf_rows}
        pairs = (
            ds.dataset(garl_path)
            .to_table(
                columns=["track_id", "rgb_member_paths", "boxes_xyxy"],
                filter=ds.field("sequence_id") == sequence,
            )
            .to_pylist()
        )
        observations = set()
        for pair in pairs:
            for frame, box in zip(pair["rgb_member_paths"], pair["boxes_xyxy"], strict=True):
                observations.add((frame, pair["track_id"], tuple(box)))
        counts: Counter[str] = Counter()
        for frame, identity, box in observations:
            raw = originals.get((frame, identity))
            if raw is None:
                counts["missing_exact_frame_instance"] += 1
                continue
            counts["exact_frame_instance_found"] += 1
            x, y, a, b = raw
            counts["same_id_raw_xyxy_exact"] += int(raw == box)
            counts["same_id_documented_xywh_exact"] += int((x, y, x + a, y + b) == box)
        results.append(
            {
                "sequence": sequence,
                "original_object_keys": len(originals),
                "hf_rows": len(hf_rows),
                "hf_unique_object_keys": len(hf_keys),
                "original_only_keys": len(originals.keys() - hf_keys),
                "hf_only_keys": len(hf_keys - originals.keys()),
                "hf_track_instance_differences": sum(
                    r["track_id"] != r["instance_id"] for r in hf_rows
                ),
                "garl_pairs": len(pairs),
                "unique_garl_observations": len(observations),
                "garl_identity_box_counts": dict(counts),
                "hf_labels_sha256": sha256(label_path),
                "original_receipt_sha256": sha256(root / "DOWNLOAD_RECEIPT.json"),
            }
        )
    result = {
        "sequences": results,
        "garl_metadata_sha256": charter["metadata_source"]["sha256"],
        "scope": "ALL_GARL_INPUT_PAIRS_IN_NINE_ORIGINAL_ROLES_NOT_ONLY_D0",
        "numeric_target_depth_velocity_fields_used": False,
        "fuzzy_identity_matching_used": False,
        "production_mapping_adopted": False,
        "scientific_updates": 0,
    }
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
