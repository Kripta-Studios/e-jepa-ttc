"""Build the signed label-independent 32/8 split over public eAP train-40."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.eap_pilot import select_eap_full_sequences  # noqa: E402
from e_jepa_ttc.utils.io import read_structured, write_structured  # noqa: E402


def main() -> int:
    """Create the full eAP split from signed metadata without opening raw data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/manifests/eap_train40_inventory_v1.json"),
    )
    parser.add_argument(
        "--pilot-split",
        type=Path,
        default=Path("data/splits/eap_pilot12_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/eap_train40_v1.json"),
    )
    parser.add_argument("--validation-sequences", type=int, default=8)
    parser.add_argument("--selection-salt", default="eap_full40_validation_v1")
    args = parser.parse_args()

    inventory = read_structured(args.inventory)
    pilot = read_structured(args.pilot_split)
    if not verify_artifact_hash(inventory) or not verify_artifact_hash(pilot):
        raise ValueError("eAP inventory and pilot split must have valid signatures.")
    if pilot.get("inventory_artifact_sha256") != inventory.get("artifact_sha256"):
        raise ValueError("eAP pilot split and inventory signatures do not match.")
    rows = [row for row in inventory.get("rows", []) if isinstance(row, dict)]
    assignments = pilot.get("assignments", {})
    preserved = assignments.get("validation", []) if isinstance(assignments, dict) else []
    selection = select_eap_full_sequences(
        rows,
        validation_count=args.validation_sequences,
        preserved_validation_ids=preserved,
        selection_salt=args.selection_salt,
    )
    split = {
        "artifact_type": "eap_train40_sequence_split_v1",
        "dataset": "NAIL-HNU/eAP-dataset-public-train40",
        "inventory_artifact_sha256": inventory["artifact_sha256"],
        "selection_method": "preserve_pilot_validation_then_salted_sha256_sequence_id",
        "selection_parameters": {
            "sequence_count": len(selection["selected"]),
            "validation_sequence_count": args.validation_sequences,
            "selection_salt": args.selection_salt,
            "preserved_pilot_validation_ids": sorted(str(value) for value in preserved),
        },
        "assignments": {
            "train": selection["train"],
            "validation": selection["validation"],
        },
        "selected": selection["selected"],
        "excluded_large_outliers": selection["excluded_large_outliers"],
        "train_validation_disjoint": not bool(
            set(selection["train"]) & set(selection["validation"])
        ),
        "labels_used_for_assignment": False,
        "rgb_opened": False,
        "evttc_used_for_selection": False,
        "benchmark10_opened": False,
    }
    write_structured(args.output, split)
    signed = read_structured(args.output)
    print(json.dumps(signed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
