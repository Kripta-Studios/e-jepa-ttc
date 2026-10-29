"""Bounded schema feasibility audit for label-independent 2D object history."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json


def main() -> None:
    """Inspect original-group schema footers and filenames, never target values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--interfaces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    interfaces = json.loads(args.interfaces.read_text(encoding="utf-8"))
    role_record = interfaces["role_manifest"]
    role_path = Path(role_record["path"])
    if sha256(role_path) != role_record["sha256"]:
        raise ValueError("discovered role map changed")
    roles = json.loads(role_path.read_text(encoding="utf-8"))["roles"]
    eap = Path(paths["eap_root"])
    schemas = []
    for sequence in roles["original"]:
        path = eap / "data/train" / sequence / "labels.parquet"
        if not path.is_file():
            schemas.append({"sequence": sequence, "path": str(path), "status": "MISSING"})
            continue
        with pq.ParquetFile(path) as file:
            fields = file.schema_arrow.names
            schemas.append(
                {
                    "sequence": sequence,
                    "path": str(path),
                    "status": "FOOTER_ONLY",
                    "rows": file.metadata.num_rows,
                    "fields": fields,
                    "has_2d_bbox_field": bool(set(fields) & {"bbox", "boxes_xyxy", "bbox_2d"}),
                    "payload_columns_read": [],
                }
            )
    roots = [
        paths["garl_code_candidate"],
        str(Path(paths["garl_annotations_candidate"]).parent.parent),
        paths["eap_root"],
    ]
    result = subprocess.run(
        [
            "rg",
            "--files",
            "--hidden",
            "--no-ignore",
            *roots,
            "-g",
            "*.pkl",
            "-g",
            "frames.parquet",
            "-g",
            "annotations.parquet",
            "-g",
            "!**/.venv/**",
            "-g",
            "!**/.cache/**",
            "-g",
            "!**/.hf/**",
            "-g",
            "!**/rgb_shards/**",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr)
    builder = Path(paths["garl_code_candidate"]) / "garl_ttc_benchmark/build_garlttc_dataset.py"
    write_new_json(
        args.output,
        {
            "artifact_type": "simplex_t_timeline_feasibility_v1",
            "roles_reference": role_record,
            "footers": schemas,
            "legacy_filename_search_roots": roots,
            "legacy_filename_candidates": result.stdout.splitlines(),
            "search_exclusions": [".venv", ".cache", ".hf", "rgb_shards"],
            "garl_builder": {"path": str(builder), "sha256": sha256(builder)},
            "garl_source_findings": {
                "function": "_train_records",
                "range_filter": "ttc < gt_range[0] or ttc > gt_range[1]",
                "height_filter": "nonfinite box3d_h or box3d_h outside [0,10]",
                "whole_pair_table_eligible_as_primary_timeline": False,
            },
            "status": "HISTORY_SOURCE_PRIVILEGED_OR_UNRESOLVED",
            "required_source": (
                "Unfiltered local timestamp/track/2D-bbox observations before TTC/3D filters"
            ),
            "excluded_substitutions": [
                "8192 target-stratified queries",
                "filtered Garl pair rows",
                "3D depth-projected boxes",
                "future whole-track ROI",
            ],
            "next_action": "Ask owner/user for an existing original annotations/frame source path",
            "current_only_family": (
                "Not blocked by history absence; still needs replay/freeze/resource QA"
            ),
            "targets_or_3d_values_read": False,
            "confirmation_or_protected_payloads_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "original_sequence_footers": len(schemas),
                "legacy_candidates": len(result.stdout.splitlines()),
            }
        )
    )


if __name__ == "__main__":
    main()
