import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from e_jepa_ttc.experiments.test_lock import check_final_test_lock


def compute_sha256(path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_audit_schema() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / "schemas" / "cache_audit_v3.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", type=Path, required=True, help="Path to the cache npz file")
    parser.add_argument("--output", type=Path, required=True, help="Output json path")
    parser.add_argument("--mode", type=str, choices=["sampled", "exhaustive"], default="exhaustive")
    parser.add_argument(
        "--evidence-type",
        type=str,
        required=True,
        choices=[
            "synthetic_ci",
            "real_smoke",
            "validation_pilot",
            "validation_matrix",
            "final_test",
        ],
    )
    args = parser.parse_args()

    npz_path = args.npz_path
    output_path = args.output
    mode = args.mode

    if not npz_path.exists():
        logging.error(f"NPZ file not found: {npz_path}")
        sys.exit(1)

    sidecar_path = npz_path.with_name(npz_path.stem + ".summary.json")
    if not sidecar_path.exists():
        logging.error(f"Sidecar file not found: {sidecar_path}")
        sys.exit(1)

    # Output dictionary building
    audit_record: dict[str, Any] = {
        "schema_version": "3.0",
        "status": "failed",
        "audit_mode": mode,
        "evidence_type": args.evidence_type,
        "cache_format_version": 0,
        "cache_path": str(npz_path.resolve()),
        "cache_sha256_computed": "",
        "cache_sha256_declared": "",
        "sidecar_sha256_matches": False,
        "normalize": False,
        "normalization": "unknown",
        "normalizer_source_split": "unknown",
        "normalizer_origins_verified": False,
        "sample_count_total": 0,
        "sample_count_audited": 0,
        "nonempty_samples_collapsed_to_zero": 0,
        "checks": {},
        "failures": [],
        "warnings": [],
        "provenance": {"timestamp": datetime.now(UTC).isoformat()},
    }

    failures = []

    # 1. Compute SHA256
    computed_sha256 = compute_sha256(npz_path)
    audit_record["cache_sha256_computed"] = computed_sha256

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception as e:
        failures.append(f"Failed to read sidecar: {e}")
        audit_record["failures"] = failures
        _write_output(output_path, audit_record)
        sys.exit(1)

    # 4. Validate Sidecar (Minimal checks for required fields first)
    if not isinstance(sidecar, dict):
        failures.append("Sidecar must be a JSON object")
    else:
        audit_record["cache_format_version"] = sidecar.get("format_version", 1)
        # 2. Read declared SHA256
        audit_record["cache_sha256_declared"] = sidecar.get("cache_sha256", sidecar.get("sha256", ""))
        # 3. Compare both
        if audit_record["cache_sha256_computed"] == audit_record["cache_sha256_declared"]:
            audit_record["sidecar_sha256_matches"] = True
        else:
            failures.append("Sidecar SHA-256 does not match computed SHA-256")

        # 11. Validate Cache format version
        if audit_record["cache_format_version"] not in (1, 2):
            failures.append(f"Invalid cache format version: {audit_record['cache_format_version']}")

        # 12. Normalization metadata
        norm_config = sidecar.get("normalization", {})
        if isinstance(norm_config, dict):
            audit_record["normalize"] = norm_config.get("enabled", False)
            audit_record["normalization"] = norm_config.get("strategy", "unknown")
            audit_record["normalizer_source_split"] = norm_config.get("source_split", "unknown")

            # 13. Verify normalization stats originated from train only
            if audit_record["normalize"]:
                if audit_record["normalizer_source_split"] == "train":
                    prov = norm_config.get("provenance", {})
                    if (
                        "training_sample_hash" in prov
                        and "train_split_manifest_hash" in prov
                        and "normalizer_statistics_hash" in prov
                        and "cache_builder_config_hash" in prov
                    ):
                        audit_record["normalizer_origins_verified"] = True
                    else:
                        failures.append("Normalization provenance is incomplete (missing hashes).")
                else:
                    failures.append(
                        f"Normalizer fitted with non-train split: {audit_record['normalizer_source_split']}"
                    )

        expected_total = sidecar.get("window_count", sidecar.get("total_samples", 0))
        audit_record["sample_count_total"] = expected_total

    if failures:
        audit_record["failures"] = failures
        _write_output(output_path, audit_record)
        sys.exit(1)

    try:
        data = np.load(npz_path, mmap_mode="r")
    except Exception as e:
        failures.append(f"Failed to load npz: {e}")
        audit_record["failures"] = failures
        _write_output(output_path, audit_record)
        sys.exit(1)

    with data:
        # 5. Validate required arrays
        required_arrays = ["x", "sequence_id", "split"]
        for arr in required_arrays:
            if arr not in data:
                failures.append(f"Missing required array: {arr}")

        if failures:
            audit_record["failures"] = failures
            _write_output(output_path, audit_record)
            sys.exit(1)

        x = data["x"]
        seqs = data["sequence_id"]
        splits = data["split"]

        # 6. Validate sample-axis consistency
        if not (x.shape[0] == seqs.shape[0] == splits.shape[0]):
            failures.append(
                f"Sample axis inconsistency: x={x.shape[0]}, seq={seqs.shape[0]}, split={splits.shape[0]}"
            )

        # 19. Validate sample count against sidecar
        if x.shape[0] != audit_record["sample_count_total"]:
            failures.append(
                f"Declared sample count {audit_record['sample_count_total']} != actual {x.shape[0]}"
            )

        # 7. Validate dtypes and shapes
        if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.integer):
            failures.append(f"Invalid x dtype: {x.dtype}")

        # Determine indices to scan
        total_samples = x.shape[0]
        if total_samples == 0:
            failures.append("Empty cache")

        if failures:
            audit_record["failures"] = failures
            _write_output(output_path, audit_record)
            sys.exit(1)

        if mode == "sampled":
            # 20. Fixed audit seed in sampled mode
            rng = np.random.RandomState(42)
            n_samples = min(max(1, total_samples // 20), 100)
            indices = rng.choice(total_samples, n_samples, replace=False)
            x_scan = x[indices]
            audit_record["sample_count_audited"] = len(indices)
            audit_record["checks"]["sampled_indices"] = indices.tolist()
        else:
            x_scan = x
            audit_record["sample_count_audited"] = total_samples

        # 8. Validate finite values
        if np.issubdtype(x_scan.dtype, np.floating):
            if np.isnan(x_scan).any():
                failures.append("NaN values found in 'x' array")
            if np.isinf(x_scan).any():
                failures.append("Infinity values found in 'x' array")

        # 9. Validate sequence/split disjointness
        # 10. Validate allowed split names
        allowed_splits = {"train", "validation", "calibration", "test"}
        if args.evidence_type != "final_test":
            allowed_splits.remove("test")
        else:
            if not check_final_test_lock("final"):
                failures.append("final_test requested but no valid unlock found")

        seq_to_split = {}
        for s, sp in zip(seqs, splits):
            if sp not in allowed_splits:
                failures.append(f"Unknown or illegal split name: {sp}")
            if s in seq_to_split and seq_to_split[s] != sp:
                failures.append(
                    f"Sequence {s} belongs to multiple splits (e.g. {seq_to_split[s]} and {sp})"
                )
            seq_to_split[s] = sp

        # 16. Detect broken all-zero channels
        if x_scan.ndim >= 2:
            # Assuming channel is axis 1 if shape is (N, C, ...)
            channel_sums = np.sum(
                np.abs(x_scan), axis=tuple(range(2, x_scan.ndim)) if x_scan.ndim > 2 else (0,)
            )
            # Sum over samples as well
            if channel_sums.ndim > 1:
                channel_sums = np.sum(channel_sums, axis=0)

            if np.any(channel_sums == 0) and total_samples > 0:
                failures.append("Detected broken all-zero channel across all audited samples")

        # 17. Detect source windows with events that became all zero
        if "event_count" in data or "source_event_count" in data:
            source_counts = (
                data["event_count"] if "event_count" in data else data["source_event_count"]
            )
            sample_sums = np.sum(np.abs(x_scan).reshape(x_scan.shape[0], -1), axis=1)

            if mode == "sampled":
                source_counts_scan = source_counts[indices]
            else:
                source_counts_scan = source_counts

            nonempty_collapsed = int(np.sum((source_counts_scan > 0) & (sample_sums == 0)))
            audit_record["nonempty_samples_collapsed_to_zero"] = nonempty_collapsed
            if nonempty_collapsed > 0:
                failures.append(f"{nonempty_collapsed} non-empty source windows converted to zero")
        else:
            audit_record["status"] = "not_verifiable"
            audit_record["reason"] = "source_event_counts_missing"
            failures.append("Source event counts missing from cache.")

        # Compute sparse-event checks instead of boolean flag
        total_occupied = int(np.sum(x_scan != 0))
        audit_record["checks"]["total_occupied_voxels"] = total_occupied

        if x_scan.ndim >= 2:
            spatial_axes = tuple(range(2, x_scan.ndim)) if x_scan.ndim > 2 else ()
            if spatial_axes:
                channel_occ = np.sum(x_scan != 0, axis=(0,) + spatial_axes)
            else:
                channel_occ = np.sum(x_scan != 0, axis=0)
            audit_record["checks"]["channel_occupancy"] = channel_occ.tolist()

            sample_sums = np.sum(np.abs(x_scan).reshape(x_scan.shape[0], -1), axis=1)
            all_zero_samples = int(np.sum(sample_sums == 0))
            audit_record["checks"]["encoded_all_zero_samples"] = all_zero_samples

        # 18. Validate navigation channels and validity masks
        if "navigation" in data:
            nav = data["navigation"]
            if nav.shape[0] != x.shape[0]:
                failures.append("Navigation sample axis inconsistency")

        if "validity_mask" in data:
            mask = data["validity_mask"]
            if mask.shape[0] != x.shape[0]:
                failures.append("Validity mask sample axis inconsistency")

    if not failures:
        audit_record["status"] = "passed"

    audit_record["failures"] = failures
    _write_output(output_path, audit_record)

    if failures:
        logging.error(f"Audit completed with {len(failures)} errors: {failures}")
        sys.exit(1)
    else:
        logging.info(f"Cache validation written to {output_path}")
        sys.exit(0)


def _write_output(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        schema = _get_audit_schema()
        jsonschema.validate(instance=record, schema=schema)
    except Exception as e:
        record["status"] = "failed"
        record["failures"].append(f"Schema validation failed: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
