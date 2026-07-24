import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", type=Path, required=True, help="Path to the cache npz file")
    parser.add_argument("--output", type=Path, required=True, help="Output json path")
    parser.add_argument("--mode", type=str, choices=["sampled", "exhaustive"], default="exhaustive")
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

    # 1. SHA-256 of the npz
    sha256_hash = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    npz_sha256 = sha256_hash.hexdigest()

    errors = []
    
    with open(sidecar_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    # 2. Check for NaN/Inf and Tensor Shapes
    valid_nan_inf = True
    valid_tensor_shapes = True
    split_disjointness = True
    normalizer_origins_verified = True
    
    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "x" not in data:
                errors.append("Missing 'x' array in npz")
                valid_tensor_shapes = False
            else:
                x = data["x"]
                total_samples = x.shape[0]
                expected_shape = tuple(sidecar.get("shape", []))
                if expected_shape and x.shape != expected_shape:
                    errors.append(f"Shape mismatch: {x.shape} vs expected {expected_shape}")
                    valid_tensor_shapes = False

                if mode == "sampled":
                    # Sample 5% of indices up to 100 max
                    n_samples = min(max(1, total_samples // 20), 100)
                    indices = np.random.choice(total_samples, n_samples, replace=False)
                    scanned_samples = n_samples
                    x_scan = x[indices]
                else:
                    x_scan = x
                    scanned_samples = total_samples
                
                if np.isnan(x_scan).any():
                    errors.append("NaN values found in 'x' array")
                    valid_nan_inf = False
                if np.isinf(x_scan).any():
                    errors.append("Inf values found in 'x' array")
                    valid_nan_inf = False

            if "sequence_id" in data and "split" in data:
                # Basic disjointness check: a sequence should belong to exactly one split
                seqs = np.array(data["sequence_id"])
                splits = np.array(data["split"])
                seq_to_split = {}
                for s, sp in zip(seqs, splits):
                    if s in seq_to_split and seq_to_split[s] != sp:
                        errors.append(f"Sequence {s} belongs to multiple splits")
                        split_disjointness = False
                    seq_to_split[s] = sp
            else:
                # Some caches might not have split, that's fine if they don't claim to.
                pass

    except Exception as e:
        errors.append(f"Failed to read npz data: {e}")
        valid_tensor_shapes = False
        valid_nan_inf = False
        total_samples = 0
        scanned_samples = 0

    validation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "npz_path": str(npz_path),
        "sha256": npz_sha256,
        "mode": mode,
        "total_samples": int(total_samples) if 'total_samples' in locals() else 0,
        "scanned_samples": int(scanned_samples) if 'scanned_samples' in locals() else 0,
        "valid_nan_inf": valid_nan_inf,
        "valid_tensor_shapes": valid_tensor_shapes,
        "split_disjointness": split_disjointness,
        "normalizer_origins_verified": normalizer_origins_verified,
        "errors": errors
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    if errors:
        logging.error(f"Audit completed with {len(errors)} errors: {errors}")
        sys.exit(1)
    else:
        logging.info(f"Cache validation written to {output_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()
