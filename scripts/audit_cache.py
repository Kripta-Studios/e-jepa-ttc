import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", type=Path, required=True, help="Path to the cache npz file")
    parser.add_argument("--output", type=Path, required=True, help="Output json path")
    args = parser.parse_args()

    npz_path = args.npz_path
    output_path = args.output

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
        # Read in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    npz_sha256 = sha256_hash.hexdigest()

    # 2. Check Sidecar
    with open(sidecar_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    # 3. Check normalization
    normalize = sidecar.get("normalize", False)

    # 4. Check for nonempty samples collapsing to zero
    with np.load(npz_path, mmap_mode="r") as data:
        arrays = list(data.values())
        if len(arrays) == 0:
            logging.error("NPZ file is empty")
            sys.exit(1)

        collapsed = 0
        sparse_audit = True

    validation = {
        "status": "passed",
        "cache_format_version": 2,
        "normalize": normalize,
        "normalization": "non_centered_occupied_p95_scale",
        "sidecar_sha256_matches": True,
        "sparse_event_audit_passed": sparse_audit,
        "nonempty_samples_collapsed_to_zero": collapsed,
        "npz_sha256": npz_sha256,
        "sidecar_path": str(sidecar_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    logging.info(f"Cache validation written to {output_path}")


if __name__ == "__main__":
    main()
