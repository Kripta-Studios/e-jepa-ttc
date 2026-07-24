import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True, help="Path to npz cache")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/subsets"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.05])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache
    if not cache_path.exists():
        logging.error(f"Cache not found: {cache_path}")
        sys.exit(1)

    logging.info(f"Loading cache {cache_path}")
    cache = np.load(cache_path, allow_pickle=False)
    split = cache["split"].astype(str)

    # Only train split
    train_idx = np.where(split == "train")[0]
    if train_idx.size == 0:
        logging.error("No 'train' split found in cache")
        sys.exit(1)

    has_sequence_id = "sequence_id" in cache
    if has_sequence_id:
        seqs = cache["sequence_id"][train_idx]
        unique_seqs = np.sort(np.unique(seqs))

    fractions = sorted(args.fractions, reverse=True)  # e.g. 0.10, 0.05

    for seed in args.seeds:
        previous_indices = None
        previous_frac = None
        for fraction in fractions:
            rng = np.random.default_rng(seed)
            if has_sequence_id:
                subset_idx = []
                for seq in unique_seqs:
                    seq_idx = train_idx[seqs == seq]
                    shuffled = rng.permutation(seq_idx)
                    subset_count = max(1, int(round(seq_idx.size * fraction)))
                    subset_idx.append(shuffled[:subset_count])
                current_indices = np.sort(np.concatenate(subset_idx)).astype(np.int64)
            else:
                shuffled = rng.permutation(train_idx)
                subset_count = max(1, int(round(train_idx.size * fraction)))
                current_indices = np.sort(shuffled[:subset_count]).astype(np.int64)

            current_set = set(current_indices.tolist())

            # Enforce nesting
            if previous_indices is not None:
                if not current_set.issubset(previous_indices):
                    logging.error(
                        f"Nesting failure: fraction {fraction} is not a strict subset of {previous_frac} for seed {seed}"
                    )
                    sys.exit(1)

            previous_indices = current_set
            previous_frac = fraction

            frac_str = f"frac{int(round(fraction * 100))}"
            manifest_path = args.output_dir / f"evttc_{frac_str}_seed{seed}.json"

            payload = {
                "global_indices": current_indices.tolist(),
                "sequence_ids": split[current_indices].tolist() if has_sequence_id else [],
                "ttc_bins": [],
                "fraction": fraction,
                "seed": seed,
            }
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)

            # Compute hash and update
            sha256 = _hash_file(manifest_path)
            payload["sha256"] = sha256
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            logging.info(f"Generated {manifest_path} (n={current_indices.size})")

    logging.info("All subsets generated and nesting parity verified.")


if __name__ == "__main__":
    main()
