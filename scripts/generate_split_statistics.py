import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate split statistics from a dataset manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output statistics JSON")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", [])

    stats = {
        "dataset": manifest.get("dataset", "Unknown"),
        "total_samples": manifest.get("total_samples", 0),
        "total_size_bytes": manifest.get("total_size_bytes", 0),
        "splits": defaultdict(lambda: {"sequences": 0, "samples": 0, "size_bytes": 0}),
    }

    sequence_ids = set()
    for s in splits:
        split_name = s.get("split", "unknown")
        seq_id = s.get("sequence_id", "unknown")

        sequence_ids.add(seq_id)
        stats["splits"][split_name]["samples"] += s.get("samples", 0)
        stats["splits"][split_name]["size_bytes"] += s.get("size_bytes", 0)

    # count unique sequences per split
    # Since sequence_id is attached to files, we can just recount unique sequences per split
    split_seqs = defaultdict(set)
    for s in splits:
        split_name = s.get("split", "unknown")
        seq_id = s.get("sequence_id", "unknown")
        split_seqs[split_name].add(seq_id)

    for split_name, seqs in split_seqs.items():
        stats["splits"][split_name]["sequences"] = len(seqs)

    stats["total_sequences"] = len(sequence_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Split statistics written to {args.output}")


if __name__ == "__main__":
    main()
