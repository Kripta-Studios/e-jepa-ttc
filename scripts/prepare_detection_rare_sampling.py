"""Build an immutable D4096 rare-sampling intervention from train metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler  # noqa: E402
from e_jepa_ttc.detection.rare_sampling import make_schedule  # noqa: E402


def main() -> None:
    """Create a fixed-budget calendar and bind it to its manifest and population audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tokens = manifest["roles"]["train"]["sample_tokens"]
    wanted = set(tokens)
    rows = {}
    with args.population.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_token"] in wanted:
                if row["sample_token"] in rows:
                    raise ValueError("Duplicated train population token")
                rows[row["sample_token"]] = row
    baseline_sampler = ResumablePermutationSampler(len(tokens), args.seed)
    baseline = [
        i for _ in range(args.updates) for i in baseline_sampler.next_batch(args.batch_size)
    ]
    plan = make_schedule(tokens, rows, baseline, seed=args.seed)
    plan["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    with args.population.open("rb") as stream:
        plan["population_file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    plan["split_sha256"] = manifest["candidate_split_sha256"]
    plan["updates"], plan["batch_size"] = args.updates, args.batch_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps(plan["audit"] | {"sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    )


if __name__ == "__main__":
    main()
