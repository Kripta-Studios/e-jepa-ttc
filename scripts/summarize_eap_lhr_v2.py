"""Collect unattended LHR-v2 run summaries into one handoff artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    experiments_root = args.root / "experiments"
    if experiments_root.exists():
        for summary_path in sorted(experiments_root.glob("L*/seed-*/summary.json")):
            relative = summary_path.parent.relative_to(experiments_root).as_posix()
            summary = _load(summary_path)
            if summary is not None:
                arms[relative] = summary
    zero_shot = (
        {path.stem: _load(path) for path in sorted((args.root / "metrics").glob("*.json"))}
        if (args.root / "metrics").exists()
        else {}
    )
    result = {
        "artifact_type": "eap_lhr_v2_unattended_handoff",
        "run_root": args.root.as_posix(),
        "arms": arms,
        "metrics": zero_shot,
        "recommended_files_to_share": [
            "run_manifest.json",
            "pipeline_summary.json",
            "logs/*.log",
            "cache_audit.json",
            "official_cache/manifest.json",
            "L*/summary.json",
            "L*/history.jsonl",
            "metrics/*.json",
            "git_diff.patch",
            "pytest*.txt",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
