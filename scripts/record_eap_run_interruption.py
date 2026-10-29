"""Record an operator interruption without fabricating training metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def record_interruption(
    output_dir: Path,
    *,
    command: str,
    reason: str,
    process_ids: list[int],
) -> Path:
    """Write a non-overwriting, factual interruption artifact for a run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "FAILURE.json"
    if failure_path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        failure_path = output_dir / f"FAILURE_{stamp}.json"
    history_path = output_dir / "history.jsonl"
    metrics_path = output_dir / "metrics.json"
    checkpoint_paths = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.glob("*.pt")
        if path.is_file()
    )
    payload = {
        "artifact_type": "eap_jepa_training_failure_v1",
        "status": "interrupted",
        "error_type": "OperatorInterruption",
        "error_message": reason,
        "command": command,
        "process_ids": process_ids,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "training_started": True,
        "epochs_completed": sum(
            1 for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if history_path.is_file()
        else 0,
        "history_path": history_path.as_posix(),
        "history_bytes": history_path.stat().st_size if history_path.is_file() else 0,
        "metrics_available": metrics_path.is_file() and metrics_path.stat().st_size > 0,
        "checkpoint_paths": checkpoint_paths,
        "metrics_path": metrics_path.as_posix(),
        "output": output_dir.as_posix(),
    }
    failure_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return failure_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--process-id", type=int, action="append", default=[])
    args = parser.parse_args()
    path = record_interruption(
        args.output_dir,
        command=args.command,
        reason=args.reason,
        process_ids=args.process_id,
    )
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
