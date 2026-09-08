"""Connect verified terminal phase references to resumable T6 postprocessing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def main() -> int:
    """Delegate analysis and archive verification; never treat metadata as results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--reconciliation-sha256", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--materializer-sha256", required=True)
    parser.add_argument("--attempt-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    root = args.campaign_root.resolve()
    if root == work / "artifacts" or not root.is_relative_to(work / "artifacts"):
        raise ValueError("campaign must remain inside companion artifacts")
    if args.verify_only and args.resume:
        parser.error("verification must not request execution")
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 8_388_608:
        parser.error("nonnegative other and at least 8388608 own reservation required")
    scripts = work / "scripts"
    materializer = scripts / "materialize_simplex_t_delivery.py"
    attempt = scripts / "execute_simplex_t_postprocessing_attempt.py"
    pins = {
        materializer: args.materializer_sha256,
        attempt: args.attempt_sha256,
        scripts / "postprocess_simplex_t_campaign.py": args.worker_sha256,
        args.reconciliation: args.reconciliation_sha256,
    }

    def check_pins() -> None:
        for path, digest in pins.items():
            if path.stat().st_size > 1_048_576 or sha256(path) != digest:
                raise ValueError("delivery orchestration input changed")

    check_pins()
    metadata = [
        sys.executable,
        "-B",
        str(materializer),
        "--campaign-root",
        str(root),
        "--run-root",
        str(args.run_root),
        "--reconciliation",
        str(args.reconciliation),
        "--reconciliation-sha256",
        args.reconciliation_sha256,
        "--analysis-commit",
        args.analysis_commit,
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
    ]
    if args.verify_only:
        metadata.append("--verify-only")
    code = subprocess.run(metadata, check=False, cwd=work).returncode
    if code:
        return code
    check_pins()
    template = root / "launches/T6.json"
    command = [
        sys.executable,
        "-B",
        str(attempt),
        "--template",
        str(template),
        "--template-sha256",
        sha256(template),
        "--worker-sha256",
        args.worker_sha256,
        "--attempt-root",
        str(root / "T6"),
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
        "--own-reserved-bytes",
        str(args.own_reserved_bytes),
    ]
    if args.verify_only:
        command.append("--verify-only")
    elif args.resume:
        command.append("--resume")
    return subprocess.run(command, check=False, cwd=work).returncode


if __name__ == "__main__":
    raise SystemExit(main())
