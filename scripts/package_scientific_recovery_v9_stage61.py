"""Package essential Stage 61/62 evidence with member SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    short = subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=args.repo, text=True
    ).strip()
    destination = args.repo / f"E_JEPA_TTC_STAGE61_STAGE62_ESSENTIAL_RESULTS_{short}.zip"
    with tempfile.TemporaryDirectory(prefix="stage61-package-") as temporary:
        staging = Path(temporary)
        members: list[tuple[Path, str]] = []
        for path in sorted(args.campaign_root.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".pt":
                members.append(
                    (path, f"artifacts/{path.relative_to(args.campaign_root).as_posix()}")
                )
        for path in (
            args.repo / "CODEX_STAGE61_STAGE62_FINAL_REPORT.md",
            args.repo / "NEXT_DECISION.json",
            args.handoff_root / "SOURCE_PINS.json",
            args.handoff_root / "STAGE61_DECISION_TREE.json",
            args.repo / "configs/protocol/scientific_recovery_v9_stage61_pair_router_x2.json",
        ):
            members.append((path, path.name))
        git_log = subprocess.check_output(
            ["git", "log", "--oneline", "6c9cd1ef5f85c6b9b7fb5c2ccbbdde5c11a39181..HEAD"],
            cwd=args.repo,
            text=True,
        )
        git_stat = subprocess.check_output(
            ["git", "diff", "--stat", "6c9cd1ef5f85c6b9b7fb5c2ccbbdde5c11a39181..HEAD"],
            cwd=args.repo,
            text=True,
        )
        (staging / "GIT_LOG.txt").write_text(git_log, encoding="utf-8")
        (staging / "GIT_DIFF_STAT.txt").write_text(git_stat, encoding="utf-8")
        members.extend(
            (
                (staging / "GIT_LOG.txt", "GIT_LOG.txt"),
                (staging / "GIT_DIFF_STAT.txt", "GIT_DIFF_STAT.txt"),
            )
        )
        checksums = "".join(f"{_sha(path)}  {name}\n" for path, name in members)
        (staging / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
        members.append((staging / "SHA256SUMS.txt", "SHA256SUMS.txt"))
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path, name in members:
                archive.write(path, name)
    digest = _sha(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )
    print(json.dumps({"zip": str(destination), "sha256": digest}))


if __name__ == "__main__":
    main()
