"""Copy the completed local comparison and its public references with hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    """Hash the actual saved bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts/final_train40_transfer_20261027"
    report = root / "docs/final_train40_transfer_20261027"
    results = json.loads((report / "RESULTS.json").read_text(encoding="utf-8"))
    if results["status"] != "COMPLETE":
        raise ValueError("The complete paired report is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copies: dict[str, dict[str, str | int]] = {}

    def copy(source: Path, relative: Path) -> None:
        target = output / relative
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != expected:
            shutil.copy2(source, target)
        if digest(target) != expected:
            raise ValueError(f"Local copy differs: {relative}")
        copies[relative.as_posix()] = {
            "source": str(source),
            "sha256": expected,
            "bytes": target.stat().st_size,
        }

    for directory, prefix in ((artifacts, "evidence"), (report, "report")):
        for source in sorted(directory.rglob("*")):
            if source.is_file():
                copy(source, Path(prefix) / source.relative_to(directory))
    for pattern in (
        "scripts/*final_train40_transfer*.py",
        "tests/unit/test_final_train40_transfer*.py",
    ):
        for source in root.glob(pattern):
            copy(source, Path("code") / source.relative_to(root))
    for name in ("post_v18_transfer_report.py", "post_v18_transfer_inputs.py"):
        copy(root / "scripts" / name, Path("code/scripts") / name)
    for number, (name, expected) in enumerate(sorted(results["label_sha256"].items())):
        source = Path(name)
        if digest(source) != expected:
            raise ValueError("Public TTC reference changed since scoring")
        copy(source, Path("public_gt") / f"{number:02d}" / source.name)
    binding = json.loads((artifacts / "garl/INFERENCE_FREEZE.json").read_text())
    public = Path(binding["public_dir"])
    for relative in ("paper_ours_full.pth", "configs/ablation/ours_full.yaml"):
        copy(public / relative, Path("garl_release") / relative)
    native = Path(binding["native_code_root"])
    for name in ("LICENSE", "README.md", "garl_ttc/__init__.py"):
        if (native / name).is_file():
            copy(native / name, Path("garl_code") / name)
    for relative, expected in binding["native_source_sha256"].items():
        source = native / relative
        if digest(source) != expected:
            raise ValueError("Bound official source changed")
        copy(source, Path("garl_code") / relative)
    receipt = {
        "status": "LOCAL_TRANSFER_PRESERVATION_COMPLETE",
        "files": copies,
        "bytes": sum(int(item["bytes"]) for item in copies.values()),
        "own_candidate": "../FINAL_CANDIDATE",
        "optimizer_updates": 0,
        "external_submission": False,
    }
    (output / "TRANSFER_PRESERVATION.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"status": receipt["status"], "files": len(copies), "bytes": receipt["bytes"]})
    )


if __name__ == "__main__":
    main()
