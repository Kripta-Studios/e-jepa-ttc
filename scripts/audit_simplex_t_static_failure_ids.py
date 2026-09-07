"""Compare whole-repository Ruff failure IDs with the starting source, without test execution."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted

BASELINE = "57b39cb2b9a5ec8378755f8f350634c007aa822e"


def failure_ids(diagnostics: list[dict], root: Path) -> dict[str, dict]:
    """Use relative path, rule, message and source position as reproducible IDs."""
    result = {}
    for item in diagnostics:
        path = Path(item["filename"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Ruff diagnostic outside audited root")
        relative = path.relative_to(root).as_posix()
        location = item["location"]
        key = f"{relative}:{location['row']}:{location['column']}:{item['code']}:{item['message']}"
        if key in result:
            raise ValueError("duplicate static failure ID")
        result[key] = dict(
            path=relative, code=item["code"], message=item["message"], location=location
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--ruff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    work = args.worktree.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("new companion T0 output required")
    if args.other_reserved_bytes < 0:
        raise ValueError("nonnegative pending output reservation required")
    snapshot = admitted([work])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 134_217_728
    ):
        raise InterruptedError("RESOURCE_PAUSE before static QA")
    started = time.monotonic()
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(work),
            "archive",
            "--format=zip",
            BASELINE,
            "src",
            "scripts",
            "tests",
            "pyproject.toml",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    if len(archive) > 134_217_728:
        raise ValueError("baseline source archive exceeds 128 MiB bound")
    output.mkdir(parents=True)
    baseline = output / "baseline_source"
    baseline.mkdir()
    extracted = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        entries = [
            info
            for info in bundle.infolist()
            if not info.is_dir()
            and (
                Path(info.filename).suffix.lower() in {".py", ".ps1", ".psm1"}
                or info.filename == "pyproject.toml"
            )
        ]
        if sum(info.file_size for info in entries) > 134_217_728:
            raise ValueError("expanded baseline sources exceed bound")
        for info in entries:
            relative = Path(info.filename)
            path = (baseline / relative).resolve()
            if relative.is_absolute() or relative.drive or not path.is_relative_to(baseline):
                raise ValueError("baseline archive path escapes QA snapshot")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(bundle.read(info))
            extracted.append({"path": relative.as_posix(), "sha256": sha256(path)})
    current_commit = (
        subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"]).decode().strip()
    )
    tool = args.ruff.resolve(strict=True)
    config = work / "pyproject.toml"
    config_hash = sha256(config)
    current_files = {
        path.relative_to(work).as_posix(): sha256(path)
        for folder in ("src", "scripts", "tests")
        for path in (work / folder).rglob("*.py")
    }
    records, ids = {}, {}
    for name, root in (("baseline", baseline), ("current", work)):
        run = subprocess.run(
            [
                str(tool),
                "check",
                "--no-cache",
                "--output-format=json",
                "--config",
                str(config),
                "src",
                "scripts",
                "tests",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if run.returncode not in {0, 1}:
            raise RuntimeError(
                f"Ruff execution failed: {run.stderr.decode('utf-8', errors='replace')}"
            )
        diagnostics = json.loads(run.stdout)
        ids[name] = failure_ids(diagnostics, root)
        write_new_json(output / f"{name.upper()}_RUFF.json", {"diagnostics": diagnostics})
        records[name] = {"exit_code": run.returncode, "failures": len(ids[name])}
    if sha256(config) != config_hash:
        raise ValueError("static QA rules changed during comparison")
    observed_files = {
        path.relative_to(work).as_posix(): sha256(path)
        for folder in ("src", "scripts", "tests")
        for path in (work / folder).rglob("*.py")
    }
    if current_files != observed_files:
        raise ValueError("static QA source inventory changed during comparison")
    before, after = set(ids["baseline"]), set(ids["current"])
    write_new_json(
        output / "COMPARISON.json",
        {
            "status": "STATIC_FAILURE_IDS_COMPARED_NOT_FULL_QA_OR_SCIENTIFIC_ADMISSION",
            "baseline_commit": BASELINE,
            "current_commit": current_commit,
            "common_rule_config_sha256": config_hash,
            "ruff_sha256": sha256(tool),
            "baseline_source_files": extracted,
            "current_source_sha256": current_files,
            "runs": records,
            "persistent_failure_ids": sorted(before & after),
            "new_or_relocated_failure_ids": sorted(after - before),
            "resolved_or_relocated_failure_ids": sorted(before - after),
            "position_sensitive_ids": True,
            "tests_executed": 0,
            "optimizer_updates": 0,
            "scientific_admission": False,
            "admission_resources": snapshot,
            "observed_seconds": time.monotonic() - started,
            "not_covered": [
                "pytest",
                "types",
                "PowerShell AST",
                "real replay and source integration",
            ],
        },
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
