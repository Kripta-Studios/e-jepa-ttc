"""Generate current complete companion Pyright and PowerShell parse evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted


def sha256(path: Path) -> str:
    """Hash a source file without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_powershell(paths: list[Path]) -> tuple[str, list[dict]]:
    """Parse every named script using the real PowerShell AST parser."""
    quoted = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    program = (
        "$rows=foreach($name in @(" + quoted + ")) {"
        "$tokens=$null;$errors=$null;"
        "$null=[System.Management.Automation.Language.Parser]::ParseFile($name,[ref]$tokens,[ref]$errors);"
        "[pscustomobject]@{path=$name;errors=@($errors|ForEach-Object {$_.Message})}};"
        "[pscustomobject]@{version=$PSVersionTable.PSVersion.ToString();rows=@($rows)}"
        "|ConvertTo-Json -Depth 6 -Compress"
    )
    run = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", program],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )
    if run.returncode:
        raise RuntimeError(f"PowerShell AST parser exited {run.returncode}: {run.stderr}")
    parsed = json.loads(run.stdout)
    rows = parsed["rows"]
    if len(rows) != len(paths) or [Path(row["path"]) for row in rows] != paths:
        raise ValueError("PowerShell AST report omitted or reordered source files")
    return parsed["version"], rows


def main() -> None:
    """Write real static reports under new T0 paths; no fit or optimizer step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--types-output", type=Path, required=True)
    parser.add_argument("--powershell-output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    work = args.worktree.resolve(strict=True)
    baseline = args.baseline_source.resolve(strict=True)
    python = args.python.resolve(strict=True)
    outputs = [args.types_output.resolve(), args.powershell_output.resolve()]
    if (
        args.other_reserved_bytes < 0
        or not baseline.is_relative_to(work / "artifacts/simplex_t/T0")
        or any(
            path.exists() or not path.is_relative_to(work / "artifacts/simplex_t/T0")
            for path in outputs
        )
    ):
        raise ValueError("new bounded T0 QA outputs and preserved baseline required")
    snapshot = admitted([work])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 134_217_728
    ):
        raise InterruptedError("RESOURCE_PAUSE before current static reports")
    typed = sorted((work / "src/e_jepa_ttc/simplex_t").rglob("*.py")) + sorted(
        (work / "scripts").glob("*simplex*.py")
    )
    typed = sorted(set(typed))
    command = [
        str(python),
        "-B",
        "-m",
        "pyright",
        "--outputjson",
        *[str(p.relative_to(work)) for p in typed],
    ]
    run = subprocess.run(
        command,
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )
    result = json.loads(run.stdout)
    type_record = {
        "status": "STATIC_TYPES_ONLY_NOT_SCIENTIFIC_ADMISSION",
        "exit_code": run.returncode,
        "optimizer_updates": 0,
        "source_sha256": {p.relative_to(work).as_posix(): sha256(p) for p in typed},
        "command": command,
        "stderr": run.stderr,
        "result": result,
    }
    write_new_json(outputs[0], type_record)
    runs = []
    version = ""
    for root in (work, baseline):
        paths = sorted(
            path
            for folder in ("src", "scripts")
            for path in (root / folder).rglob("*")
            if path.is_file() and path.suffix.lower() in {".ps1", ".psm1"}
        )
        version, rows = parse_powershell(paths)
        runs.append(
            {
                "root": str(root),
                "files": [
                    {"path": str(path), "sha256": sha256(path), "errors": row["errors"]}
                    for path, row in zip(paths, rows, strict=True)
                ],
            }
        )
    write_new_json(
        outputs[1],
        {
            "status": "POWERSHELL_AST_ONLY_NOT_SCRIPT_EXECUTION",
            "powershell_version": version,
            "runs": runs,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
