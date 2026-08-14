"""Keep V8 Python changes clean without rewriting the historical Ruff backlog.

The signed baseline is an engineering-provenance artifact, not an experimental
result.  It records the known repository-wide Ruff diagnostics at the V8 base
commit; normal runs reject additions and require changed Python files to be
both lint-clean and already formatted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash

DEFAULT_BASELINE = Path("configs/quality/ruff_baseline_v8.json")
DEFAULT_BASE_REF = "f9331b29596c4107430af5a8c78935bd127ccf94"
SCHEMA_VERSION = 1


class QualityGateError(RuntimeError):
    """Raised when the quality gate cannot establish a trustworthy result."""


@dataclass(frozen=True, order=True)
class Violation:
    """A repository-relative, line-independent Ruff diagnostic count."""

    path: str
    code: str
    message: str
    count: int

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the identity used when comparing diagnostic multiplicities."""
        return (self.path, self.code, self.message)

    def as_json(self) -> dict[str, str | int]:
        """Return the deterministic artifact representation."""
        return {
            "path": self.path,
            "rule_code": self.code,
            "message": self.message,
            "count": self.count,
        }


def _run(
    command: Sequence[str], cwd: Path, *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise QualityGateError(f"Required executable is unavailable: {command[0]!r}.") from error


def repository_root(start: Path) -> Path:
    """Return the Git root for *start*, or raise an actionable error."""
    result = _run(["git", "rev-parse", "--show-toplevel"], start)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "not a Git work tree"
        raise QualityGateError(f"Cannot locate repository root with git: {detail}.")
    return Path(result.stdout.strip()).resolve()


def _git_lines(root: Path, arguments: Sequence[str], purpose: str) -> list[str]:
    result = _run(["git", *arguments], root)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git failure"
        raise QualityGateError(f"Cannot {purpose}: {detail}.")
    return [line for line in result.stdout.splitlines() if line]


def changed_python_files(root: Path, base_ref: str) -> list[Path]:
    """Return existing changed or untracked Python files, relative to *root*."""
    changed = _git_lines(
        root,
        ["diff", "--name-only", "--diff-filter=ACMR", base_ref, "--"],
        f"compare the work tree with base ref {base_ref!r}",
    )
    untracked = _git_lines(
        root,
        ["ls-files", "--others", "--exclude-standard"],
        "list untracked files",
    )
    paths = {
        Path(*name.split("/")) for name in [*changed, *untracked] if Path(name).suffix == ".py"
    }
    return sorted(
        (path for path in paths if (root / path).is_file()), key=lambda path: path.as_posix()
    )


def _relative_path(path_value: object, root: Path) -> str:
    if not isinstance(path_value, str):
        raise QualityGateError("Ruff JSON diagnostic is missing a string filename.")
    candidate = Path(path_value)
    absolute = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError as error:
        raise QualityGateError(
            f"Ruff reported a file outside the repository: {path_value!r}."
        ) from error


def normalize_ruff_diagnostics(payload: object, root: Path) -> list[Violation]:
    """Convert Ruff JSON output into sorted, path-stable violation records."""
    if not isinstance(payload, list):
        raise QualityGateError("Ruff did not return a JSON list of diagnostics.")
    counts: Counter[tuple[str, str, str]] = Counter()
    for item in payload:
        if not isinstance(item, dict):
            raise QualityGateError("Ruff returned a malformed diagnostic entry.")
        filename = _relative_path(item.get("filename"), root)
        code = item.get("code")
        message = item.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise QualityGateError("Ruff JSON diagnostic is missing its rule code or message.")
        counts[(filename, code, message)] += 1
    return [
        Violation(path, code, message, count)
        for (path, code, message), count in sorted(counts.items())
    ]


def ruff_diagnostics(root: Path) -> list[Violation]:
    """Run Ruff's lint checker via the active Python interpreter."""
    result = _run([sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"], root)
    if result.returncode not in (0, 1):
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Ruff failure"
        raise QualityGateError(f"Ruff check could not run: {detail}.")
    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise QualityGateError("Ruff check did not emit valid JSON diagnostics.") from error
    return normalize_ruff_diagnostics(payload, root)


def ruff_version(root: Path) -> str:
    """Return the version of Ruff invoked from the active Python environment."""
    result = _run([sys.executable, "-m", "ruff", "--version"], root)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Ruff failure"
        raise QualityGateError(f"Could not determine Ruff version: {detail}.")
    return result.stdout.strip()


def load_baseline(path: Path) -> list[Violation]:
    """Load and verify a signed quality baseline artifact."""
    if not path.is_file():
        raise QualityGateError(
            f"Quality baseline does not exist: {path}. Run with --write-baseline."
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualityGateError(f"Cannot read quality baseline {path}: {error}.") from error
    if not isinstance(raw, dict):
        raise QualityGateError("Quality baseline must contain a JSON object.")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise QualityGateError(
            f"Unsupported quality baseline schema: {raw.get('schema_version')!r}."
        )
    if not verify_artifact_hash(raw):
        raise QualityGateError(
            "Quality baseline artifact_sha256 does not match its canonical contents."
        )
    records = raw.get("normalized_violations")
    if not isinstance(records, list):
        raise QualityGateError("Quality baseline is missing normalized_violations.")
    violations: list[Violation] = []
    for record in records:
        if not isinstance(record, dict):
            raise QualityGateError("Quality baseline contains a malformed violation record.")
        path_value = record.get("path")
        code = record.get("rule_code")
        message = record.get("message")
        count = record.get("count")
        if (
            not isinstance(path_value, str)
            or not isinstance(code, str)
            or not isinstance(message, str)
            or not isinstance(count, int)
            or count < 1
        ):
            raise QualityGateError("Quality baseline contains an invalid normalized violation.")
        violations.append(Violation(path_value, code, message, count))
    return sorted(violations)


def baseline_payload(root: Path, base_ref: str, violations: list[Violation]) -> dict[str, Any]:
    """Build a signed, deterministic JSON artifact for the current Ruff backlog."""
    commit = _git_lines(
        root, ["rev-parse", "--verify", f"{base_ref}^{{commit}}"], "resolve base ref"
    )[0]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_against_git_commit": commit,
        "ruff_version": ruff_version(root),
        "normalized_violations": [violation.as_json() for violation in violations],
    }
    return sign_artifact(payload)


def _new_violations(current: list[Violation], baseline: list[Violation]) -> list[Violation]:
    baseline_counts = {violation.key: violation.count for violation in baseline}
    additions: list[Violation] = []
    for violation in current:
        excess = violation.count - baseline_counts.get(violation.key, 0)
        if excess > 0:
            additions.append(Violation(*violation.key, excess))
    return additions


def _changed_files_clean(root: Path, paths: list[Path]) -> tuple[bool, str]:
    if not paths:
        return True, "No changed or untracked Python files."
    names = [path.as_posix() for path in paths]
    check = _run([sys.executable, "-m", "ruff", "check", *names], root)
    formatted = _run([sys.executable, "-m", "ruff", "format", "--check", *names], root)
    messages: list[str] = []
    if check.returncode != 0:
        messages.append(
            "Ruff check failed for changed Python files:\n" + check.stdout + check.stderr
        )
    if formatted.returncode != 0:
        messages.append(
            "Ruff format --check failed for changed Python files:\n"
            + formatted.stdout
            + formatted.stderr
        )
    return not messages, "\n".join(messages)


def run_gate(root: Path, baseline_path: Path, base_ref: str, *, write_baseline: bool) -> int:
    """Execute the baseline lifecycle or normal no-new-debt quality gate."""
    root = repository_root(root)
    if write_baseline:
        payload = baseline_payload(root, base_ref, ruff_diagnostics(root))
        output_path = baseline_path if baseline_path.is_absolute() else root / baseline_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            display_path = output_path.relative_to(root).as_posix()
        except ValueError:
            display_path = str(output_path)
        print(f"Wrote signed Ruff baseline to {display_path}.")
        return 0

    resolved_baseline = baseline_path if baseline_path.is_absolute() else root / baseline_path
    baseline = load_baseline(resolved_baseline)
    additions = _new_violations(ruff_diagnostics(root), baseline)
    changed = changed_python_files(root, base_ref)
    clean_changed, changed_detail = _changed_files_clean(root, changed)
    if additions or not clean_changed:
        if additions:
            print("New Ruff violations beyond the signed historical baseline:", file=sys.stderr)
            for violation in additions:
                print(
                    f"  {violation.path}: {violation.code} {violation.message} "
                    f"(new count: {violation.count})",
                    file=sys.stderr,
                )
        if not clean_changed:
            print(changed_detail.rstrip(), file=sys.stderr)
        return 1
    print(
        "Quality baseline gate passed "
        f"({len(baseline)} historical violation groups; "
        f"{len(changed)} changed Python files checked)."
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the quality gate's command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)
    parser.add_argument("--write-baseline", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and render actionable setup or gate errors."""
    args = parse_args(argv)
    try:
        return run_gate(
            Path.cwd(), args.baseline, args.base_ref, write_baseline=args.write_baseline
        )
    except QualityGateError as error:
        print(f"quality baseline gate error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
