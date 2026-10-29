"""Verify bundled evidence and generate a compact, reader-facing source index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(path: Path) -> object:
    """Read an unchanged evidence manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, expected: str) -> None:
    """Reject altered evidence, including under Python's optimized mode."""
    if digest(path) != expected:
        raise ValueError(f"Evidence hash mismatch: {path.relative_to(ROOT)}")


def literal(text: str) -> str:
    """Format an identifier or path with line breaking through xurl."""
    if any(char in text for char in "{}\n\r"):
        raise ValueError("Unsafe TeX source identifier")
    return r"\path{" + text + "}"


def table_start(headers: tuple[str, str, str]) -> list[str]:
    """Use the same compact table dimensions for result and code references."""
    head = " & ".join(headers) + r" \\"
    return [
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.17\linewidth}"
        r">{\raggedright\arraybackslash}p{.56\linewidth}"
        r">{\raggedright\arraybackslash}p{.22\linewidth}@{}}",
        r"\toprule",
        head,
        r"\midrule\endfirsthead",
        r"\toprule",
        head,
        r"\midrule\endhead",
        r"\midrule\multicolumn{3}{r}{\scriptsize Continued on next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]


def table_end() -> list[str]:
    """Close a local table and restore document typography."""
    return [r"\end{longtable}", r"\endgroup"]


def verify_selector(path: Path, selector: str) -> None:
    """Require a public projection selector to resolve in its distributed ledger."""
    if selector == "all projected rows":
        with path.open(encoding="utf-8", newline="") as stream:
            if not next(csv.DictReader(stream), None):
                raise ValueError(f"Empty projected CSV: {path.name}")
        return
    payload = read(path)
    if selector.startswith("/"):
        current = payload
        for token in selector[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            current = current[int(token)] if isinstance(current, list) else current[token]
        if current is None:
            raise ValueError(f"Null projection at {selector}")
        return
    key, separator, value = selector.partition("=")
    if not separator or key not in {"id", "evidence_id"}:
        raise ValueError(f"Unsupported public selector: {selector}")
    pending = [payload]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get(key) == value:
                return
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    raise ValueError(f"Unresolved public selector: {selector}")


def build_private_index() -> None:
    """Verify the 95 primary sources and regenerate only the TeX appendix."""
    entries: list[tuple[str, str, str]] = []
    for source in read(DATA / "early_source_manifest.json"):
        path = DATA / "early_sources" / f"{source['id']}.json"
        verify(path, source["file_sha256"])
        entries.append((source["id"], path.relative_to(ROOT).as_posix(), source["file_sha256"]))
    for source in read(DATA / "middle_manifest.json"):
        verify(ROOT / source["bundled"], source["sha256"])
        entries.append((source["id"], source["bundled"], source["sha256"]))
    for source in read(DATA / "late_results.json")["sources"]:
        path = DATA / source["snapshot"]
        verify(path, source["sha256"])
        entries.append((source["evidence_id"], path.relative_to(ROOT).as_posix(), source["sha256"]))
    if len(entries) != 95 or len({key for key, _, _ in entries}) != len(entries):
        raise ValueError("Expected 95 primary result sources with distinct keys")

    official = read(DATA / "official_benchmark.json")
    for key, path in [
        ("B01", ROOT / "research/official_leaderboard_20261028.json"),
        ("B02", DATA / "official_scoring_scores.json"),
    ]:
        identity = official["provenance"][key]["original_sha256"]
        verify(path, identity)
        entries.append((key, path.relative_to(ROOT).as_posix(), identity))

    lines = [
        r"\section{Evidence index}",
        r"\label{app:evidence}",
        r"Source keys identify the evidence used in the text and tables. "
        r"The build verifies 95 historical snapshots and two official benchmark sources "
        r"against the full SHA-256 "
        r"digests in the manifests. The tables below show 16-character digest "
        r"prefixes; full digests and metric selectors remain in the result ledgers.",
        r"\subsection{Primary result snapshots}",
    ]
    lines.extend(table_start(("Source key", "Bundled file", "SHA-256 prefix")))
    for key, bundled, sha in entries:
        lines.append(f"{literal(key)} & {literal(bundled)} & {literal(sha[:16])} " + r"\\")
    lines.extend(table_end())
    lines.extend(
        [
            r"\subsection{Implementation and protocol references}",
            r"M01--M20 identify code and protocol records at the scientific source "
            r"revision given in Appendix~\ref{app:reproduction}. These repository "
            r"files describe implementation rather than measured outcomes. "
            r"M21 identifies the bundled test receipts. "
            r"The manifest \path{research/architecture_sources.json} contains "
            r"the full source digests and descriptions.",
        ]
    )
    lines.extend(
        table_start(("Source key", "Repository file or bundled receipt", "SHA-256 prefix"))
    )
    for source in read(ROOT / "research/architecture_sources.json"):
        displayed = 0
        for item in source["sources"]:
            path = item["path"]
            if source["id"] == "M21":
                # XML receipts carry the test cases; duplicate console logs are not bundled.
                if Path(path).suffix == ".log":
                    continue
                path = "data/" + Path(path).name
                verify(ROOT / path, item["sha256"])
            key = source["id"] if displayed == 0 else ""
            lines.append(
                f"{literal(key)} & {literal(path)} & {literal(item['sha256'][:16])} " + r"\\"
            )
            displayed += 1
    lines.extend(table_end())
    lines.extend(
        [
            r"\subsection{Verification records and sensor illustration}",
            r"\path{data/verification_manifest.json} records individual test outcomes. "
            r"\path{research/late_verification.json} records checks of result extraction "
            r"and arithmetic. The source manifests preserve full provenance identities.",
            r"The FCWD1:0007 illustration is bundled as "
            r"\path{figures/fcwd_example.png}. Its source identity, full digest and "
            r"byte-identical-copy status appear in \path{data/visual_source_manifest.json}. "
            r"The supplied scripts regenerate the remaining figures.",
        ]
    )
    visual = read(DATA / "visual_source_manifest.json")
    verify(ROOT / visual["file"], visual["sha256"])
    (ROOT / "sections/evidence_index.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"verified_primary_snapshots": len(entries), "index": "sections/evidence_index.tex"}
        )
    )


def build_public_index(*, write: bool = True) -> None:
    """Verify distributed derived data without implying access to original sources."""
    registry = read(DATA / "public_evidence_registry.json")
    if registry.get("schema") != "phase_ttc_public_evidence_registry_v1":
        raise ValueError("Unsupported public evidence registry schema")
    records = registry["records"]
    implementation = list(registry.get("implementation_records", []))
    results = []
    for record in records:
        if record.get("evidence_kind") == "implementation":
            implementation.append(record)
        else:
            results.append(record)
    all_records = results + implementation
    if len({record["evidence_id"] for record in all_records}) != len(all_records):
        raise ValueError("Duplicate public evidence identifier")
    if len(results) != 99 or not {"B01", "B02", "T01", "T02"}.issubset(
        {r["evidence_id"] for r in results}
    ):
        raise ValueError(
            "Expected 95 historical, two official benchmark, and two competition/literature sources"
        )
    files = registry["public_files"]
    for item in files:
        verify(ROOT / item["path"], item["sha256"])
    public_paths = {item["path"] for item in files}
    for record in results:
        if not isinstance(record.get("projection_distributed"), bool):
            raise ValueError(f"Missing projection availability for {record['evidence_id']}")
        if record["projection_distributed"] and record["public_ledger"] not in public_paths:
            raise ValueError(f"Unverified public ledger for {record['evidence_id']}")
        if record["projection_distributed"]:
            verify_selector(ROOT / record["public_ledger"], record["public_selector"])
        if not record["projection_distributed"] and (
            record.get("public_ledger") is not None or record.get("public_selector") is not None
        ):
            raise ValueError(f"Unprojected source has a public mapping: {record['evidence_id']}")
    for record in implementation:
        if record["public_ledger"] not in public_paths:
            raise ValueError(f"Unverified implementation manifest for {record['evidence_id']}")
        verify_selector(ROOT / record["public_ledger"], record["public_selector"])
    lines = [
        r"\section{Evidence index}",
        r"\label{app:evidence}",
        r"The source keys below identify original result artifacts and indicate "
        r"which have a projection in the distributed derived ledgers. "
        r"The public ledgers reproduce the included tables, figures and selected "
        r"arithmetic checks. The original SHA-256 prefixes identify retained "
        r"source artifacts. Historical snapshots and the original leaderboard "
        r"response are outside this archive; B02 distributes the original scorer JSON. "
        r"The public build verifies the "
        r"derived-data hashes listed in \path{data/public_evidence_registry.json}. "
        r"That registry contains full digests and selectors for the available "
        r"projections. Rows marked `Digest only' have no distributed data projection.",
        r"\subsection{Result-source identities}",
    ]
    lines.extend(table_start(("Source key", "Distributed ledger", "Original SHA-256")))
    for record in results:
        sha = record["original_sha256"]
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError(f"Invalid original digest for {record['evidence_id']}")
        location = (
            literal(record["public_ledger"])
            if record["projection_distributed"]
            else r"\emph{Digest only}"
        )
        lines.append(
            f"{literal(record['evidence_id'])} & {location} & {literal(sha[:16])} " + r"\\"
        )
    lines.extend(table_end())
    if implementation:
        lines.extend(
            [
                r"\subsection{Implementation and verification references}",
                r"These keys refer to implementation descriptions and the scope of "
                r"selected checks. The distributed manifests retain source revision; "
                r"a code reference does not establish an experimental outcome.",
            ]
        )
        lines.extend(table_start(("Source key", "Distributed manifest", "Ledger selector")))
        for record in implementation:
            selector = str(record.get("public_selector", record["evidence_id"]))
            lines.append(
                f"{literal(record['evidence_id'])} & {literal(record['public_ledger'])} & "
                f"{literal(selector)} " + r"\\"
            )
        lines.extend(table_end())
    lines.extend(
        [
            r"\needspace{14\baselineskip}",
            r"\subsection{Distributed data integrity}",
            r"These digest prefixes refer to the distributed files themselves. "
            r"The build checks their full hashes before regenerating the manuscript.",
        ]
    )
    lines.extend(table_start(("File ID", "Distributed file", "SHA-256 prefix")))
    for number, item in enumerate(files, 1):
        lines.append(
            f"D{number:02d} & {literal(item['path'])} & {literal(item['sha256'][:16])} " + r"\\"
        )
    lines.extend(table_end())
    if write:
        (ROOT / "sections/evidence_index.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "public_mode": True,
                "verified_public_files": len(files),
                "original_result_identities": len(results),
                "original_source_bytes_verified": False,
                "index": "sections/evidence_index.tex",
            }
        )
    )


def main() -> None:
    """Choose an explicit public build or the original full-evidence audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public", action="store_true", help="Verify distributed derived data only"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the public index without rewriting its prose",
    )
    args = parser.parse_args()
    if args.public:
        build_public_index(write=not args.verify_only)
    else:
        build_private_index()


if __name__ == "__main__":
    main()
