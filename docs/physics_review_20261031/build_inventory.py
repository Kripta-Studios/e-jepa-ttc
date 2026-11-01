"""Rebuild the read-only Git/document inventory for the scientific review."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SINCE = "2026-04-30T00:00:00+02:00"
UNTIL = "2026-11-01T00:00:00+01:00"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.quotepath=false", *args], cwd=ROOT,
        encoding="utf-8", errors="strict",
    )


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    (OUT / "git").mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "HEAD").strip()
    raw = git("log", "--all", f"--since={SINCE}", f"--until={UNTIL}",
              "--date-order", "--format=%x1e%H%x1f%aI%x1f%cI%x1f%P%x1f%B%x1f", "--numstat")
    commits = []
    for block in raw.split("\x1e")[1:]:
        sha, author_date, commit_date, parents, body, stats = block.split("\x1f", 5)
        paths = []
        for line in stats.strip().splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                paths.append(dict(added=parts[0], deleted=parts[1], path=parts[2]))
        commits.append(dict(sha=sha, author_date=author_date, commit_date=commit_date,
                            parents=parents.split(), subject=body.strip().splitlines()[0],
                            body=body.strip(), paths=paths))
    dump("git/commits.json", commits)
    head_shas = set(git("rev-list", "HEAD").splitlines())
    by_month: dict[str, list[dict]] = collections.defaultdict(list)
    for row in commits:
        by_month[row["author_date"][:7]].append(row)
    for month, rows in sorted(by_month.items()):
        lines = [f"# Historial Git {month}", "", f"{len(rows)} commits alcanzables en las referencias locales inventariadas.", "",
                 "Fechas de autor; la selección temporal de Git usa fechas de commit. Un commit registra código/documentos, no certifica una ejecución.", "",
                 "Los merges pueden carecer de numstat. El JSON conserva cuerpo, padres y rutas completas.", ""]
        for row in sorted(rows, key=lambda r: (r["author_date"], r["sha"])):
            sha = row["sha"]
            lines.extend([f"## {row['author_date']} · {sha[:10]}", "",
                          f"[{row['subject']}](https://github.com/Kripta-Studios/phase-ttc/commit/{sha})", "",
                          f"Alcanzable desde HEAD: {'sí' if sha in head_shas else 'no'}. Padres: {', '.join(x[:10] for x in row['parents']) or 'raíz'}.", ""])
            if row["body"] != row["subject"]:
                lines.extend(["```text", row["body"], "```", ""])
            if row["paths"]:
                lines.extend(["| Añadidas | Eliminadas | Ruta |", "|---:|---:|---|"])
                for path in row["paths"]:
                    lines.append(f"| {path['added']} | {path['deleted']} | `{path['path']}` |")
                lines.append("")
        (OUT / "git" / f"{month}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    file_rows = []
    tracked = git("ls-files", "-z").split("\0")
    for name in tracked:
        path = ROOT / name
        if not name or not path.is_file() or name.startswith("docs/physics_review_20261031/"):
            continue
        if path.suffix not in {".md", ".tex", ".bib", ".json", ".csv", ".yaml", ".yml"}:
            continue
        data = path.read_bytes()
        text = data.decode("utf-8-sig", errors="replace")
        headings = re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE) if path.suffix == ".md" else []
        if path.suffix == ".tex":
            headings = re.findall(r"\\(?:sub)*section\{([^}]+)\}", text)
        file_rows.append(dict(path=name, suffix=path.suffix, bytes=len(data), lines=len(text.splitlines()),
                              sha256=hashlib.sha256(data).hexdigest(), headings=headings))
    dump("corpus_inventory.json", file_rows)
    with (OUT / "corpus_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "suffix", "bytes", "lines", "sha256"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in file_rows)
    docs = [r for r in file_rows if r["suffix"] in {".md", ".tex"}]
    lines = ["# Índice documental", "", "Inventario mecánico completo de Markdown y LaTeX versionados en el árbol actual; no equivale a lectura crítica de cada línea.", "",
             "Las secciones y hashes completos están en corpus_inventory.json. La revisión narrativa enlaza los documentos usados como evidencia.", "",
             "| Documento | Líneas | SHA-256 (prefijo) |", "|---|---:|---|"]
    from urllib.parse import quote
    for row in docs:
        lines.append(f"| [{row['path']}]({quote('../../' + row['path'])}) | {row['lines']} | `{row['sha256'][:12]}` |")
    (OUT / "DOCUMENT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = dict(head=head, since=SINCE, until_exclusive=UNTIL,
                   shallow=git("rev-parse", "--is-shallow-repository").strip(),
                   reachable_commits_in_window=len(commits),
                   commits_outside_head=sum(r["sha"] not in head_shas for r in commits),
                   first_author_date=min(r["author_date"] for r in commits),
                   last_author_date=max(r["author_date"] for r in commits),
                   by_author_month={k: len(v) for k, v in sorted(by_month.items())},
                   tracked_file_count=len([p for p in tracked if p]),
                   inventoried_by_suffix=dict(collections.Counter(r["suffix"] for r in file_rows)),
                   changed_path_entries=sum(len(r["paths"]) for r in commits),
                   unique_changed_paths=len({p["path"] for r in commits for p in r["paths"]}),
                   refs=git("for-each-ref", "--format=%(refname) %(objectname)").splitlines(),
                   scope="Local reachable refs, not remote refresh; inventories are not a reproduction of training or a line-by-line patch audit.")
    dump("inventory_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "refs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
