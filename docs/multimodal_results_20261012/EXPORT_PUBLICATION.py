"""Copy local text results without altering scientific sources or their hashes."""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

destination = Path(__file__).resolve().parent
root = destination.parents[1]
allowed = {".md", ".json", ".csv", ".txt"}
sources = {}


def copy(path: Path) -> None:
    relative = path.relative_to(root)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert path.read_bytes() == target.read_bytes()
    sources[target.relative_to(destination).as_posix()] = relative.as_posix()


research = root / "artifacts/multimodal_research_20261012"
for path in research.iterdir():
    if path.is_file() and path.suffix in allowed:
        copy(path)
for folder in (
    "studies",
    "fits",
    "group_crossfit",
    "external_evaluation",
    "external_evaluation_v2",
    "parent_replay",
    "phase_only_cost_v1",
):
    for path in (research / folder).rglob("*"):
        if path.is_file() and path.suffix in allowed:
            copy(path)
for folder in ("test12_phase_only", "test12_readiness"):
    for path in (research / folder).iterdir():
        if path.is_file() and path.suffix in allowed:
            copy(path)
copy(research / "test12_phase_only/PHASE_ONLY_S7_GarlTTC_submission.zip")
validation = root / "artifacts/multimodal_validation_20261012"
for path in validation.rglob("*"):
    if path.is_file() and path.suffix in allowed:
        copy(path)
for suffix in ("verification.json", "independent-verification.json", "delivery.json"):
    path = root / f"artifacts/E_JEPA_TTC_RESEARCH_RESULT_20261012T164206Z.{suffix}"
    target = destination / "receipts" / path.name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    sources[target.relative_to(destination).as_posix()] = path.relative_to(root).as_posix()

authorization = {
    "recorded_utc": datetime.now(UTC).isoformat(),
    "user_instruction": (
        "explica en .md en detalle todo lo que has hecho y los resultados conseguidos "
        "y haz push a github de eso y crea un .zip con esos informes y resultados"
    ),
    "authorized": ["detailed Markdown reports", "GitHub branch push", "reports and results ZIP"],
    "official_submission_authorized_by_this_request": False,
    "scientific_closeout_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "branch": "scientific-recovery-v16-multimodal-validation",
    "historical_receipts_preserved": True,
}
(destination / "PUBLICATION_AUTHORIZATION.json").write_text(
    json.dumps(authorization, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
entries = []
for path in sorted(destination.rglob("*")):
    if (
        not path.is_file()
        or path.name == "PUBLICATION_MANIFEST.json"
        or "__pycache__" in path.parts
    ):
        continue
    relative = path.relative_to(destination).as_posix()
    content = path.read_bytes()
    entries.append(
        {
            "path": relative,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source": sources.get(relative, "publication documentation"),
        }
    )
manifest = {
    "schema": "multimodal_publication_v1",
    "created_utc": datetime.now(UTC).isoformat(),
    "files": entries,
    "bytes": sum(row["bytes"] for row in entries),
    "copy_policy": (
        "Text reports, protocols, CSV predictions and receipts from both phases; "
        "full study tables and fit receipts; top-level test12 receipts and "
        "official-format ZIP. Original bytes preserved."
    ),
    "full_bundle_only": (
        "Weights, tensor caches, prepared sensor inputs, executed source snapshots, "
        "diagnostic image panels and verbose runtime logs. "
        "Raw sensor datasets remain external to both bundles."
    ),
    "historical_bundle": {
        "name": "E_JEPA_TTC_RESEARCH_RESULT_20261012T164206Z.zip",
        "bytes": 461398299,
        "sha256": "a59bac60a1a97c4c5e0b279247c98b96b1b4ce1ff3dc8db7781b4cf01372bab9",
    },
}
(destination / "PUBLICATION_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps({"files": len(entries), "bytes": manifest["bytes"]}))
