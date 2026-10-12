"""Package the published reports and the immutable full scientific evidence."""

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

reports = Path(__file__).resolve().parent
root = reports.parents[1]
branch = "scientific-recovery-v16-multimodal-validation"
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
remote = subprocess.check_output(
    ["git", "ls-remote", "--heads", "origin", branch], cwd=root, text=True
).split()[0]
assert remote == commit, (remote, commit)
publication = json.loads((reports / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
historical = root / "artifacts" / publication["historical_bundle"]["name"]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert sha(historical) == publication["historical_bundle"]["sha256"]
archive = root / "artifacts/E_JEPA_TTC_INFORMES_Y_RESULTADOS_20261012.zip"
assert not archive.exists(), "Preserve an existing delivery archive"
entries = []
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    paths = [reports / row["path"] for row in publication["files"]]
    paths.append(reports / "PUBLICATION_MANIFEST.json")
    for path in paths:
        name = "reports/" + path.relative_to(reports).as_posix()
        bundle.write(path, name)
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    name = "evidence/" + historical.name
    bundle.write(historical, name, compress_type=zipfile.ZIP_STORED)
    entries.append({"path": name, "bytes": historical.stat().st_size, "sha256": sha(historical)})
    github = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "remote_head": remote,
        "branch": branch,
        "repository": "https://github.com/Kripta-Studios/e-jepa-ttc",
        "official_submission": False,
    }
    extra = {
        "README.md": (
            "# Informes y resultados multimodales V16\n\n"
            "Abrir [el informe detallado](reports/INFORME_DETALLADO.md).\n\n"
            "`reports/` contiene los informes, tablas y recibos publicados en GitHub. "
            "`evidence/` conserva el ZIP completo anterior sin modificar. "
            "Extraer ese segundo ZIP para acceder a pesos, estados y su verificador.\n\n"
            "No hay puntuación oficial nueva ni SOTA demostrado.\n"
        ),
        "GITHUB_PUBLICATION.json": json.dumps(github, indent=2) + "\n",
    }
    for name, content in extra.items():
        payload = content.encode("utf-8")
        bundle.writestr(name, payload)
        entries.append(
            {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    bundle.writestr("MANIFEST.json", json.dumps({"files": entries}, indent=2) + "\n")
with zipfile.ZipFile(archive) as bundle:
    assert len(bundle.namelist()) == len(set(bundle.namelist()))
    assert bundle.testzip() is None
    for entry in entries:
        with bundle.open(entry["path"]) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        assert bundle.getinfo(entry["path"]).file_size == entry["bytes"]
receipt = {
    "utc": datetime.now(UTC).isoformat(),
    "status": "PASS",
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": sha(archive),
    "payloads": len(entries),
    "crc_pass": True,
    "every_payload_sha256_verified": True,
    "github": github,
    "nested_scientific_bundle_unchanged": True,
}
archive.with_suffix(".verification.json").write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False))
