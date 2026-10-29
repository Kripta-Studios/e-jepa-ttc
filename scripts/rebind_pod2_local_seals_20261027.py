"""Rebind only physical file identities after full content verification on a new Pod."""

import concurrent.futures as cf
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUT = Path("/workspace/ejepa_event_operations/resume_20261027")


def digest(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    result = {
        "schema": "ejepa_physical_relocation_v1",
        "pod_id": "lreiivpf18vxu7",
        "previous_pod_id": "8mqxzys5bz6218",
        "utc": datetime.now(UTC).isoformat(),
        "scientific_files_modified": False,
        "roots": {},
    }
    for name in [
        "ejepa_event_bundle",
        "ejepa_event_supplement",
        "ejepa_event_authority_supplement_v2",
    ]:
        root = Path("/opt") / name
        old = root.with_suffix(".SEAL.json")
        manifest = root / (
            "BUNDLE_MANIFEST.json" if name == "ejepa_event_bundle" else "SUPPLEMENT_MANIFEST.json"
        )
        data = json.loads(manifest.read_text())
        seal = json.loads(old.read_text())

        def verify(item: dict[str, Any], *, root_path: Path = root) -> dict[str, int | str]:
            p = root_path / item["relative"]
            assert p.resolve().is_relative_to(root_path.resolve()) and not p.is_symlink()
            assert p.stat().st_size == item["bytes"] and digest(p) == item["sha256"], str(p)
            p.chmod(0o444)
            s = p.stat()
            return {
                "relative": item["relative"],
                "bytes": s.st_size,
                "device": s.st_dev,
                "inode": s.st_ino,
                "mode": s.st_mode,
                "mtime_ns": s.st_mtime_ns,
            }

        with cf.ThreadPoolExecutor(4) as pool:
            stats = list(pool.map(verify, data["files"]))
        assert len(stats) == len(seal["files"])
        seal["files"] = stats
        new = OUT / (name + ".RELOCATED_SEAL.json")
        new.write_text(json.dumps(seal, indent=2, sort_keys=True))
        result["roots"][str(root)] = {
            "original_seal": str(old),
            "original_seal_sha256": digest(old),
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "new_seal": str(new),
            "new_seal_sha256": digest(new),
            "files_verified": len(stats),
        }
        (OUT / "REBIND_PROGRESS.json").write_text(
            json.dumps({"roots_verified": list(result["roots"])})
        )
    p = OUT / "PHYSICAL_RELOCATION.json"
    p.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"status": "VERIFIED", "receipt": str(p), "sha256": digest(p)}), flush=True)


if __name__ == "__main__":
    main()
