"""Finish extraction of an existing archive without retaining publisher Torch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from . import continue_campaign
from . import finalize_continuation as final


def recover(archive: Path) -> int:
    """Validate immutable published bytes, then run the admitted serial verifiers."""
    import psutil

    if final.LEASE.exists():
        raise RuntimeError("preserve the existing finalizer owner")
    final.save(
        final.LEASE, dict(pid=os.getpid(), create_time=psutil.Process().create_time(), child=None)
    )
    try:
        authority = continue_campaign.policy()
        continue_campaign.apply_policy(authority)
        from operational.simplex_t_cost_context.engine import Resources

        resources = Resources()
        resources.check()
        queue = final.read(final.QUEUE)
        if queue["status"] != "PUBLICATION_RUNNING" or queue["jobs"]:
            raise ValueError("use the canonical verification resume for an admitted queue")
        if archive.parent.resolve() != (final.NIGHT / "delivery").resolve():
            raise ValueError("archive outside the campaign delivery directory")
        with zipfile.ZipFile(archive) as z:
            manifest_bytes = z.read("CONTENT_MANIFEST.json")
            manifest = json.loads(manifest_bytes)
            import hashlib

            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            if not archive.name.endswith(manifest_sha[:12] + ".zip"):
                raise ValueError("archive filename does not bind the manifest")
            names = z.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest["members"]) | {
                "CONTENT_MANIFEST.json"
            }:
                raise ValueError("duplicate or missing archive members")
            extract = archive.parent / ("extracted_" + manifest_sha[:12])
            resources.reserve_artifacts(
                sum(v["bytes"] for v in manifest["members"].values()), "RECOVER_EXISTING_EXTRACTION"
            )
            for index, name in enumerate(names):
                resources.check()
                target = (extract / name).resolve()
                if not target.is_relative_to(extract.resolve()):
                    raise ValueError("unsafe archive member")
                expected = (
                    manifest_sha
                    if name == "CONTENT_MANIFEST.json"
                    else manifest["members"][name]["sha256"]
                )
                if not target.exists() or final.sha(target) != expected:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".recovering")
                    with z.open(name) as source, temporary.open("wb") as out:
                        while chunk := source.read(1024 * 1024):
                            out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                    if final.sha(temporary) != expected:
                        raise ValueError("extracted member hash mismatch")
                    os.replace(temporary, target)
                # Reading every member also checks CRC, including already extracted files.
                with z.open(name) as source:
                    while source.read(1024 * 1024):
                        pass
                if index % 100 == 0:
                    print(
                        json.dumps(dict(verified_members=index + 1, total=len(names))), flush=True
                    )
            index = final.read(extract / "execution/ANALYSIS_EXPORT_INDEX.json")
            if sorted(index["fits"]) != queue["scope"]["cached_head_ids"]:
                raise ValueError("archive cached-head scope differs from admitted publication")
        archive_sha = final.sha(archive)
        decision = final.read(extract / "NEXT_DECISION_NOCTURNA.json")
        receipt = dict(
            archive=str(archive.resolve()),
            sha256=archive_sha,
            bytes=archive.stat().st_size,
            content_manifest_sha256=manifest_sha,
            verified_members=len(names),
            all_sha256_verified=True,
            crc_verified=True,
            extraction_verified=True,
            termination_reason=decision["termination_reason"],
            checkpoint_verification="DEFERRED_SERIAL_VERIFICATION",
            head_regeneration="DEFERRED_SERIAL_VERIFICATION",
            verification_queue=str(final.QUEUE),
        )
        for script, prefix in [
            ("verify_checkpoints.py", "CHECKPOINT_VERIFICATION"),
            ("regenerate.py", "REGENERATION"),
        ]:
            output = archive.parent / (prefix + "_" + manifest_sha[:12] + ".json")
            queue["jobs"].append(
                dict(
                    argv=[
                        sys.executable,
                        "-B",
                        str(extract / script),
                        "--root",
                        str(extract),
                        "--output",
                        str(output),
                    ],
                    cwd=str(extract),
                    output=str(output),
                    status="PENDING",
                    returncode=None,
                )
            )
        queue.update(status="PUBLICATION_COMPLETE_VERIFICATION_PENDING", archive_sha256=archive_sha)
        final.save(final.NIGHT / "FINAL_DELIVERY_NOCTURNA.json", receipt)
        final.save(final.QUEUE, queue)
        return final.coordinator(True)
    finally:
        if final.LEASE.exists() and not final.alive(final.read(final.LEASE).get("child")):
            final.LEASE.unlink()


def main() -> int:
    """Recover a specific, already published campaign archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    return recover(parser.parse_args().archive)


if __name__ == "__main__":
    raise SystemExit(main())
