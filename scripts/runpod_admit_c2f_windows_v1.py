"""Verify Linux C2F proofs on Windows, then launch the three authorized fits."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from runpod_verify_windows_c2f_probe import verify

from operational.runpod_migration.ownership_c2f import verify_claim


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _main() -> None:
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "artifacts/runpod_migration_20261026"
    launch_intent = artifacts / "C2F_LAUNCH_INTENT.json"
    if launch_intent.exists() or (artifacts / "C2F_SCIENTIFIC_START.json").exists():
        raise ValueError("C2F launch already attempted; reconcile recorded PIDs before retrying")
    returns = repo / "artifacts/runpod_returns/irtz8y3sigd621"
    key = Path.home() / ".ssh/id_ed25519"
    host, port = "root@213.173.109.37", "13608"
    ssh = ["ssh", "-i", str(key), "-p", port, "-o", "StrictHostKeyChecking=yes", host]
    scp = ["scp", "-i", str(key), "-P", port, "-o", "StrictHostKeyChecking=yes"]
    ops = "/workspace/ejepa_operations"
    budget = json.loads((artifacts / "USER_BUDGET_24USD.json").read_text())
    if datetime.now(UTC) >= datetime.fromisoformat(budget["execution_deadline_utc"]):
        raise ValueError("Shared budget authorization expired")

    def remote(command: list[str]) -> str:
        return subprocess.check_output(ssh + [shlex.join(command)], text=True, timeout=180)

    for rotation in (2, 3, 4):
        status = json.loads(remote(["cat", f"{ops}/r{rotation}_c2f/TECHNICAL_STATUS.json"]))
        if (
            status.get("status") != "TECHNICAL_READY"
            or status.get("scientific_optimizer_updates") != 0
        ):
            raise ValueError(f"C2F proof not ready: {rotation}")
        verify_claim(
            repo / "artifacts/final_refit_20261019_scientific_oof",
            artifacts / f"r{rotation}_c2f/LOCAL_QUEUE_EXCLUSION.json",
            rotation,
        )

    for rotation in (2, 3, 4):
        run = f"/workspace/ejepa_technical/r{rotation}_c2f_v3"
        folder = artifacts / f"r{rotation}_c2f"
        target = returns / f"technical_r{rotation}_c2f_v3"
        target.mkdir(parents=True, exist_ok=True)
        archive_remote = f"{ops}/r{rotation}_c2f/TECHNICAL_RETURN.tar"
        remote(
            [
                "tar",
                "--hard-dereference",
                "-cf",
                archive_remote,
                "-C",
                run,
                "technical_resume_probe/R_C2F",
                "MIGRATION_BINDING.json",
            ]
        )
        expected = remote(["sha256sum", archive_remote]).split()[0]
        archive = target / "TECHNICAL_RETURN.tar"
        subprocess.run(scp + [f"{host}:{archive_remote}", str(archive)], check=True, timeout=180)
        if digest(archive) != expected:
            raise ValueError("Downloaded C2F proof archive hash differs")
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            if sum(item.size for item in members) > 2 * 1024**3:
                raise ValueError("Unexpectedly large technical archive")
            for item in members:
                destination = target / item.name
                if not (item.isfile() or item.isdir()) or not destination.resolve().is_relative_to(
                    target.resolve()
                ):
                    raise ValueError("Unsafe technical archive member")
            stream.extractall(target, members=members, filter="data")
        compatibility = verify(target, rotation)
        migration = digest(target / "MIGRATION_BINDING.json")
        assignment = {
            "schema": "ejepa_cloud_assignment_v1",
            "execution_authorized": True,
            "tasks": [{"rotation": rotation, "fit": "R_C2F"}],
            "migration_binding_sha256": migration,
            "maximum_compute_usd": budget["maximum_total_usd"],
            "execution_deadline_utc": budget["execution_deadline_utc"],
            "technical_update_cap": 8,
            "budget_is_global_same_pod_not_additive": True,
            "local_queue_exclusion_receipt_sha256": digest(folder / "LOCAL_QUEUE_EXCLUSION.json"),
        }
        assignment_path = folder / f"SCIENTIFIC_R{rotation}_C2F.json"
        assignment_path.write_text(json.dumps(assignment, indent=2) + "\n")
        shutil.copyfile(
            target / "WINDOWS_CHECKPOINT_COMPATIBILITY.json",
            folder / "WINDOWS_CHECKPOINT_COMPATIBILITY.json",
        )
        shutil.copyfile(artifacts / "USER_BUDGET_24USD.json", folder / "USER_BUDGET_24USD.json")
        subprocess.run(
            scp
            + [
                str(assignment_path),
                str(folder / "WINDOWS_CHECKPOINT_COMPATIBILITY.json"),
                str(folder / "USER_BUDGET_24USD.json"),
                f"{host}:{ops}/r{rotation}_c2f/",
            ],
            check=True,
            timeout=60,
        )
        print(
            json.dumps({"rotation": rotation, "windows_compatibility": compatibility["status"]}),
            flush=True,
        )

    starts = {}
    launch_intent.write_text(json.dumps({"status": "STARTING", "pids": starts}) + "\n")
    for rotation in (2, 3, 4):
        run = f"/workspace/ejepa_technical/r{rotation}_c2f_v3"
        folder = f"{ops}/r{rotation}_c2f"
        remote(["test", "!", "-e", f"{run}/rotations/{rotation}/fits/R_C2F/PROGRESS.json"])
        command = [
            "/opt/ejepa_env/bin/python",
            "-u",
            f"{ops}/runpod_scientific_overlay_c2f_v1.py",
            "producer",
            "--bundle",
            "/workspace/ejepa_bundle",
            "--run",
            run,
            "--rotation",
            str(rotation),
            "--fit",
            "R_C2F",
            "--budget-bytes",
            "60999999488",
            "--assignment",
            f"{folder}/SCIENTIFIC_R{rotation}_C2F.json",
        ]
        repo_remote = "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
        shell = (
            f"PYTHONPATH={repo_remote}/src:{repo_remote} PYTHONDONTWRITEBYTECODE=1 nohup "
            + shlex.join(command)
            + f" > {folder}/SCIENTIFIC_C2F.log 2>&1 < /dev/null & echo $!"
        )
        pid = int(subprocess.check_output(ssh + [shell], text=True, timeout=30).strip())
        starts[str(rotation)] = pid
        launch_intent.write_text(json.dumps({"status": "STARTING", "pids": starts}) + "\n")
        backup_root = returns / f"scientific_r{rotation}_c2f"
        backup_root.mkdir(parents=True, exist_ok=True)
        with (backup_root / "helper.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(repo / "scripts/runpod_backup_c2f_v1.py"),
                    "--host",
                    host,
                    "--port",
                    port,
                    "--key",
                    str(key),
                    "--run",
                    run,
                    "--rotation",
                    str(rotation),
                    "--output",
                    str(backup_root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        (backup_root / "BACKUP_PROCESS.json").write_text(
            json.dumps({"pid": process.pid, "poll_seconds": 15}) + "\n"
        )
    result = {
        "status": "THREE_C2F_LAUNCHED",
        "utc": datetime.now(UTC).isoformat(),
        "pids": starts,
        "scientific_recipe_changed": False,
        "budget_shared_total_usd": 24,
    }
    (artifacts / "C2F_SCIENTIFIC_START.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    from operational.final_refit.continuation_queue import queue_lock

    root = Path(__file__).resolve().parents[1] / "artifacts/runpod_migration_20261026"
    with queue_lock(root / "C2F_ADMISSION.lock"):
        _main()


if __name__ == "__main__":
    main()
