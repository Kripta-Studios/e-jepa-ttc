"""Publish current campaign evidence with the separately authorized continuation metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import continue_campaign

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
QUEUE = NIGHT / "SERIAL_DELIVERY_VERIFICATION.json"
LEASE = NIGHT / "FINALIZER.lock"


def alive(owner: dict | None) -> bool:
    """Check PID and creation time, protecting against PID reuse."""
    import psutil

    return bool(
        owner
        and psutil.pid_exists(owner["pid"])
        and psutil.Process(owner["pid"]).create_time() == owner["create_time"]
    )


def child_owner(pid: int | None) -> None:
    """Persist the only heavy child so an orphan cannot be duplicated."""
    import psutil

    lease = read(LEASE)
    lease["child"] = dict(pid=pid, create_time=psutil.Process(pid).create_time()) if pid else None
    save(LEASE, lease)


def await_adoption() -> None:
    """Wait for durable child adoption before importing any heavy numerical package."""
    import psutil

    lineage = [psutil.Process(), *psutil.Process().parents()]
    identities = {(p.pid, p.create_time()) for p in lineage}
    end = time.monotonic() + 10
    while time.monotonic() < end:
        lease = read(LEASE)
        if not alive(lease):
            raise RuntimeError("coordinator exited before adopting this child")
        child = lease.get("child") or {}
        if (child.get("pid"), child.get("create_time")) in identities:
            return
        time.sleep(0.05)
    raise RuntimeError("coordinator did not durably adopt this child")


def verify_job(index: int) -> None:
    """Run only an admitted canonical verifier after the ownership handshake."""
    await_adoption()
    job = read(QUEUE)["jobs"][index]
    script = Path(job["argv"][2]).resolve(strict=True)
    root = Path(job["argv"][job["argv"].index("--root") + 1]).resolve(strict=True)
    if script.parent != root or script.name not in {"regenerate.py", "verify_checkpoints.py"}:
        raise ValueError("verification script outside the admitted extracted root")
    if sha(script) != read(root / "CONTENT_MANIFEST.json")["members"][script.name]["sha256"]:
        raise ValueError("canonical verification source changed")
    sys.argv = [str(script), *job["argv"][3:]]
    runpy.run_path(str(script), run_name="__main__")


def read(path: Path) -> dict:
    """Read a durable JSON without holding an open file during publication."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Hash one file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    """Commit a receipt atomically, including a filesystem flush."""
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("wb") as stream:
        stream.write((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def export_scope(inventory: dict) -> dict:
    """Require the exact complete-arm scope and its sealed cached payloads."""
    execution = NIGHT / "execution"
    complete = [r for r in inventory["fits"] if r["saved_updates"] == 2500]
    arms = {r["arm"] for r in complete if r["family"] != "N1"}
    arms = {a for a in arms if sum(r["arm"] == a for r in complete) == 3}
    complete_candidates = {r["id"]: r for r in complete if r["family"] != "N1" and r["arm"] in arms}
    index = read(execution / "ANALYSIS_EXPORT_INDEX.json")
    if not set(index["fits"]).issubset(complete_candidates):
        raise ValueError("cached export contains an unauthorized or incomplete endpoint")
    published_arms = {r["arm"] for key, r in complete_candidates.items() if key in index["fits"]}
    expected = {key: r for key, r in complete_candidates.items() if r["arm"] in published_arms}
    if set(index["fits"]) != set(expected):
        raise ValueError("cached export scope differs from the authorized complete arms")
    if inventory["endpoints"] == 24 and len(expected) != 18:
        raise ValueError("complete union requires all 18 C0 cached heads")
    for family in {r["family"] for r in expected.values()}:
        binding = index["families"][family]
        seal_path = Path(binding["seal_path"]).resolve(strict=True)
        if (
            not seal_path.is_relative_to(NIGHT.resolve())
            or sha(seal_path) != binding["seal_sha256"]
        ):
            raise ValueError("published family seal differs")
        seal = read(seal_path)
        if not (
            seal.get("all_family_frozen_before_evaluation")
            or seal.get("frozen_before_first_evaluation")
        ):
            raise ValueError("family was not sealed before evaluation")
        sealed = {r["key"]: r for r in seal["fits"]}
        for key, fit in expected.items():
            if fit["family"] == family and (
                key not in sealed or sealed[key]["checkpoint_sha256"] != fit["checkpoint_sha256"]
            ):
                raise ValueError("export endpoint is not the sealed endpoint")
    for key, row in index["fits"].items():
        if row["checkpoint_sha256"] != expected[key]["checkpoint_sha256"]:
            raise ValueError("cached export binds another checkpoint")
        payloads = [
            (row[name + "_path"], row[name + "_sha256"])
            for name in ("weights", "predictions", "publication")
        ]
        payloads += [
            (f[name + "_path"], f[name + "_sha256"])
            for f in row["fragments"]
            for name in ("input", "output", "receipt")
        ]
        for relative, digest in payloads:
            path = (execution / relative).resolve(strict=True)
            if not path.is_relative_to(execution.resolve()) or sha(path) != digest:
                raise ValueError("cached export payload or binding differs")
    return dict(
        cached_head_ids=sorted(expected),
        checkpoint_ids=sorted(r["id"] for r in inventory["fits"] if r["checkpoint"]),
        export_index_sha256=sha(execution / "ANALYSIS_EXPORT_INDEX.json"),
        complete_but_not_exported_ids=sorted(
            {r["id"] for r in complete if r["family"] != "N1"} - set(expected)
        ),
        optimizer_updates=0,
    )


def amended_decision(decision: dict, inventory: dict, authority: dict) -> dict:
    """Apply operational authority without changing any score or scientific criterion."""
    updated = dict(decision)
    remaining = authority["scientific_saved_update_cap"] - inventory["scientific_saved_updates"]
    if remaining < 0:
        raise ValueError("physical inventory exceeds the scientific authorization")
    updated["no_push"] = False
    updated["push_authorized_after_delivery"] = True
    updated["future_optimizer_updates_authorized"] = remaining
    updated["operational_continuation"] = {
        k: authority[k]
        for k in (
            "accepted_at_utc",
            "deadline_utc",
            "own_artifact_budget_bytes",
            "original_window_sha256",
            "scientific_protocol_sha256",
            "operational_source_sha256",
            "startup_margin_waiver_by_user",
        )
    }
    updated["resume_command"] = (
        (
            "& '../e-jepa-ttc/.venv/Scripts/python.exe' -B "
            "-m operational.simplex_t_io_recovery.continue_campaign"
        )
        if remaining
        else None
    )
    if inventory["endpoints"] == 24 and remaining == 0:
        updated["termination_reason"] = "AUTHORIZED_TRAINING_COMPLETE"
        updated["next_training_task"] = None
    return updated


def reconcile_head_cost() -> None:
    """Publish only completed physical profile receipts, including partial cost scope."""
    columns = [
        "model",
        "status",
        "scope",
        "device",
        "fold",
        "batch",
        "measurements",
        "p50_ms",
        "p95_ms",
        "parameters",
        "batch128_windows_per_second",
        "endpoint_sha256",
        "selection_sha256",
    ]
    expected = {
        "H1_SEED7",
        "H8_SEED7",
        "H16_SEED7",
        "FULL_C0",
        "A5_ONLY_C0",
        "C2F_ONLY_C0",
        "A5_PAIR_C0",
        "SET_AGE_C0",
        "SET_NOTIME_C0",
    }
    rows = []
    for path in sorted((NIGHT / "profiling").glob("*.json")):
        receipt = read(path)
        if receipt.get("model") not in expected or receipt.get("status") != "MEASURED":
            continue
        if (
            receipt["scope"] != "prepared_input_head_only"
            or receipt["batch"] != 1
            or len(receipt["raw_ms"]) != receipt["measurements"]
        ):
            raise ValueError("head profile measurement receipt is incomplete")
        rows.append({key: receipt[key] for key in columns})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = NIGHT / "HEAD_COST.csv.next"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        output.write(stream.getvalue())
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, NIGHT / "HEAD_COST.csv")
    admission = NIGHT / "profiling/FULL_ROUTE_ADMISSION.json"
    save(
        NIGHT / "N4_COST_STATUS.json",
        dict(
            prepared_head="COMPLETE" if {r["model"] for r in rows} == expected else "PARTIAL",
            models=len(rows),
            missing_head_models=sorted(expected - {r["model"] for r in rows}),
            full_route="NOT_MEASURED",
            missing_dependency=read(admission)["missing_dependency"],
            full_route_admission_sha256=sha(admission),
            route_cost_does_not_follow_from_head_latency=True,
            profile_receipts_reconciled_without_new_measurements=True,
        ),
    )


def publisher() -> None:
    """Publish in one heavy process; queue independent verification for after exit."""
    await_adoption()
    authority = continue_campaign.policy()
    continue_campaign.apply_policy(authority)
    from operational.simplex_t_cost_context import deliver, engine

    engine.Resources().check()
    original_report = deliver.report
    original_selection = deliver.selected_files
    original_run = subprocess.run
    queue: dict[str, Any] = dict(status="PUBLICATION_RUNNING", jobs=[], optimizer_updates=0)

    def report(inventory: dict) -> dict:
        queue["scope"] = export_scope(inventory)
        save(QUEUE, queue)
        reconcile_head_cost()
        decision = amended_decision(original_report(inventory), inventory, authority)
        interpretation = NIGHT / "SCIENTIFIC_INTERPRETATION.json"
        if interpretation.exists():
            decision["scientific_interpretation"] = read(interpretation)
            decision["scientific_interpretation_sha256"] = sha(interpretation)
        deliver.atomic_json(NIGHT / "NEXT_DECISION_NOCTURNA.json", decision)
        report_path = NIGHT / "INFORME_NOCTURNO_SIMPLEX_T.md"
        head = (
            "# Autoridad operativa vigente de esta entrega\n\n"
            f"Continuación autorizada: {authority['accepted_at_utc']} → "
            f"{authority['deadline_utc']}; presupuesto propio 10 GB decimales. "
            "La ventana histórica y el protocolo científico conservan sus bytes. "
            "El push posterior a la entrega está autorizado.\n\n"
            f"Inventario físico: {inventory['endpoints']} endpoints completos, "
            f"{inventory['scientific_saved_updates']} updates científicos guardados; "
            f"{decision['future_optimizer_updates_authorized']} updates pendientes. "
            "Los resultados numéricos y criterios permanecen los del protocolo.\n\n"
            "El ZIP incluye el bundle H16 independiente previamente regenerado, "
            "con sus seis cabezas y cachés; conserva su SHA-256 original. Las cabezas "
            "C0 publicadas se regeneran de nuevo desde la extracción del bundle actual. "
            "No se reconstruyen datos raw/TRAIN ni expertos.\n\n"
            "NEXT_DECISION_NOCTURNA.json contiene la autoridad vigente; el detalle "
            "del publicador canónico que sigue conserva también metadatos históricos.\n\n---\n\n"
        )
        body = report_path.read_text(encoding="utf-8").replace(
            "artefactos propios2GiB", "artefactos propios10GB decimales"
        )
        start = body.index("Comando local de recuperación con campaña precisa:")
        end = body.find("\n## ", start)
        if end < 0:
            end = len(body)
        replacement = (
            "No quedan fits autorizados pendientes. No se debe relanzar el trainer.\n"
            if decision["future_optimizer_updates_authorized"] == 0
            else f"Comando de recuperación: `{decision['resume_command']}`.\n"
        )
        body = body[:start] + replacement + body[end:]
        conclusions = NIGHT / "CONCLUSIONES_NOCTURNAS.md"
        if conclusions.exists():
            body += "\n\n" + conclusions.read_text(encoding="utf-8")
        deliver.atomic_bytes(report_path, (head + body).encode())
        return decision

    def selected() -> dict[str, Path]:
        files = {
            name: path
            for name, path in original_selection().items()
            if not name.startswith("compact_delivery/")
            and name
            not in {
                "RECOVERED_DRIVER.log",
                "IO_REPLACE_RETRIES.jsonl",
                "SERIAL_PUBLICATION.log",
                "SERIAL_DELIVERY_VERIFICATION.json",
                "FINALIZER.lock",
            }
        }
        directory = ROOT / "operational/simplex_t_io_recovery"
        for name in (
            "continue_campaign.py",
            "finalize_continuation.py",
            "run.py",
            "README.md",
            "export_profile_inputs.py",
            "profile_cached.py",
        ):
            files["provenance/operational/simplex_t_io_recovery/" + name] = directory / name
        files["profile_cached.py"] = directory / "profile_cached.py"
        for receipt_path in (NIGHT / "driver_receipts").glob("RESUME_FROM_52777.json"):
            log_path = receipt_path.with_suffix(".log")
            closed = read(receipt_path)
            if "returncode" not in closed or sha(log_path) != closed["log_sha256"]:
                raise ValueError("remaining-training run log is not durably closed")
            for path in (receipt_path, log_path):
                files[path.relative_to(NIGHT).as_posix()] = path
        h16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
        archive = h16 / "E_JEPA_TTC_H16_REPLICATION_20261003.zip"
        archive_sha256 = sha(archive)
        if archive_sha256 != "08ee2aae60d56006a73fdb8b249a6467cee0704b0a258bb9ad5659f863b3e348":
            raise ValueError("previously verified independent H16 bundle changed")
        regenerated = read(h16 / "REGENERATION.json")
        h16_delivery = read(h16 / "FINAL_DELIVERY.json")
        if h16_delivery["sha256"] != archive_sha256 or h16_delivery["regeneration_sha256"] != sha(
            h16 / "REGENERATION.json"
        ):
            raise ValueError("independent H16 regeneration receipt is not bound to its archive")
        if (
            regenerated["optimizer_updates"] != 0
            or {(r["seed"], r["fold"]) for r in regenerated["heads"]}
            != {(seed, fold) for seed in (13, 23) for fold in range(3)}
            or any(r["max_ttc_error"] != 0 for r in regenerated["heads"])
        ):
            raise ValueError("H16 inherited regeneration receipt does not cover its six heads")
        files["h16/" + archive.name] = archive
        return files

    deliver.report = report
    deliver.selected_files = selected

    def defer(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:  # noqa: ANN401
        if len(argv) > 2 and Path(argv[2]).name in {"verify_checkpoints.py", "regenerate.py"}:
            output = Path(argv[argv.index("--output") + 1])
            queue["jobs"].append(
                dict(
                    argv=argv,
                    cwd=str(kwargs["cwd"]),
                    output=str(output),
                    status="PENDING",
                    returncode=None,
                )
            )
            save(QUEUE, queue)
            return subprocess.CompletedProcess(argv, 125, "Deferred until publisher exits.\n", "")
        return original_run(argv, **kwargs)

    deliver.subprocess.run = defer
    receipt: dict[str, Any] = {
        "status": "CANONICAL_PUBLICATION_WITH_OPERATIONAL_METADATA_AMENDMENT",
        "updated_utc": datetime.now(UTC).isoformat(),
        "publisher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "continuation_authorization_sha256": hashlib.sha256(
            continue_campaign.AUTH.read_bytes()
        ).hexdigest(),
        "optimizer_updates": 0,
        "numeric_implementation_unchanged": True,
        "journal_sha256": {
            "N1": sha(ROOT / "artifacts/simplex_t/h16_replication_20261003/PHYSICAL_WORK.json"),
            "N2_N3": sha(NIGHT / "execution/PHYSICAL_WORK.json"),
        },
    }
    amendment = NIGHT / "FINAL_PUBLICATION_AMENDMENT.json"
    if amendment.exists():
        previous = read(amendment)
        if all(
            previous.get(key) == receipt[key]
            for key in (
                "publisher_source_sha256",
                "continuation_authorization_sha256",
                "journal_sha256",
            )
        ):
            receipt = previous
    deliver.atomic_json(NIGHT / "FINAL_PUBLICATION_AMENDMENT.json", receipt)
    # Canonical main still enforces writer ownership, physical checkpoint
    # admission, publication reservations, CRC/SHA and independent regeneration.
    deliver.main()
    final = read(NIGHT / "FINAL_DELIVERY_NOCTURNA.json")
    final["checkpoint_verification"] = "DEFERRED_SERIAL_VERIFICATION"
    final["head_regeneration"] = "DEFERRED_SERIAL_VERIFICATION"
    final["verification_queue"] = str(QUEUE)
    save(NIGHT / "FINAL_DELIVERY_NOCTURNA.json", final)
    queue["status"] = "PUBLICATION_COMPLETE_VERIFICATION_PENDING"
    queue["archive_sha256"] = final["sha256"]
    save(QUEUE, queue)
    print(json.dumps(receipt), flush=True)


def coordinator(resume: bool) -> int:
    """Verify serially after the publisher exits; failures remain durable and nonzero."""
    import psutil

    driver_lock = NIGHT / "DRIVER.lock"
    if driver_lock.exists():
        owner = read(driver_lock)
        if psutil.pid_exists(owner["pid"]) and (
            psutil.Process(owner["pid"]).create_time() == owner["create_time"]
        ):
            raise RuntimeError("wait for the current campaign driver before final publication")
    if QUEUE.exists():
        for job in read(QUEUE)["jobs"]:
            if (
                job.get("pid")
                and psutil.pid_exists(job["pid"])
                and (psutil.Process(job["pid"]).create_time() == job.get("create_time"))
            ):
                raise RuntimeError("an independent verifier is still alive; preserve its owner")
    if not resume:
        command = [
            sys.executable,
            "-B",
            "-m",
            "operational.simplex_t_io_recovery.finalize_continuation",
            "--publisher",
        ]
        with (NIGHT / "SERIAL_PUBLICATION.log").open("w", encoding="utf-8") as log:
            publisher_child = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            child_owner(publisher_child.pid)
            returncode = publisher_child.wait()
            child_owner(None)
        if returncode:
            return returncode
    authority = continue_campaign.policy()
    continue_campaign.apply_policy(authority)
    from operational.simplex_t_cost_context import engine

    if "torch" in sys.modules:
        raise RuntimeError("verification coordinator must not retain Torch")
    queue = read(QUEUE)
    final = read(NIGHT / "FINAL_DELIVERY_NOCTURNA.json")
    if queue["archive_sha256"] != final["sha256"] or sha(Path(final["archive"])) != final["sha256"]:
        raise ValueError("verification queue belongs to another archive")
    for index, job in enumerate(queue["jobs"]):
        output = Path(job["output"])
        if job["status"] == "VERIFIED" and output.exists() and sha(output) == job["receipt_sha256"]:
            continue
        engine.Resources().check()
        job["status"] = "RUNNING"
        save(QUEUE, queue)
        with output.with_suffix(".log").open("w", encoding="utf-8") as log:
            command = [
                sys.executable,
                "-B",
                "-m",
                "operational.simplex_t_io_recovery.finalize_continuation",
                "--verify-job",
                str(index),
            ]
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job["pid"] = child.pid
            job["create_time"] = psutil.Process(child.pid).create_time()
            child_owner(child.pid)
            save(QUEUE, queue)
            while child.poll() is None:
                time.sleep(3)
            job["returncode"] = child.returncode
            child_owner(None)
        if child.returncode != 0 or not output.exists():
            job["status"] = "BLOCKED_OR_FAILED"
            queue["status"] = "VERIFICATION_PENDING_AFTER_FAILURE"
            save(QUEUE, queue)
            return child.returncode or 1
        result = read(output)
        if result["optimizer_updates"] != 0:
            raise ValueError("verification performed optimizer work")
        if Path(job["argv"][2]).name == "regenerate.py":
            ids = {r["key"] for r in result["heads"]}
            expected = set(queue["scope"]["cached_head_ids"])
            field, receipt_field = "head_regeneration", "regeneration_receipt_sha256"
        else:
            ids = {r["id"] for r in result["checkpoints"]}
            expected = set(queue["scope"]["checkpoint_ids"])
            field, receipt_field = (
                "checkpoint_verification",
                "checkpoint_verification_receipt_sha256",
            )
        rows = (
            result["heads"]
            if Path(job["argv"][2]).name == "regenerate.py"
            else result["checkpoints"]
        )
        if ids != expected or len(rows) != len(expected):
            job["status"] = "FAILED_SCOPE_ADMISSION"
            queue["status"] = "VERIFICATION_PENDING_AFTER_FAILURE"
            save(QUEUE, queue)
            raise ValueError("verification receipt covers a different scope")
        job["status"] = "VERIFIED"
        job["receipt_sha256"] = sha(output)
        final[field] = "VERIFIED"
        final[receipt_field] = sha(output)
        save(NIGHT / "FINAL_DELIVERY_NOCTURNA.json", final)
        save(QUEUE, queue)
    if {Path(j["argv"][2]).name for j in queue["jobs"]} != {
        "verify_checkpoints.py",
        "regenerate.py",
    }:
        raise ValueError("both independent verification jobs are required")
    queue["status"] = "SERIAL_DELIVERY_VERIFIED_EXACT_SCOPE"
    save(QUEUE, queue)
    final["verification_status"] = queue["status"]
    final["verified_cached_head_ids"] = queue["scope"]["cached_head_ids"]
    final["optimizer_updates_for_verification"] = 0
    final["h16_cached_inference"] = dict(
        scope="Six new H16 heads in the included independent nested essential archive",
        member="h16/E_JEPA_TTC_H16_REPLICATION_20261003.zip",
        sha256="08ee2aae60d56006a73fdb8b249a6467cee0704b0a258bb9ad5659f863b3e348",
        verification=(
            "Inherited exact regeneration of the byte-identical H16 archive; receipt included"
        ),
        additional_optimizer_updates=0,
    )
    save(NIGHT / "FINAL_DELIVERY_NOCTURNA.json", final)
    print(json.dumps(final), flush=True)
    return 0


def main() -> int:
    """Coordinate publication and separate, resumable zero-update verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher", action="store_true")
    parser.add_argument("--resume-verification", action="store_true")
    parser.add_argument("--verify-job", type=int)
    args = parser.parse_args()
    if args.verify_job is not None:
        verify_job(args.verify_job)
        return 0
    if args.publisher:
        sys.argv = [sys.argv[0]]
        publisher()
        return 0
    import psutil

    if LEASE.exists():
        prior = read(LEASE)
        if alive(prior) or alive(prior.get("child")):
            raise RuntimeError("another final publisher or verifier owns this directory")
        LEASE.replace(NIGHT / ("STALE_FINALIZER_" + sha(LEASE)[:12] + ".json"))
    owner = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), child=None)
    with LEASE.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        return coordinator(args.resume_verification)
    finally:
        if LEASE.exists() and not alive(read(LEASE).get("child")):
            LEASE.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
