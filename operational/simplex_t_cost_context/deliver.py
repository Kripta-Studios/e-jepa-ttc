"""Publish complete or partial nocturnal evidence; never fabricate absent scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
H16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
sys.path.insert(0, str(ROOT / "operational/simplex_t_h16_replication"))
from common import atomic_bytes, atomic_json, digest, durable_stream, record  # noqa: E402


class ResourceGuard(Protocol):
    def check(self) -> None: ...


def ledger(root: Path) -> dict:
    path = root / "PHYSICAL_WORK.json"
    return (
        record(path)
        if path.exists()
        else dict(
            fits={},
            accounting=dict(
                scientific_saved_updates=0,
                scientific_uncertain_lost_upper=0,
            ),
        )
    )


def admit_checkpoint(path: Path, fit: dict, historical: bool, progress: dict) -> int:
    """Read a complete durable state without rewriting its journal or checkpoint."""
    from e_jepa_ttc.simplex_t.training import load_checkpoint

    protocol_path = H16 / "PROTOCOL.json" if historical else NIGHT / "PROTOCOL_COST_CONTEXT.json"
    p = record(protocol_path)
    source = p["sources"][str(fit["fold"])]
    source_sha = (
        source["train_sha256"] if historical else source["wrapped"][fit["arm"]]["train_sha256"]
    )
    state = load_checkpoint(path)
    identity = state["identity"]
    completed = state["completed_updates"]
    journal_saved = progress.get("completed", 0)
    pending = progress.get("pending")
    upper = pending[1] if pending else journal_saved
    if (
        not journal_saved <= completed <= upper <= 2500
        or identity["source"] != source_sha
        or identity["freeze"] != digest(protocol_path)
        or identity["seed"] != fit["seed"]
        or identity["endpoint"] != 2500
        or identity["batch"] != 128
        or identity["device"] != "cpu"
    ):
        raise ValueError(f"physical checkpoint cannot reconcile with admitted journal: {path}")
    return completed


def inventory(*, admit_physical_checkpoints: bool = False) -> dict:
    authorized = record(NIGHT / "QUEUE_AUTHORIZED.json")["fits"]
    journals = {"N1": ledger(H16), "N2_N3": ledger(NIGHT / "execution")}
    rows = []
    for fit in authorized:
        key = fit["id"]
        historical = key.startswith("H16_REPLICATION")
        journal = journals["N1" if historical else "N2_N3"]
        progress = journal["fits"].get(key, {})
        saved = progress.get("completed", 0)
        root = H16 if historical else NIGHT / "execution"
        checkpoint = root / "fits" / key / "checkpoint_last.pt"
        if not checkpoint.exists() and saved:
            raise FileNotFoundError(f"saved checkpoint is missing: {checkpoint}")
        journal_saved = saved
        if admit_physical_checkpoints and checkpoint.exists():
            saved = admit_checkpoint(checkpoint, fit, historical, progress)
        rows.append(
            dict(
                id=key,
                family="N1"
                if historical
                else (
                    "N2"
                    if fit["arm"] in ("FULL_C0", "A5_ONLY_C0", "C2F_ONLY_C0", "A5_PAIR_C0")
                    else "N3"
                ),
                arm=fit["arm"],
                fold=fit["fold"],
                seed=fit["seed"],
                saved_updates=saved,
                journal_saved_updates=journal_saved,
                checkpoint_journal_reconciled=saved != journal_saved,
                status="COMPLETE_ENDPOINT"
                if saved == 2500
                else ("RECOVERABLE_PARTIAL" if checkpoint.exists() else "NOT_STARTED"),
                checkpoint=str(checkpoint) if checkpoint.exists() else None,
                checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else None,
                uncertain_lost_upper=progress.get("uncertain_lost_upper", 0),
                pending_unknown_upper=(progress["pending"][1] - saved)
                if progress.get("pending")
                else 0,
            )
        )
    return dict(
        fits=rows,
        journals=journals,
        scientific_saved_updates=sum(v["saved_updates"] for v in rows),
        repeated_or_uncertain_upper=(
            sum(v["accounting"]["scientific_uncertain_lost_upper"] for v in journals.values())
            + sum(v["pending_unknown_upper"] for v in rows)
        ),
        endpoints=sum(v["saved_updates"] == 2500 for v in rows),
        scientific_physical_upper=sum(v["saved_updates"] for v in rows)
        + sum(v["accounting"]["scientific_uncertain_lost_upper"] for v in journals.values())
        + sum(v["pending_unknown_upper"] for v in rows),
        technical=record(NIGHT / "TECHNICAL_WORK.json")
        if (NIGHT / "TECHNICAL_WORK.json").exists()
        else {},
    )


def report(state: dict) -> dict:
    window = record(NIGHT / "WINDOW_AUTHORIZATION.json")
    deadline = datetime.fromisoformat(window["deadline_utc"])
    states = {}
    resultpaths = {
        "N1": H16 / "analysis/RESULTS.json",
        "N2": NIGHT / "execution/analysis/N2/RESULTS.json",
        "N3": NIGHT / "execution/analysis/N3/RESULTS.json",
    }
    for family, path in resultpaths.items():
        fits = [v for v in state["fits"] if v["family"] == family]
        if path.exists():
            results = record(path)
            gaps = results.get("pending_comparators", [])
            states[family] = dict(
                status="COMPLETE"
                if all(v["saved_updates"] == 2500 for v in fits) and not gaps
                else ("ANALYZED_WITH_COMPARATOR_GAPS" if gaps else "PARTIAL_ANALYZED"),
                pending_comparators=gaps,
                result_path=str(path),
                sha256=digest(path),
                conclusion=results.get("conclusion", "EXPLORATORY"),
            )
        else:
            blocks = list(NIGHT.glob(f"BLOCK_{family}*.json"))
            states[family] = dict(
                status="BLOCKED" if blocks else "INCOMPLETE",
                saved_updates=sum(v["saved_updates"] for v in fits),
                endpoints=sum(v["saved_updates"] == 2500 for v in fits),
                blockers=[record(p) for p in blocks],
            )
    cost = (
        record(NIGHT / "N4_COST_STATUS.json")
        if (NIGHT / "N4_COST_STATUS.json").exists()
        else dict(prepared_head="INCOMPLETE", full_route="NOT_MEASURED")
    )
    states["N4"] = cost
    completed = all(states[k]["status"] == "COMPLETE" for k in ("N1", "N2", "N3"))
    reason = (
        "QUEUE_COMPLETE"
        if completed
        else (
            "TEN_HOUR_DEADLINE_REACHED"
            if datetime.now(UTC) >= deadline
            else "CONCRETE_BRANCH_BLOCKS_OR_INCOMPLETE"
        )
    )
    resume = (
        f'& "{ROOT.parent / "e-jepa-ttc/.venv/Scripts/python.exe"}" -B '
        f'"{ROOT / "operational/simplex_t_cost_context/driver.py"}" '
        "--campaign SIMPLEX_T_NOCTURNAL_20261003 --adopt-h16"
    )
    decision = dict(
        schema="nocturnal_delivery_decision_v1",
        termination_reason=reason,
        stages=states,
        endpoints=state["endpoints"],
        scientific_saved_updates=state["scientific_saved_updates"],
        scientific_physical_updates_upper=state["scientific_physical_upper"],
        total_physical_updates_upper=state["scientific_physical_upper"]
        + state["technical"].get("reserved_updates", 0),
        technical=state["technical"],
        repeated_or_uncertain_upper=state["repeated_or_uncertain_upper"],
        authorized_updates_ceiling=60000,
        registered_candidate="TPR-D1-H8-C160",
        registered_candidate_unchanged=True,
        automatic_promotion=False,
        no_push=True,
        raw_reconstruction_included=False,
        independent_units=9,
        incomplete_fits=[v["id"] for v in state["fits"] if v["saved_updates"] != 2500],
        resume_command=resume,
        resume_after_expired_window_requires_new_explicit_window=True,
        next_action=(
            "Review paired precision and measured cost independently; no further fits or "
            "holdouts authorized by this delivery."
        ),
    )
    atomic_json(NIGHT / "NEXT_DECISION_NOCTURNA.json", decision)
    lines = [
        "# Informe nocturno SIMPLEX-T",
        "",
        f"Estado de terminación: **{reason}**.",
        "",
        f"{state['endpoints']}/24 endpoints completos; "
        f"{state['scientific_saved_updates']}/60.000 updates científicos guardados. "
        "Trabajo perdido/repetido/incierto: cota superior "
        f"{state['repeated_or_uncertain_upper']} updates.",
        f"Ventana: {window['accepted_at_europe_madrid']} a "
        f"{window['deadline_europe_madrid']}; Europe/Madrid.",
        "",
        (
            "T6 queda cerrado e intacto. El candidato histórico continúa siendo "
            "TPR-D1-H8-C160. No se han abierto holdouts ni realizado push."
        ),
        "",
        "## Resultados y alcance",
        "",
    ]
    for family in ("N1", "N2", "N3"):
        lines.append(f"- {family}: {json.dumps(states[family], ensure_ascii=False)}")
        if not resultpaths[family].exists():
            continue
        results = record(resultpaths[family])
        lines += [
            "",
            "| Contraste | Candidato MiD | Referencia MiD | Delta de pérdidas | "
            "CI95 jerárquico | CI95 secuencias |",
            "|---|---:|---:|---:|---|---|",
        ]
        for label, contrast in results["comparisons"].items():
            candidate = contrast["h16_mid"] if family == "N1" else contrast["candidate_MiD"]
            reference = contrast["h8_mid"] if family == "N1" else contrast["reference_MiD"]
            lines.append(
                f"| {label} | {candidate:.9f} | {reference:.9f} | "
                f"{contrast['paired_loss_delta']:.9f} | {contrast['hierarchical_ci95']} | "
                f"{contrast['sequence_only']['ci95']} |"
            )
        if family != "N1":
            lines += ["", "MiD de todos los brazos disponibles, sin selección por score:", ""]
            lines.extend(f"- {name}: {score:.9f}." for name, score in results["scores"].items())
        else:
            lines += [
                "",
                "Seed7 es exploratoria ya observada; seeds13/23 son nuevas. Los resúmenes "
                "new13_23 y all7_13_23 promedian pérdidas emparejadas, nunca predicciones TTC.",
            ]
    lines += [
        "",
        (
            "Las cifras nuevas se publican únicamente desde predicciones de endpoints2500 "
            "congelados. Las familias incompletas y métricas ausentes no se completan con "
            "estimaciones. Las comparaciones son exploratorias sobre OLD_DEV reutilizado, "
            "con nueve secuencias independientes; las seeds son réplicas de cabezas, no "
            "de todos los expertos."
        ),
        "",
        (
            "El inventario acotado BASELINE_REGISTRY.csv registra 22 comparadores y "
            "distingue contrato, información disponible y evidencia insuficiente. Una "
            "mejor cifra de otra cohorte no establece superioridad comparable."
        ),
        "",
        "## Precisión y coste",
        "",
        f"N4: {json.dumps(cost, ensure_ascii=False)}",
        "",
        (
            "HEAD_COST.csv, cuando existe, mide sólo la cabeza con entradas preparadas, "
            "batch1 y batch128 por separado. No mide el coste del contexto del ROI actual "
            "ni permite afirmar ahorro del sistema, reacción AEB más rápida, tracking "
            "persistente o incertidumbre calibrada."
        ),
        "",
        "## Contabilidad y recuperación",
        "",
        (
            "ENDPOINT_INVENTORY.json conserva cada checkpoint y su SHA-256. Los journals "
            "PHYSICAL_WORK.json distinguen updates guardados y trabajo incierto. La "
            "prueba real H16 restaura el checkpoint100 y continúa desde101 sin "
            "entrenamiento duplicado. Las pruebas sintéticas del estudio nuevo tienen "
            "contabilidad separada y no son resultados de eAP."
        ),
        "",
        (
            "Se conserva CPU FP32/batch128/cuatro threads/interop2. RAM disponible "
            "mínima2GiB, RSS máximo4GiB y margen10GB tras reserva1GiB; techo de "
            "artefactos propios2GiB. La revisión operativa pre-update preservó v1 y no "
            "cambió la receta H16."
        ),
        "",
        "## Bundle y regeneración",
        "",
        (
            "El manifiesto relaciona bytes y SHA-256 de todos los miembros; el ZIP se "
            "verifica por CRC y extracción independiente. Los inputs normalizados y pesos "
            "incluidos permiten regenerar las cabezas incluidas. No contiene todos los "
            "datos raw/TRAIN ni pesos de expertos, y no promete reconstrucción desde "
            "eventos."
        ),
        "",
        (
            "Las pruebas/recibos declaran el alcance exacto comprobado. Si no hay cabezas "
            "completas/publicadas, la comprobación del bundle es de integridad de "
            "archivos y checkpoints recuperables, sin scores científicos nuevos."
        ),
        "",
        "## Pendiente",
        "",
        *[
            f"- {v['id']}: {v['saved_updates']}/2500 guardados; {v['status']}."
            for v in state["fits"]
            if v["saved_updates"] != 2500
        ],
        "",
        "Comando local de recuperación con campaña precisa:",
        "",
        "```powershell",
        resume,
        "```",
        "",
        (
            "Tras vencer la ventana, este comando conserva la pausa: más entrenamiento "
            "requiere nueva autorización temporal explícita. No habilita automáticamente "
            "otros fits."
        ),
    ]
    cost_path = NIGHT / "HEAD_COST.csv"
    if cost_path.exists():
        import csv

        lines += [
            "",
            "## Latencia medida con entradas preparadas",
            "",
            "| Modelo | p50 ms | p95 ms | Medidas batch1 | Parámetros |",
            "|---|---:|---:|---:|---:|",
        ]
        with cost_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                lines.append(
                    f"| {row['model']} | {float(row['p50_ms']):.6f} | "
                    f"{float(row['p95_ms']):.6f} | {row['measurements']} | "
                    f"{row['parameters']} |"
                )
        lines += [
            "",
            "Estas medidas incluyen la emisión TTC de la cabeza; excluyen generar contexto "
            "y ejecutar expertos. COST_ACCURACY.csv aplica el cribado de precisión/p95 "
            "con el mismo alcance y conserva system_substitution_supported=false.",
        ]
    atomic_bytes(NIGHT / "INFORME_NOCTURNO_SIMPLEX_T.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return decision


def tables() -> None:
    import pandas as pd

    metric = NIGHT / "METRICS_COSTE_CONTEXTO.csv"
    if not metric.exists():
        atomic_bytes(metric, b"arm,comparison,status,mid,delta,ci95_lower,ci95_upper,scope\n")
    frame = pd.read_csv(metric, float_precision="round_trip")  # type: ignore[call-overload]
    cost = NIGHT / "HEAD_COST.csv"
    if cost.exists() and "candidate" in frame.columns:
        c = pd.read_csv(cost, float_precision="round_trip")  # type: ignore[call-overload]
        c.loc[c.model == "H8_SEED7", "model"] = "H8@7"
        frame = frame.merge(c, left_on="candidate", right_on="model", how="left")
        reference = c.loc[:, ["model", "p95_ms", "selection_sha256"]].rename(
            columns={
                "model": "reference",
                "p95_ms": "reference_p95_ms",
                "selection_sha256": "reference_selection_sha256",
            }
        )
        frame = frame.merge(reference, on="reference", how="left")
        frame["head_p95_ratio"] = frame.p95_ms / frame.reference_p95_ms
        frame["same_profile_inputs"] = frame.selection_sha256 == frame.reference_selection_sha256
        frame["head_cost_screen_pass"] = frame.head_p95_ratio.le(0.8) & frame.same_profile_inputs
        frame["head_only_followup_screen_pass"] = (
            frame.precision_screen_pass.eq(True) & frame.head_cost_screen_pass
        )
        frame["system_substitution_supported"] = False
    if "scope" not in frame.columns:
        frame["scope"] = "prepared_input_head_only_if_measured"
    atomic_bytes(
        NIGHT / "COST_ACCURACY.csv", frame.to_csv(index=False, float_format="%.17g").encode()
    )


def selected_files() -> dict[str, Path]:
    files = {}
    for root, prefix in ((NIGHT, ""), (H16, "h16/")):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(
                v in {"independent_extraction", "essential_delivery", "delivery", "driver_receipts"}
                for v in relative.parts
            ):
                continue
            if path.suffix in {".lock", ".pending", ".tmp", ".zip"} or path.name in {
                "PROGRESS.json",
                "ACTIVITY.md",
                "ACTIVE_CHILD.json",
                "DRIVER.log",
                "RESOURCES.json",
                "FINAL_DELIVERY_NOCTURNA.json",
                "CONTENT_MANIFEST.json",
            }:
                continue
            files[prefix + relative.as_posix()] = path
    # Include closed worker logs, never the currently growing delivery log.
    for receipt in sorted((NIGHT / "driver_receipts").rglob("attempt_*.json")):
        if receipt.parent.name == "N5_DELIVERY":
            continue
        log = receipt.with_suffix(".log")
        if log.exists() and digest(log) == record(receipt)["log_sha256"]:
            for path in (receipt, log):
                files[path.relative_to(NIGHT).as_posix()] = path
    for path in (H16 / "essential_delivery/historical_controls").glob("*.parquet"):
        files["h16/historical_controls/" + path.name] = path
    p16 = record(H16 / "PROTOCOL.json")
    for row in record(Path(p16["launch"]["freeze"]))["files"]:
        if (
            row["category"] == "code"
            and row["root"] == "work"
            and row["relative_path"].startswith("src/")
        ):
            path = ROOT / row["relative_path"]
            if digest(path) != row["sha256"]:
                raise ValueError("historical scientific source changed")
            files["provenance/" + row["relative_path"]] = path
    for directory in ("simplex_t_cost_context", "simplex_t_h16_replication"):
        for path in (ROOT / "operational" / directory).glob("*"):
            if path.is_file():
                files["provenance/operational/" + directory + "/" + path.name] = path
    files["regenerate.py"] = ROOT / "operational/simplex_t_cost_context/regenerate.py"
    files["verify_checkpoints.py"] = (
        ROOT / "operational/simplex_t_cost_context/verify_checkpoints.py"
    )
    return files


def fragmented_zip(
    archive: Path,
    files: dict[str, Path],
    manifest: dict,
    manifest_bytes: bytes,
    resources: ResourceGuard,
) -> None:
    """Commit32 members per durable ZIP fragment, retaining the preceding valid ZIP."""
    pending = archive.with_suffix(".zip.pending")
    backup = archive.with_suffix(".zip.checkpoint")
    journal = archive.with_suffix(".publication.json")
    entries = list(files)
    try:
        with zipfile.ZipFile(pending) as z:
            completed = set(z.namelist())
            if len(completed) != len(z.namelist()) or z.testzip() is not None:
                raise zipfile.BadZipFile("partial ZIP integrity")
    except (FileNotFoundError, zipfile.BadZipFile):
        if pending.exists():
            pending.replace(pending.with_name(pending.name + ".damaged." + digest(pending)[:12]))
        if backup.exists():
            shutil.copyfile(backup, pending)
            with zipfile.ZipFile(pending) as z:
                completed = set(z.namelist())
        else:
            completed = set()
    if not completed.issubset(set(entries) | {"CONTENT_MANIFEST.json"}):
        raise ValueError("publication contains unregistered members")
    remaining = [name for name in entries if name not in completed]
    for start in range(0, len(remaining), 32):
        resources.check()
        chunk = remaining[start : start + 32]
        with zipfile.ZipFile(
            pending, "a" if pending.exists() else "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as z:
            for name in chunk:
                path = files[name]
                if digest(path) != manifest["members"][name]["sha256"]:
                    raise ValueError(f"member changed while publishing: {name}")
                z.write(path, name)
        with pending.open("r+b") as stream:
            os.fsync(stream.fileno())
        snapshot = backup.with_name(backup.name + ".next")
        shutil.copyfile(pending, snapshot)
        with snapshot.open("r+b") as stream:
            os.fsync(stream.fileno())
        snapshot.replace(backup)
        completed.update(chunk)
        atomic_json(
            journal,
            dict(
                status="PUBLICATION_FRAGMENT_COMMITTED",
                completed_members=len(completed),
                total_members=len(entries) + 1,
                pending_sha256=digest(pending),
                checkpoint_sha256=digest(backup),
            ),
        )
        print(
            json.dumps(dict(status="PUBLICATION_FRAGMENT_COMMITTED", members=len(completed))),
            flush=True,
        )
    if "CONTENT_MANIFEST.json" not in completed:
        with zipfile.ZipFile(
            pending, "a" if pending.exists() else "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as z:
            z.writestr("CONTENT_MANIFEST.json", manifest_bytes)
    with pending.open("r+b") as stream:
        os.fsync(stream.fileno())
    pending.replace(archive)


def bundle(files: dict[str, Path], decision: dict) -> dict:
    from .engine import Resources

    resources = Resources()
    resources.check()
    manifest: dict = dict(
        schema="essential_nocturnal_manifest_v1",
        members={name: dict(bytes=p.stat().st_size, sha256=digest(p)) for name, p in files.items()},
        scope=(
            "Included normalized cached heads and complete recoverable checkpoints; no "
            "raw/TRAIN reconstruction."
        ),
        optimizer_updates_for_regeneration=0,
    )
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    delivery = NIGHT / "delivery"
    delivery.mkdir(exist_ok=True)
    archive = (
        delivery / f"E_JEPA_TTC_SIMPLEX_T_NOCTURNO_{state_count(decision)}_{manifest_sha[:12]}.zip"
    )
    extract = delivery / ("extracted_" + manifest_sha[:12])
    uncompressed = sum(row["bytes"] for row in manifest["members"].values()) + len(manifest_bytes)
    # Conservative DEFLATE/ZIP bound, including filenames and both headers.
    zip_upper = (
        uncompressed
        + uncompressed // 1000
        + 65536
        + sum(1024 + 2 * len(name.encode("utf-8")) for name in files)
    )
    peak = max(3 * zip_upper, 2 * zip_upper + uncompressed)
    recognized = [
        archive,
        archive.with_suffix(".zip.pending"),
        archive.with_suffix(".zip.checkpoint"),
        archive.with_suffix(".zip.checkpoint").with_name(archive.name + ".checkpoint.next"),
    ]
    existing = sum(p.stat().st_size for p in recognized if p.is_file())
    if extract.exists():
        existing += sum(p.stat().st_size for p in extract.rglob("*") if p.is_file())
    resources.reserve_artifacts(max(0, peak - existing), "NOCTURNAL_BUNDLE_PUBLICATION_PEAK")
    if not archive.exists():
        fragmented_zip(archive, files, manifest, manifest_bytes, resources)
    sha = digest(archive)
    atomic_bytes(archive.with_suffix(".zip.sha256"), (sha + "  " + archive.name + "\n").encode())
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError("archive duplicate/CRC failure")
        for member in z.infolist():
            resources.check()
            target = (extract / member.filename).resolve()
            if not target.is_relative_to(extract.resolve()):
                raise ValueError("unsafe essential archive member")
            expected = (
                manifest_sha
                if member.filename == "CONTENT_MANIFEST.json"
                else manifest["members"][member.filename]["sha256"]
            )
            with z.open(member) as stream:
                durable_stream(target, stream, expected)
    receipt: dict = dict(
        archive=str(archive),
        sha256=sha,
        bytes=archive.stat().st_size,
        content_manifest_sha256=manifest_sha,
        verified_members=len(files) + 1,
        all_sha256_verified=True,
        crc_verified=True,
        extraction_verified=True,
        termination_reason=decision["termination_reason"],
        head_regeneration="NO_COMPLETE_PUBLISHED_NEW_HEADS"
        if not (extract / "execution/ANALYSIS_EXPORT_INDEX.json").exists()
        else "PENDING",
    )
    checkpoint_receipt = delivery / ("CHECKPOINT_VERIFICATION_" + manifest_sha[:12] + ".json")
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            str(extract / "verify_checkpoints.py"),
            "--root",
            str(extract),
            "--output",
            str(checkpoint_receipt),
        ],
        cwd=extract,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    atomic_bytes(checkpoint_receipt.with_suffix(".log"), (child.stdout + child.stderr).encode())
    receipt["checkpoint_verification"] = (
        "VERIFIED" if child.returncode == 0 else "BLOCKED_OR_FAILED"
    )
    if checkpoint_receipt.exists():
        receipt["checkpoint_verification_receipt_sha256"] = digest(checkpoint_receipt)
    if (extract / "execution/ANALYSIS_EXPORT_INDEX.json").exists():
        output = delivery / ("REGENERATION_" + manifest_sha[:12] + ".json")
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                str(extract / "regenerate.py"),
                "--root",
                str(extract),
                "--output",
                str(output),
            ],
            cwd=extract,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        atomic_bytes(output.with_suffix(".log"), (child.stdout + child.stderr).encode())
        receipt["head_regeneration"] = "VERIFIED" if child.returncode == 0 else "BLOCKED_OR_FAILED"
        receipt["regeneration_returncode"] = child.returncode
        if output.exists():
            receipt["regeneration_receipt_sha256"] = digest(output)
    h16regen = H16 / "REGENERATION.json"
    receipt["h16_separate_regeneration_receipt_sha256"] = (
        digest(h16regen) if h16regen.exists() else None
    )
    return receipt


def state_count(decision: dict) -> str:
    return str(decision["scientific_saved_updates"]).zfill(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from .engine import EXEC, Resources, _owner

    if any(_owner(path) is not None for path in (H16 / "WRITER.lock", EXEC / "WRITER.lock")):
        raise RuntimeError("publication cannot reconcile a live training writer")
    Resources().check()
    state = inventory(admit_physical_checkpoints=True)
    atomic_json(
        NIGHT / "CHECKPOINT_JOURNAL_RECONCILIATION.json",
        dict(
            status="READ_ONLY_PHYSICAL_CHECKPOINT_ADMISSION",
            journal_files_unchanged=True,
            fits=[
                {k: r[k] for k in ("id", "journal_saved_updates", "saved_updates")}
                for r in state["fits"]
                if r["checkpoint_journal_reconciled"]
            ],
            optimizer_updates=0,
        ),
    )
    atomic_json(NIGHT / "ENDPOINT_INVENTORY.json", state)
    if (NIGHT / "ACTIVITY.md").exists():
        atomic_bytes(NIGHT / "DELIVERY_ACTIVITY.md", (NIGHT / "ACTIVITY.md").read_bytes())
    decision = report(state)
    tables()
    # The parent continues logging and our guard updates RESOURCES during ZIP
    # fragments. Bind immutable copies, never hash those changing source files.
    for source, target in (
        (NIGHT / "DRIVER.log", NIGHT / "DELIVERY_DRIVER_LOG.log"),
        (EXEC / "RESOURCES.json", NIGHT / "DELIVERY_RESOURCES.json"),
    ):
        if source.exists() and not target.exists():
            atomic_bytes(target, source.read_bytes())
    receipt = bundle(selected_files(), decision)
    atomic_json(NIGHT / "FINAL_DELIVERY_NOCTURNA.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
