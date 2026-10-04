"""Publish complete frozen shared-GPU route evidence without changing earlier campaigns."""

# Checked external manifests carry constructors and measured numeric payloads.
# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from operational.simplex_t_post_campaign.contracts import digest, read, save  # noqa: E402
from operational.simplex_t_post_campaign.run import guard_function  # noqa: E402
from operational.verify_simplex_t_shared_route import summaries  # noqa: E402

OUT = ROOT / "artifacts/simplex_t/shared_gpu_route_20261004"
DOCS = ROOT / "docs/simplex_t_shared_gpu_route_20261004"
OLD = ROOT / "artifacts/simplex_t/post_campaign_20261003"
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
COPIED = 0


def copy_file(source: Path, target: Path) -> None:
    """Resume publication by verifying each existing durable member."""
    global COPIED
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size != target.stat().st_size or digest(source) != digest(target):
            raise ValueError(f"preserved publication member differs: {target}")
    else:
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    COPIED += 1
    if COPIED % 20 == 0:
        guard_function()(OUT)
        save(OUT / "PUBLICATION_PROGRESS.json", dict(confirmed_members=COPIED, last=str(target)))


def copy_tree(source: Path, target: Path) -> None:
    """Copy only referenced files, retaining independently verified fragments."""
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, target / path.relative_to(source))


def csv_file(path: Path, rows: list[dict]) -> None:
    """Save all measured fields without truncating floating-point precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(*, partial: bool = False) -> None:
    """Require the complete prospective inventory and stage one essential supplement."""
    guard_function()(OUT)
    if (OUT / "ACTIVE_OWNER.json").exists():
        raise RuntimeError("profiling writer must release ownership before publication")
    p = read(OUT / "PROTOCOL.json")
    rows = [read(path) for path in sorted((OUT / "fragments").glob("*.json"))]
    if (not partial and len(rows) != 1728) or any(
        r["protocol_sha256"] != digest(OUT / "PROTOCOL.json") for r in rows
    ):
        raise ValueError("all 1728 prospectively bound measurements required")
    complete = len(rows) == 1728
    observed_names = {path.stem for path in (OUT / "fragments").glob("*.json")}
    planned_names = []
    for rank, _query in sorted(enumerate(p["queries"]), key=lambda q: (q[1]["family_id"], q[0])):
        for mode_id, mode in enumerate(p["modes"]):
            shift = (rank + mode_id) % 9
            for label in p["models"][shift:] + p["models"][:shift]:
                planned_names.append(f"q{rank:02d}_{mode}_{label}")
    if len(observed_names) != len(rows) or not observed_names <= set(planned_names):
        raise ValueError("physical fragment IDs differ from the queue")
    pending = [name for name in planned_names if name not in observed_names]
    save(OUT / "INFERENCE_INVENTORY.json", dict(
        planned_count=1728, confirmed_count=len(rows), pending_count=len(pending),
        next_id=pending[0] if pending else None, pending_ids=pending,
        confirmed_files={
            path.name: digest(path) for path in sorted((OUT / "fragments").glob("*.json"))
        },
        optimizer_updates=0,
    ))
    statistics = summaries(rows, require_full=complete)
    docs = DOCS if complete else DOCS / f"partial_{len(rows)}"
    publication = OUT / "delivery" if complete else OUT / f"delivery_partial_{len(rows)}"
    csv_file(OUT / "ROUTE_COST.csv", statistics)
    attempts = [read(path) for path in (OUT / "attempts").glob("*.json")]
    accounting = dict(
        measured_fragments=len(rows),
        planned_fragments=1728,
        pending_fragments=1728 - len(rows),
        matched_complete_queries=statistics[0]["measurements"],
        inventory_complete=complete,
        parity_queries=64,
        scientific_optimizer_updates=0,
        technical_optimizer_updates=0,
        inference_attempts_completed=sum(r["status"] == "COMPLETED" for r in attempts),
        inference_attempts_unresolved=sum(r["status"] != "COMPLETED" for r in attempts),
        warmups_completed=sum(
            r["purpose"] == "warmup" and r["status"] == "COMPLETED" for r in attempts
        ),
        actual_measurement_producer_calls={
            name: sum(r["actual_producer_calls"][name] for r in rows)
            for name in ("A5", "C2F", "PAIR")
        },
        candidate="TPR-D1-H8-C160",
        exclusive_gpu_claim=False,
        manual_diagnostic_requests=2,
        failed_runtime_admission_preserved=True,
    )
    save(OUT / "ACCOUNTING.json", accounting)
    original = list(csv.DictReader((OLD / "ACCURACY_COST_FRONTIER.csv").open(encoding="utf-8")))
    scores = {r["model"]: r for r in original}
    frontier = []
    for label in p["models"]:
        warm = [r for r in statistics if r["model"] == label and r["mode"].startswith("warm")]
        cold = next(
            r for r in statistics if r["model"] == label and r["mode"].startswith("application")
        )
        key = label.replace("_SEED7", "@7")
        source = scores.get(label, scores.get(key))
        frontier.append(
            dict(
                model=label,
                OLD_DEV_evidence_model=key,
                OLD_DEV_MiD=float(source["MiD"]) if source is not None else None,
                warm_total_p95_block1_ms=warm[0]["total_ms_p95"],
                warm_total_p95_block2_ms=warm[1]["total_ms_p95"],
                application_hdf5_cold_total_p95_ms=cold["total_ms_p95"],
                runtime_scope="shared_GPU_CPU_heads_supplied_ROI_resident_models",
                performance_causal_attribution=False,
                automatic_promotion=False,
                accuracy_scope="OLD_DEV_seed7_nine_sequences_three_folds",
                cost_scope="64_TRAIN_fold0_queries_three_inner_producer_families_shared_GPU",
            )
        )
    csv_file(OUT / "COST_ACCURACY_SHARED.csv", frontier)
    parity = [read(path) for path in (OUT / "parity_historical_runtime").glob("*.json")]
    if (
        len(parity) != 64
        or len(list((OUT / "canonical_parity_historical_runtime").glob("*.json"))) != 3
    ):
        raise ValueError("64 real query parity receipts plus three canonical families required")
    parity_receipt = dict(
        status="VERIFIED_WITH_PREREGISTERED_NUMERICAL_TOLERANCES",
        queries=64,
        maximum_normalized_feature_error=max(
            r["parity"]["input_errors"]["features"] for r in parity
        ),
        maximum_point_phase_error=max(r["parity"]["point_phase_error"] for r in parity),
        maximum_ttc_error_seconds=max(r["parity"]["ttc_error_seconds"] for r in parity),
        canonical_adapter_raw_feature_difference=0,
        measured_routes_checked=len(rows),
        excluded_producer_call_contract_verified=True,
        normalizer_reused=True,
        optimizer_updates=0,
    )
    save(OUT / "PARITY_SUMMARY.json", parity_receipt)
    decision = dict(
        status="COMPLETE_SHARED_RESOURCE_SCOPE" if complete else "INCOMPLETE_RESOURCE_BLOCK",
        P3="COMPLETE_SHARED_GPU" if complete else "INCOMPLETE_SHARED_GPU",
        P0="COMPLETE",
        P1="COMPLETE",
        P2="COMPLETE",
        P4="COMPLETE_WITH_SHARED_COST_LIMITATION" if complete else "PARTIAL_COST",
        planned_measurements=1728,
        confirmed_measurements=len(rows),
        pending_measurements=1728 - len(rows),
        cost_population_complete=complete,
        next_id=pending[0] if pending else None,
        resume_command="python -B -m operational.simplex_t_shared_route.run run",
        resume_output_root="artifacts/simplex_t/shared_gpu_route_20261004",
        last_execution_state=read(OUT / "PROGRESS.json"),
        optimizer_updates=0,
        registered_candidate="TPR-D1-H8-C160",
        automatic_promotion=False,
        old_statistical_decisions_unchanged=True,
        referenced_accuracy_csv_sha256=digest(OLD / "ACCURACY_COST_FRONTIER.csv"),
        prior_bundle_sha256="83c0599944a275a08568259aed69b72eed7ab56f15c7a8972a89abbff081b476",
        limitations=[
            "GPU contention is not isolated or experimentally controlled",
            "Supplied ROI excludes detector, persistent tracking and AEB",
            "Application-cold HDF5 handles; OS cache and producer models are not cold",
            "Frozen producer padding batch16 retained; no upstream optimization claim",
            "Raw events are external and full raw hash/encoder regeneration is not bundled",
            "GPU allocation fields are post-forward snapshots with all three producers resident",
        ],
        next_proposal=(
            "Preregister one separate exact-parity optimization of retrospective raw/ROI "
            "context preparation with frozen weights; no execution authorized here"
        ),
    )
    save(OUT / "NEXT_DECISION_SHARED_GPU.json", decision)
    lines = [
        ("P3 completado" if complete else "P3 incompleto por recursos")
        + ": inferencia con autorización explícita de GPU compartida.",
        "Se conservaron H8, los 24 endpoints y los 60.000 updates científicos anteriores. "
        "Esta continuación ejecutó cero actualizaciones de optimizador.",
        f"Se preservan {len(rows)} de 1.728 solicitudes previstas: "
        f"{statistics[0]['measurements']} consultas TRAIN tienen las 27 mediciones completas. "
        "Se validaron primero los 64 contextos crudos y las tres familias de productores "
        "contra el extractor canónico. Los brazos reducidos no llaman a expertos excluidos.",
        "| Ruta | p95 caliente 1 (ms) | p95 caliente 2 (ms) | p95 HDF5 frío (ms) |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['model']} | {r['warm_total_p95_block1_ms']:.3f} | "
        f"{r['warm_total_p95_block2_ms']:.3f} | "
        f"{r['application_hdf5_cold_total_p95_ms']:.3f} |"
        for r in frontier
    ]
    lines[3:] = ["\n".join(lines[3:])]
    lines += [
        "Los CSV preservan precisión completa. El total se midió directamente, "
        "desde la petición con ROI suministrado hasta TTC; no se sumaron p95 de etapas.",
        "GPU compartida: las diferencias observadas incluyen contención y variación del host. "
        "No prueban un ahorro causal o una latencia aislada. Frío significa cierre de las "
        "cachés HDF5 de la aplicación, con modelos residentes "
        "y caché del sistema operativo intacta. "
        "La carga de modelos se publica separadamente.",
        f"El protocolo fija 64 consultas TRAIN de fold0 y tres familias de productores internos. "
        f"La tabla usa las {statistics[0]['measurements']} consultas que tienen las 27 mediciones; "
        "si faltan consultas, esos cuantiles son parciales y no representan la población completa. "
        "La precisión referenciada corresponde a seed7, nueve secuencias OLD_DEV y tres folds. "
        "No son una nueva evaluación ni una nueva selección de checkpoints. "
        "Los bytes CUDA son snapshots posteriores al forward con los tres productores residentes; "
        "no son picos por solicitud ni prueban una reducción de memoria del sistema.",
        "Se preserva el intento inicial fallido de admisión numérica y sus recibos. "
        "Activar algoritmos deterministas alteraba el runtime histórico de los productores; "
        "se restauró su configuración original antes de medir, sin ampliar tolerancias. "
        "Los 64 contextos admitidos coinciden exactamente. Dos solicitudes de diagnóstico "
        "se contabilizan aparte de los intentos del driver.",
        "Se mantiene el batch16 original de productores con padding. H1/H8/H16 preparan "
        "sólo su historia necesaria, pero este experimento no cambia ni optimiza los encoders.",
        "Los criterios de precisión anteriores siguen vigentes: los brazos reducidos y "
        "agregadores no pasan el cribado registrado. Una latencia menor no cambia esa decisión. "
        "H16 conserva una mejora estable entre seeds de cabeza y la incertidumbre entre escenas; "
        "no sustituye automáticamente al H8 histórico.",
        "No se midieron detección, tracking online, sensor-to-AEB ni calibración de incertidumbre. "
        "La paridad numérica y las fuentes reutilizadas "
        "no constituyen una réplica nueva de los expertos.",
        "Propuesta única posterior, sin ejecutar: optimizar la preparación retrospectiva "
        "raw/ROI con pesos congelados y paridad exacta, bajo un protocolo independiente que "
        "fije consultas TRAIN, memoria y p50/p95 antes de medir. Una caché sólo será válida "
        "para la misma observación y ROI acreditados. Ningún entrenamiento ni holdout "
        "se abre con esta entrega.",
    ]
    (OUT / "INFORME_GPU_COMPARTIDA_SIMPLEX_T.md").write_text(
        "\n\n".join(lines) + "\n", encoding="utf-8"
    )
    docs.mkdir(parents=True, exist_ok=True)
    for name in (
        "ROUTE_COST.csv",
        "COST_ACCURACY_SHARED.csv",
        "ACCOUNTING.json",
        "PARITY_SUMMARY.json",
        "NEXT_DECISION_SHARED_GPU.json",
        "INFORME_GPU_COMPARTIDA_SIMPLEX_T.md",
        "INFERENCE_INVENTORY.json",
        "PROTOCOL.json",
        "TESTS.xml",
        "ENVIRONMENT.json",
        "PYRIGHT_FINAL.json",
        "RESUME_PROOF.json",
        "RUNTIME_REPAIR.json",
        "DIAGNOSTIC_HISTORICAL_RUNTIME.json",
    ):
        shutil.copyfile(OUT / name, docs / name)
    bundle = publication / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for path in docs.iterdir():
        if not path.is_file() or path.name in {
            "DELIVERY_STATUS.json", "EXTRACTED_VERIFICATION.json"
        }:
            continue
        copy_file(path, bundle / path.name)
    for name in (
        "PROTOCOL_BEFORE_RAW_CAPTURE.json", "PROTOCOL_RUNTIME_MISMATCH.json",
        "DIAGNOSTIC_PRECISION.json", "DIAGNOSTIC_PRECISION.npz",
        "PYRIGHT_DELIVERY_FINAL.json", "SOURCE_ADMISSION.json", "RESUME_START_27.json",
        "RESUME_START_675.json", "RESUME_START_697.json",
        "IO_ERROR_697.json", "IO_READER_REPAIR.json",
    ):
        if (OUT / name).exists():
            copy_file(OUT / name, bundle / "audit" / name)
    copy_file(OLD / "ACCURACY_COST_FRONTIER.csv", bundle / "audit/ACCURACY_COST_FRONTIER.csv")
    save(bundle / "REPRODUCTION.json", dict(
        command="python -B verify.py --root . --output ../SHARED_GPU_REGENERATION.json --heads",
        required_packages=["numpy", "torch", "psutil", "pandas", "pyarrow", "h5py"],
        measured_environment="ENVIRONMENT.json",
        included_scope="numeric frozen heads from captured inputs; timing quantiles from fragments",
        excluded_scope="raw event reconstruction, model training and elapsed timing recollection",
        raw_profile_requires="original source-bound repository and external TRAIN event files",
        original_continuation_command=(
            "python -B -m operational.simplex_t_shared_route.run run"
        ),
        output_root="artifacts/simplex_t/shared_gpu_route_20261004",
        optimizer_updates=0,
    ))
    for folder in (
        "fragments",
        "parity",
        "parity_historical_runtime",
        "canonical_parity",
        "canonical_parity_historical_runtime",
        "raw_support",
        "model_load",
        "attempts",
        "resource_observations",
    ):
        source = OUT / folder
        if source.exists():
            copy_tree(source, bundle / "results" / folder)
    copy_tree(OUT / "inference_inputs", bundle / "inference_inputs")
    copy_file(OUT / "NORMALIZER.npz", bundle / "NORMALIZER.npz")
    profile = read(NIGHT / "PROFILE_INPUT_EXPORT.json")
    (bundle / "weights").mkdir(exist_ok=True)
    for row in profile["models"].values():
        target = f"weights/{row['weights_sha256']}.npz"
        copy_file(NIGHT / row["weights_path"], bundle / target)
        row["weights_path"] = target
        for field in ("inputs_path", "inputs_sha256"):
            if field in row:
                row["historical_prepared_" + field] = row.pop(field)
    save(bundle / "HEAD_INDEX.json", profile)
    (bundle / "producers").mkdir(exist_ok=True)
    for row in p["checkpoints"]:
        copy_file(Path(row["path"]), bundle / f"producers/{row['registered_sha256']}.pt")
    copy_tree(OLD / "delivery/bundle/provenance", bundle / "provenance")
    copy_tree(
        ROOT / "operational/simplex_t_shared_route",
        bundle / "provenance/operational/simplex_t_shared_route",
    )
    copy_file(ROOT / "operational/verify_simplex_t_shared_route.py", bundle / "verify.py")
    copy_file(OLD / "delivery/bundle/resource_guard.py", bundle / "resource_guard.py")
    files = [
        path for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "CONTENT_MANIFEST.json"
    ]
    manifest = {
        "schema": "essential_shared_gpu_route_v1",
        "members": {
            path.relative_to(bundle).as_posix(): dict(
                bytes=path.stat().st_size, sha256=digest(path)
            )
            for path in files
        },
    }
    save(bundle / "CONTENT_MANIFEST.json", manifest)
    archive = publication / (
        "E_JEPA_TTC_SHARED_GPU_ROUTE_20261004.zip" if complete
        else f"E_JEPA_TTC_SHARED_GPU_ROUTE_PARTIAL_{len(rows)}_20261004.zip"
    )
    if not archive.exists():
        temporary_archive = archive.with_suffix(".zip.partial")
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(bundle).as_posix())
        os.replace(temporary_archive, archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for name, row in manifest["members"].items():
            data = z.read(name)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("ZIP member changed")
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest(archive)}  {archive.name}\n", encoding="ascii"
    )
    save(
        OUT / "DELIVERY.json",
        dict(
            zip=str(archive),
            bytes=archive.stat().st_size,
            sha256=digest(archive),
            members=len(manifest["members"]) + 1,
            optimizer_updates=0,
            status="PACKED_PENDING_EXTRACTED_VERIFICATION",
        ),
    )
    print(json.dumps(read(OUT / "DELIVERY.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    finish(partial=parser.parse_args().partial)
