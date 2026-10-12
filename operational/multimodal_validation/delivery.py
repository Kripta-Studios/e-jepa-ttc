"""Verify completed inference artifacts, summarize measured costs and build the bundle."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import psutil

from .common import OUT, REG, ROOT, RUN, V12, V13, read, registrations, sha, write
from .score import sealed


def snapshot() -> dict:
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "operational.range_readout" in command:
                processes.append(
                    dict(pid=process.pid, command=command, rss=process.info["memory_info"].rss)
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    other = ROOT.parent / "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012"
    return dict(
        utc=datetime.now(UTC).isoformat(),
        v14=read(
            ROOT.parent
            / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010/NEXT_DECISION.json"
        ),
        v15_active_stage=read(other / "ACTIVE_STAGE.json"),
        v15_processes=processes,
        v15_state_is_snapshot_not_campaign_completion=True,
        git={
            str(p): subprocess.check_output(
                ["git", "-C", str(p), "status", "--short"], text=True, encoding="utf-8"
            )
            for p in (ROOT, V13)
        },
        optimizer_updates=0,
    )


def cost_review() -> dict:
    stage = pd.read_csv(OUT / "cpu_profile/STAGE_ATTRIBUTION.csv")
    total = cast(pd.DataFrame, stage.loc[stage.stage == "TOTAL_SHARED_E_R_F_ROUTES"])
    assert len(total) == 128
    assert not total.duplicated(["scope", "query_id", "repeat"]).any()
    garl = pd.read_csv(OUT / "GARL_CPU_COST.csv")
    assert len(garl) == 128 and garl.bitwise_parity.all()
    assert not garl.duplicated(["cohort", "query_id", "iteration"]).any()
    assert set(zip(total.scope, total.query_id, total["repeat"], strict=True)) == set(
        zip(garl.cohort, garl.query_id, garl.iteration, strict=True)
    )
    rows = []
    for key, frame in stage.groupby(["scope", "repeat", "stage"]):
        cohort, repeat, name = cast(tuple, key)
        values = frame.model_ms.to_numpy()
        rows.append(
            dict(
                route="shared_E_R_F",
                cohort=cohort,
                repeat=int(repeat),
                stage=name,
                calls=len(values),
                **dict(
                    zip(("p50", "p95", "p99"), np.quantile(values, [0.5, 0.95, 0.99]), strict=True)
                ),
            )
        )
    for key, frame in garl.groupby(["cohort", "iteration"]):
        cohort, repeat = cast(tuple, key)
        for name in [c for c in garl if c.endswith("_ms")]:
            values = frame[name].to_numpy()
            rows.append(
                dict(
                    route="Garl_full_fallback_event",
                    cohort=cohort,
                    repeat=int(repeat),
                    stage=name,
                    calls=len(values),
                    **dict(
                        zip(
                            ("p50", "p95", "p99"),
                            np.quantile(values, [0.5, 0.95, 0.99]),
                            strict=True,
                        )
                    ),
                )
            )
    result = dict(
        device="CPU_FP32_1_thread",
        queries=64,
        repetitions=2,
        concurrent_V15=True,
        model_load=read(OUT / "cpu_profile/MODEL_LOAD_COST.json"),
        Garl_load_ms=read(OUT / "GARL_CPU_COST_BINDING.json")["model_load_ms"],
        per_stage_milliseconds=rows,
        V12_prepared_heads=read(OUT / "V12_HEAD_COST.json"),
        isolated_GPU_service_benchmark="PENDING_V15_RESOURCE_RELEASE_AND_SERVICE_PERIMETER",
        optimizer_updates=0,
    )
    lines = [
        "# Coste medido: CPU FP32, un hilo",
        "",
        "64 identidades idénticas entre pilotos, 32 Dev32 y 32 FCWD; dos pasadas. "
        "V15 siguió activo: no es GPU aislada ni SLA de despliegue. Primera pasada no "
        "significa disco sin caché. Cada fila resume duraciones observadas, sin sumar medianas.",
        "",
        "| Ruta / etapa, segunda pasada | Cohorte | n | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selected = {
        "TOTAL_SHARED_E_R_F_ROUTES",
        "E_CTX_ROUTE_END_TO_END",
        "REGISTRATION_SEARCH",
        "total_raw_system_ms",
        "resident_network_ms",
    }
    for r in rows:
        if r["repeat"] == 1 and r["stage"] in selected:
            lines.append(
                f"| {r['route']} / {r['stage']} | {r['cohort']} | {r['calls']} | "
                f"{r['p50']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} |"
            )
    lines.extend(
        [
            "",
            "TOTAL_SHARED_E_R_F ejecuta productores compartidos y cabezas E_CTX, R_CTX y "
            "F_TRUE: no debe presentarse como coste aislado F_TRUE. La ruta E_CTX excluye "
            "registro RGB. Garl total incluye registro FCWD, lectura, crop, representación nativa "
            "y red full/fallback event; resident_network sólo repite la red sobre el mismo tensor. "
            "Se usa la preparación nativa vectorizada exacta ya admitida, "
            "no un Garl lento artificial.",
            "",
            "El estado temporal se reconstruye desde sensores por consulta; no es un servicio "
            "persistente con commit, resets y varios objetos. No se midieron H2D/CUDA graphs. "
            "Quedan pendientes el perímetro de servicio GPU aislado y su pasada "
            "cronológica completa "
            "antes de promover una implementación. Los tiempos por etapa y carga se conservan "
            "en COST_SUMMARY.json, sin inferir un ganador de velocidad entre perímetros distintos.",
        ]
    )
    (OUT / "COST_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "COST_SUMMARY.json", result)
    return result


class NoAnnotations(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        return ast.Assign(targets=[node.target], value=node.value or ast.Constant(None))


def source_parity() -> dict:
    original = OUT / "executed_sources/operational/multimodal_validation/garl.py"
    current = ROOT / "operational/multimodal_validation/garl.py"
    bodies = []
    for path in (original, current):
        tree = NoAnnotations().visit(ast.parse(path.read_text(encoding="utf-8")))
        bodies.append({n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)})
    comparisons = {name: body == bodies[1][name] for name, body in bodies[0].items()}
    assert all(comparisons.values())
    result = dict(
        function_AST_parity=comparisons,
        executed_source_sha256=sha(original),
        current_source_sha256=sha(current),
        excluded="type annotations, comments and imports; all function bodies identical",
    )
    write(OUT / "SOURCE_STYLE_PARITY.json", result)
    return result


def validate() -> dict:
    full, visual = sealed("full"), sealed("visual")
    registrations()
    frame = pd.read_csv(OUT / "FCWD_MULTIMODAL_PREDICTIONS.csv")
    assert len(frame) == 630 and frame.query_id.is_unique
    assert frame.truth_ttc_seconds.notna().sum() == 597
    assert full.available.sum() == visual.available.sum() == 495
    missing = ~frame.garl_full_available
    assert missing.sum() == 135 and frame.loc[missing, "truth_ttc_seconds"].notna().sum() == 125
    assert np.array_equal(
        frame.loc[missing, "GARL_FULL_REGISTERED_FALLBACK_EVENT"],
        frame.loc[missing, "public_Garl_event_lhr"],
    )
    assert np.array_equal(
        frame.loc[~frame.rgb_available, "F_TRUE"], frame.loc[~frame.rgb_available, "E_CTX_MATCHED"]
    )
    state = read(OUT / "STATE_BINDING.json")
    for path, expected in state["dependencies"].items():
        assert sha(path) == expected
    for endpoint in state["endpoints"].values():
        assert sha(endpoint["checkpoint_path"]) == endpoint["verified_sha256"]
    assert len(state["endpoints"]) == 12
    assert read(OUT / "TRACE_PARITY.json")["all_bitwise_equal"]
    assert read(OUT / "H_RAW_COMPLETE.json")["queries"] == 32
    assert len(list((OUT / "geometry/panels").glob("*.png"))) == 12
    annotations = pd.read_csv(OUT / "geometry/CORRESPONDENCES_TO_ANNOTATE.csv")
    assert len(annotations) == 72 and annotations.event_x.isna().all()
    historic = pd.read_csv(OUT / "HISTORICAL_TRACE_PARITY.csv")
    # Instrumentation parity is exact above. Historical device/route parity is
    # a separately reported measurement, not an invented prospective tolerance.
    assert np.isfinite(historic.absolute_delta).all()
    profile = pd.read_csv(OUT / "cpu_profile/HISTORICAL_TRACE_PARITY.csv")
    profile.nlargest(12, "absolute_delta").to_csv(
        OUT / "cpu_profile/LARGEST_HISTORICAL_DELTAS.csv", index=False
    )
    wide = profile.pivot(
        index=["cohort", "query_id", "model"], columns="repeat", values="cpu_prediction"
    )
    assert np.array_equal(wide[0], wide[1])
    garl_cost = pd.read_csv(OUT / "GARL_CPU_COST.csv")
    wide_garl = garl_cost.pivot(
        index=["cohort", "query_id"], columns="iteration", values="prediction_ttc"
    )
    assert np.array_equal(wide_garl[0], wide_garl[1])
    fcwd_cost = garl_cost[garl_cost.cohort == "FCWD"]
    q1_predictions = frame.set_index("query_id")["GARL_FULL_REGISTERED_FALLBACK_EVENT"]
    cost_delta = np.abs(
        fcwd_cost.prediction_ttc.to_numpy() - q1_predictions.loc[fcwd_cost.query_id].to_numpy()
    )
    assert cost_delta.max() < 0.001
    for filename, expected in [
        ("RUFF.log", "All checks passed!"),
        ("RUFF_FORMAT.log", "already formatted"),
        ("PYRIGHT.log", "0 errors, 0 warnings"),
        ("PYTEST.log", "[100%]"),
        ("PACKAGE_TESTS.log", "Ran 10 tests"),
    ]:
        assert expected in (OUT / filename).read_text(encoding="utf-8")
    incoming = Path.home() / "Downloads/E_JEPA_TTC_MULTIMODAL_NEXT_20261012.zip"
    assert sha(incoming) == "051eaba78cf5ffe58dca720d07c02701d1f1b30a190fe8da028e0acb3fa1f2c4"
    return dict(
        status="PASSED",
        endpoint_hashes_reverified=12,
        cohort_rows=630,
        labels=597,
        common_queries=495,
        common_labels=472,
        instrumented_calls=read(OUT / "TRACE_PARITY.json")["instrumented_calls"],
        historical_max_delta_s=float(historic.absolute_delta.max()),
        historical_route_parity="NOT_BITWISE; controlled_same_device_replay_pending",
        repeated_profile_serialized_outputs_equal=True,
        profile_historical_max_delta_s=float(profile.absolute_delta.max()),
        FCWD_cost_vs_sealed_prediction_max_delta_s=float(cost_delta.max()),
        checks="6 extension tests, 10 package tests, scoped Ruff/format/Pyright passed",
        incoming_zip=dict(sha256=sha(incoming), bytes=incoming.stat().st_size),
        geometry_panels=12,
        empty_independent_annotation_slots=72,
        replay=read(OUT / "public_replay/REPLAY_RESULT.json"),
        source_parity=source_parity(),
        optimizer_updates=0,
    )


def dependencies() -> dict:
    paths = [
        REG / "REGISTRATION_SEALED.json",
        REG / "BINDING.json",
        V12 / "artifacts/sota_campaign_20261008/fcwd_population/QUERY_MANIFEST.json",
        V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json",
        *[RUN / f"{p}_MANIFEST.json" for p in ("P", "H", "V")],
        *[RUN / f"features/H/{p}_NORMALIZER.json" for p in ("E", "R")],
    ]
    return dict(
        paths={str(p): sha(p) for p in paths},
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "h5py", "psutil", "pytest")
        },
        Garl_full=read(OUT / "garl_full/BINDING.json"),
        Garl_visual=read(OUT / "garl_visual/BINDING.json"),
        Garl_event=read(OUT / "GARL_CPU_COST_BINDING.json")["event_checkpoint_sha256"],
        origin="https://huggingface.co/NAIL-HNU/GarlTTC-model",
        upstream="https://github.com/NAIL-HNU/Garl-TTC",
        visual_download_revision="b676fcdaf26c04bcf896cdb2b208c9c424e8462a",
        license="No explicit license identified in downloaded model README; weights excluded",
        source_relocation=read(OUT / "SOURCE_RELOCATION.json"),
        endpoint_inventory="STATE_BINDING.json",
    )


def bundle() -> None:
    files = []
    for directory in [
        ROOT / "operational/multimodal_validation",
        ROOT / "docs/multimodal_next_handoff_20261012",
        OUT,
    ]:
        for p in directory.rglob("*"):
            if not p.is_file() or any(s in p.parts for s in ("__pycache__", ".cache")):
                continue
            if p.suffix.lower() in (".pt", ".pth", ".h5", ".hdf5", ".tar", ".pyc", ".tmp"):
                continue
            if p.name in ("BUNDLE_VERIFICATION.json", "BUNDLE_MANIFEST.json"):
                continue
            files.append(p)
    files.append(ROOT / "tests/test_multimodal_validation.py")
    members = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    write(OUT / "BUNDLE_MANIFEST.json", members)
    destination = ROOT / "artifacts/E_JEPA_TTC_MULTIMODAL_V16_RESULT_20261012.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(OUT / "BUNDLE_MANIFEST.json", "BUNDLE_MANIFEST.json")
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name, expected in members.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    receipt = dict(
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha(destination),
        files_verified=len(members),
        CRC="PASSED",
    )
    write(OUT / "BUNDLE_VERIFICATION.json", receipt)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    cost_review()
    write(OUT / "VALIDATION.json", validate())
    write(OUT / "DEPENDENCIES.json", dependencies())
    write(OUT / "FINAL_STATE.json", snapshot())
    write(
        OUT / "UPDATE_LEDGER.json",
        dict(
            extension="V16_multimodal_validation",
            scientific=0,
            technical=0,
            recovery=0,
            physical_optimizer_updates=0,
            V15="independent_authorization_not_in_this_ledger",
            training_proposals_executed=False,
        ),
    )
    write(
        OUT / "NEXT_DECISION.json",
        dict(
            status="FROZEN_COMPARISON_COMPLETE_WITH_EXPLICIT_GEOMETRY_AND_COST_DEPENDENCIES",
            Q0="COMPLETE",
            Q1="COMPLETE_ESTIMATED_REGISTRATION_NOT_OFFICIAL_EXTRINSICS",
            Q2="COMPLETE_1229_BITWISE_PARITY_CALLS",
            Q3="36_TRIALS_COMPLETE_72_INDEPENDENT_CORRESPONDENCES_PENDING",
            Q4="64_QUERY_CPU_PILOT_COMPLETE_ISOLATED_GPU_SERVICE_PERIMETER_PENDING",
            sota_every_regime=False,
            new_recipe_promoted=False,
            optimizer_updates=0,
            next_training="R_ANCHOR_EVENT_vs_R_ANCHOR_ZERO_requires_new_identity_and_authorization",
            V15="preserve_existing_campaign_and_authorization",
        ),
    )
    bundle()


if __name__ == "__main__":
    main()
