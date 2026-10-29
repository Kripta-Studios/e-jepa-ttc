"""Execute the authorized frozen-system study without optimizer or raw-data access."""

# JSON receipts and Pandas/Torch payloads are dynamic at the checked I/O boundary.
# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import runpy
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from operational.simplex_t_post_campaign.contracts import (  # noqa: E402
    MODELS,
    checked,
    digest,
    fragment,
    order,
    read,
    route_admission,
    save,
    timed,
)

NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
H16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
DOCS = ROOT / "docs/simplex_t_nocturnal_20261003"
OUT = ROOT / "artifacts/simplex_t/post_campaign_20261003"
SOURCE_DOCS = ROOT / "docs/simplex_t_post_campaign_20261003"
EXTRACT = NIGHT / "delivery/extracted_bbabc02921d2"
STUDY = "FROZEN_SYSTEM_POST_20261003"
CONTRASTS = (
    ("H16@7", "H8@7"),
    ("H16@13", "H8@13"),
    ("H16@23", "H8@23"),
    ("FULL_C0", "H8@7"),
    ("A5_ONLY_C0", "FULL_C0"),
    ("C2F_ONLY_C0", "FULL_C0"),
    ("A5_PAIR_C0", "FULL_C0"),
    ("SET_AGE_C0", "FULL_C0"),
    ("SET_NOTIME_C0", "SET_AGE_C0"),
)
IDENTITY = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]


def now() -> str:
    """UTC timestamp for a durable boundary."""
    return datetime.now(UTC).isoformat()


@lru_cache(maxsize=1)
def guard_function() -> Any:
    """Verify the delivered guard once per process, before heavy imports."""
    path = EXTRACT / "regenerate.py"
    manifest = read(EXTRACT / "CONTENT_MANIFEST.json")
    checked(path, manifest["members"]["regenerate.py"]["sha256"])
    return runpy.run_path(str(path), run_name="post_campaign_resource_guard")["resource_guard"]


def guard() -> dict:
    """Check current resources without repeatedly hashing admitted sources."""
    return guard_function()(OUT)


class Owner:
    """Exclusive writer identity including process birth time, protecting PID reuse."""

    def __enter__(self) -> Owner:
        import psutil

        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "ACTIVE_OWNER.json"
        if self.path.exists():
            old = read(self.path)
            try:
                alive = abs(psutil.Process(old["pid"]).create_time() - old["created_unix"]) < 0.01
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                raise RuntimeError("post-campaign writer already alive")
            save(OUT / "owners" / f"stale_{old['pid']}_{old['created_unix']}.json", old)
            self.path.unlink()
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(
                dict(pid=os.getpid(), created_unix=psutil.Process().create_time(), study=STUDY),
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *args: Any) -> None:
        self.path.unlink(missing_ok=True)


def source_files() -> list[Path]:
    return [
        *sorted((ROOT / "operational/simplex_t_post_campaign").glob("*.py")),
        ROOT / "src/e_jepa_ttc/evaluation/stage63_65.py",
        ROOT / "src/e_jepa_ttc/simplex_t/model.py",
        ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        ROOT / "operational/simplex_t_cost_context/model.py",
    ]


def admitted() -> dict:
    """Verify the prospective study protocol and frozen operational/scientific source."""
    p = read(OUT / "PROTOCOL.json")
    for name, sha in p["source_pins"].items():
        checked(ROOT / name, sha)
    checked(DOCS / "PLAN_CONTINUACION_POST_NOCTURNA.json", p["plan_sha256"])
    return p


def progress(stage: str, status: str, **values: Any) -> None:
    """Persist verified work, never confuse historical progress with active state."""
    path = OUT / "PROGRESS.json"
    p: dict[str, Any] = (
        read(path) if path.exists() else dict(study=STUDY, optimizer_updates=0, stages={})
    )
    p["stages"][stage] = dict(status=status, observed_utc=now(), **values)
    p["next_task"] = next(
        (
            s
            for s in ("P0", "P1", "P2", "P3", "P4")
            if p["stages"].get(s, {}).get("status") != "COMPLETE"
        ),
        None,
    )
    save(path, p)
    print(json.dumps(dict(stage=stage, status=status, **values), ensure_ascii=False), flush=True)


def prepare() -> None:
    """Freeze the independent no-training protocol before any new timing."""
    plan_path = DOCS / "PLAN_CONTINUACION_POST_NOCTURNA.json"
    plan = read(plan_path)
    for name, expected in plan["sources"].items():
        checked(DOCS / name, expected)
    index = read(NIGHT / "PROFILE_INPUT_EXPORT.json")
    if set(index["models"]) != set(MODELS):
        raise ValueError("nine complete target-free frozen heads required")
    for row in index["models"].values():
        if row["target_fields_included"] or row["role"] != "TRAIN_FOLD0_FIXED_HASH_SELECTION":
            raise ValueError("TRAIN target-free profile contract required")
        for kind in ("weights", "inputs"):
            checked(NIGHT / row[kind + "_path"], row[kind + "_sha256"])
    binding = dict(
        schema="frozen_system_post_campaign_v1",
        study=STUDY,
        authorization=(
            "User explicitly requested implementation and tests of the post-campaign plan; "
            "zero fits/optimizer updates."
        ),
        optimizer_updates_authorized=0,
        sealed_access=False,
        new_models_or_data=False,
        candidate="TPR-D1-H8-C160",
        plan_sha256=digest(plan_path),
        source_pins={
            str(x.relative_to(ROOT)).replace("\\", "/"): digest(x) for x in source_files()
        },
        evidence_pins=plan["sources"],
        profile_index_sha256=digest(NIGHT / "PROFILE_INPUT_EXPORT.json"),
        selection_sha256=digest(NIGHT / "profiling/SELECTION.json"),
        profile=dict(
            blocks=3,
            models=list(MODELS),
            measurements_per_model_per_block=500,
            fragment_size=25,
            warmups_per_model_per_block=25,
            batch=1,
            device="CPU",
            precision="FP32",
            threads=4,
            interop=2,
            order=[[list(order(b, f)) for f in range(20)] for b in range(3)],
            timer=(
                "perf_counter_ns; head forward plus canonical float64 TTC emission; "
                "excludes slicing, guards, finite checks and serialization"
            ),
            scope="prepared_input_head_only",
            percentile="linear",
            host_causal_attribution=False,
            raw_or_expert_calls=False,
        ),
        diagnostics=dict(
            scope="post_hoc_descriptive; all 8192 queries and nine sequences",
            contrasts=list(CONTRASTS),
            episode_gap_us=100000,
            episode_rule=(
                "All queries sorted by sequence/track/anchor; new episode only when gap "
                "exceeds fixed 100 ms. No error-based selection."
            ),
            intervals=(
                "reuse original hierarchical and sequence-only receipts; "
                "no new bootstrap or seed-scene pseudoreplication"
            ),
            loss_means_not_ttc_ensembles=True,
        ),
        comparator_review_ids=["OFFICIAL_V7_C2F", "GARL_LOCAL_FROZEN"],
        resources=dict(
            ram_min_bytes=2 * 1024**3,
            rss_max_bytes=4 * 1024**3,
            disk_free_after_reserve_bytes=10000000000,
            artifact_cap_bytes=10000000000,
        ),
        full_route=(
            "Requires explicit exclusive GPU/heavy-I/O slot and separately verified TRAIN sources; "
            "no inferred permission from idle hardware."
        ),
    )
    path = OUT / "PROTOCOL.json"
    if path.exists() and read(path) != binding:
        raise ValueError("prospective protocol already exists with different source/design")
    save(path, binding)
    guard()
    # Bounded shared interface, no private result or ledger access.
    h = read(H16 / "PROTOCOL.json")
    paths = read(Path(h["launch"]["local_paths"]))
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack = read(ack_path)
    import psutil

    live = []
    for row in ack["resources"].get("active_processes_and_parents", []):
        cmd = " ".join(row.get("cmdline", []))
        if "build_stage71" not in cmd:
            continue
        try:
            proc = psutil.Process(row["pid"])
            alive = abs(proc.create_time() - row["created_unix"]) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        live.append(
            dict(
                pid=row["pid"],
                historical_created_unix=row["created_unix"],
                same_process_alive=alive,
            )
        )
    owners = []
    for name in ack["resources"].get("active_owner_files", []):
        path = Path(name)
        owners.append(dict(path=name, exists=path.exists()))
        if path.exists():
            raise InterruptedError(
                "shared owner is live or unverified; defer CPU admission until validated"
            )
    if any(row["same_process_alive"] for row in live):
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise InterruptedError("conditional shared CPU overlap requires 8 GiB available RAM")
    a = route_admission(ack)
    save(
        OUT / "FULL_ROUTE_ADMISSION.json",
        dict(
            **a,
            observed_utc=now(),
            ack_path=str(ack_path),
            ack_sha256=digest(ack_path),
            current_owner_metadata=owners,
            historical_worker_pid_checks=live,
            eap_root_exists=Path(paths["eap_root"]).is_dir(),
            no_stage_changes=True,
            cpu_admission=(
                "no verified historical expansion worker or owner alive; "
                "own CPU/resource contract applies"
            ),
        ),
    )
    progress("P3", "BLOCKED", dependency=a["missing_dependency"])
    progress("P0", "ADMITTED", protocol_sha256=digest(OUT / "PROTOCOL.json"))


def load_frames() -> tuple[dict[str, Any], list[dict]]:
    """Load only explicit physically hash-checked published development predictions."""
    import pandas as pd

    h = read(H16 / "PROTOCOL.json")
    parts: dict[str, list] = {}
    inventory = []

    def add(label: str, path: Path, expected: str) -> None:
        guard()
        checked(path, expected)
        frame = pd.read_parquet(path)
        parts.setdefault(label, []).append(frame)
        inventory.append(dict(model=label, path=str(path), sha256=expected, rows=len(frame)))

    for row in h["historical_controls"].values():
        fit = row["endpoint"]["fit"]
        label = f"H{16 if 'H16' in fit['name'] else 8}@{fit['seed']}"
        add(label, Path(row["prediction"]), row["prediction_sha256"])
    for row in read(H16 / "ENDPOINTS.json")["fits"]:
        path = H16 / "publication" / row["key"] / "PREDICTIONS.parquet"
        receipt = read(path.parent / "PUBLICATION.json")
        add(f"H16@{row['seed']}", path, receipt["predictions_sha256"])
    for row in read(NIGHT / "execution/ANALYSIS_EXPORT_INDEX.json")["fits"].values():
        add(row["arm"], NIGHT / "execution" / row["predictions_path"], row["predictions_sha256"])
    frames = {
        k: pd.concat(v, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        for k, v in parts.items()
    }
    base = frames["H8@7"]
    for label, f in frames.items():
        if len(f) != 8192 or f.sample_token.duplicated().any() or f.sequence_id.nunique() != 9:
            raise ValueError(f"incomplete canonical cohort: {label}")
        if not f[IDENTITY].equals(base[IDENTITY]):
            raise ValueError(f"unpaired scientific cohort: {label}")
    return frames, inventory


def table(name: str, rows: list[dict]) -> None:
    """Save a full precision table and its hash after a complete analysis fragment."""
    import pandas as pd

    path = OUT / name
    temp = path.with_suffix(path.suffix + ".pending")
    pd.DataFrame(rows).to_csv(temp, index=False, float_format="%.17g")
    with temp.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def diagnostics() -> None:
    """Generate paired full-cohort strata and every temporal episode without retuning."""
    admitted()
    guard()
    import numpy as np

    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass

    frames, inventory = load_frames()
    base = frames["H8@7"]
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    target = base.target_ttc.to_numpy()
    buckets = np.select(
        [target > 6, target > 3, target > 0],
        ["positive_6_10", "positive_3_6", "positive_0_3"],
        default="negative_10_0",
    )
    seqs = sorted(base.sequence_id.unique())
    groups: list[tuple[str, str, Any]] = [("global", "all", np.ones(len(base), bool))]
    groups += [("sequence", s, (base.sequence_id == s).to_numpy()) for s in seqs]
    groups += [("fold", str(f), (base.outer_fold == f).to_numpy()) for f in range(3)]
    for b in ("positive_0_3", "positive_3_6", "positive_6_10", "negative_10_0"):
        groups.append(("bucket", b, buckets == b))
        groups.extend(
            ("sequence_bucket", s + "/" + b, (base.sequence_id == s).to_numpy() & (buckets == b))
            for s in seqs
        )
    metrics, paired = [], []
    scores = {}
    for label, f in frames.items():
        loss = f.loss.to_numpy(np.float64)
        regenerated = 10000 * np.abs(
            f.prediction_phase.to_numpy(np.float64) - f.target_phase.to_numpy(np.float64)
        )
        if not np.isfinite(loss).all() or not np.allclose(loss, regenerated, rtol=0, atol=1e-10):
            raise ValueError(f"canonical PHASE17/loss mismatch: {label}")
        scores[label] = float(mass @ loss)
        for kind, key, mask in groups:
            wt = mass[mask]
            total = float(wt.sum())
            row: dict[str, Any] = dict(
                model=label,
                group=kind,
                key=key,
                queries=int(mask.sum()),
                global_mass=total,
                MiD=None,
            )
            if total:
                wt = wt / total
                row["MiD"] = float(wt @ loss[mask])
                for field in (
                    "wrong_sign",
                    "phase_support_saturation",
                    "finite_ttc_cap",
                    "phase_interval_covers",
                    "phase_interval_width",
                    "cold_start",
                    "escape_gain",
                    "escape_harm",
                ):
                    row[field] = float(wt @ f[field].to_numpy(np.float64)[mask])
                row["escape_fraction_full_three_expert_hull"] = float(
                    wt @ (f.hull_position != "inside").to_numpy(np.float64)[mask]
                )
                row["finite_coverage"] = float(
                    wt @ np.isfinite(f.prediction_ttc_s.to_numpy())[mask]
                )
            row["hull_scope"] = "full_three_frozen_experts_diagnostic_only"
            row["interval_scope"] = "descriptive_coverage_not_calibration_claim"
            metrics.append(row)
        guard()
    losses = {k: f.loss.to_numpy(np.float64) for k, f in frames.items()}
    contrasts = [(c, r, losses[c], losses[r]) for c, r in CONTRASTS]
    for seeds, label in (((13, 23), "H16_NEW13_23_MEAN_LOSS"), ((7, 13, 23), "H16_ALL3_MEAN_LOSS")):
        contrasts.append(
            (
                label,
                "H8_PAIRED_MEAN_LOSS",
                np.mean([losses[f"H16@{s}"] for s in seeds], axis=0),
                np.mean([losses[f"H8@{s}"] for s in seeds], axis=0),
            )
        )
    for candidate, reference, lc, lr in contrasts:
        for kind, key, mask in groups:
            total = float(mass[mask].sum())
            paired.append(
                dict(
                    candidate=candidate,
                    reference=reference,
                    group=kind,
                    key=key,
                    queries=int(mask.sum()),
                    global_mass=total,
                    delta_MiD=float(mass[mask] @ (lc - lr)[mask] / total) if total else None,
                    weighted_global_contribution=float(mass[mask] @ (lc - lr)[mask]),
                    post_hoc=True,
                    predictions_ensembled=False,
                )
            )
    table("MODEL_DIAGNOSTICS.csv", metrics)
    table("PAIRED_STRATA.csv", paired)
    # Deterministic all-query episode partition, independent of error outcomes.
    episodes = []
    work = base.assign(_position=np.arange(len(base))).sort_values(
        ["sequence_id", "track_id", "anchor_us", "sample_token"]
    )
    for (seq, track), group in work.groupby(["sequence_id", "track_id"], sort=True):
        number = (group.anchor_us.diff().fillna(0) > 100000).cumsum()
        for ep, piece in group.groupby(number, sort=True):
            positions = piece._position.to_numpy(int)
            wt = mass[positions]
            total = float(wt.sum())
            for candidate, reference, lc, lr in contrasts:
                episodes.append(
                    dict(
                        sequence_id=seq,
                        track_id=track,
                        episode=int(ep),
                        start_us=int(piece.anchor_us.min()),
                        end_us=int(piece.anchor_us.max()),
                        queries=len(piece),
                        global_mass=total,
                        candidate=candidate,
                        reference=reference,
                        delta_MiD=float(wt @ (lc - lr)[positions] / total),
                        global_contribution=float(wt @ (lc - lr)[positions]),
                        post_hoc=True,
                    )
                )
    table("ALL_ERROR_EPISODES.csv", episodes)
    save(
        OUT / "PREDICTION_INVENTORY.json",
        dict(
            models=12,
            query_rows=8192 * 12,
            inputs=inventory,
            scores=scores,
            all_sequences=seqs,
            mass_sum=float(mass.sum()),
            bootstrap_repeated=False,
            optimizer_updates=0,
            candidate_replaced=False,
        ),
    )
    endpoint_rows = []
    for path, parent in (
        (H16 / "ENDPOINTS.json", H16),
        (NIGHT / "execution/N2_ENDPOINTS.json", NIGHT / "execution"),
        (NIGHT / "execution/N3_ENDPOINTS.json", NIGHT / "execution"),
    ):
        for row in read(path)["fits"]:
            checkpoint = parent / row["checkpoint"]
            checked(checkpoint, row["checkpoint_sha256"])
            endpoint_rows.append(
                dict(
                    key=row["key"],
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=row["checkpoint_sha256"],
                    saved_updates=2500,
                )
            )
    if len(endpoint_rows) != 24:
        raise ValueError("all 24 final endpoints required")
    save(
        OUT / "FINAL_ENDPOINT_INDEX.json",
        dict(
            endpoints=endpoint_rows,
            saved_updates=60000,
            historical_snapshots_are_not_active_state=True,
            new_updates=0,
        ),
    )
    review_comparators(base, mass)
    # Reconcile to existing physically published results without regenerating T6.
    h = read(DOCS / "H16_RESULTS.json")
    errors = []
    for seed in (7, 13, 23):
        for label, field in ((f"H8@{seed}", "h8_mid"), (f"H16@{seed}", "h16_mid")):
            errors.append(abs(scores[label] - h["comparisons"][str(seed)][field]))
    for name in ("N2_RESULTS.json", "N3_RESULTS.json"):
        for label, value in read(DOCS / name)["scores"].items():
            errors.append(abs(scores[label] - value))
    if max(errors) > 1e-9:
        raise ValueError("full precision scores disagree with published campaign")
    save(
        OUT / "DIAGNOSTIC_RECEIPT.json",
        dict(
            status="COMPLETE",
            canonical_queries=8192,
            models=12,
            endpoint_count=24,
            saved_updates=60000,
            new_updates=0,
            maximum_score_difference=max(errors),
            strata_rows=len(paired),
            episode_rows=len(episodes),
            post_hoc=True,
            all_queries_preserved=True,
            no_new_intervals=True,
        ),
    )
    progress("P0", "COMPLETE", endpoints=24, saved_updates=60000, comparators_reviewed=2)
    progress(
        "P1", "COMPLETE", prediction_rows=8192 * 12, strata=len(paired), episode_rows=len(episodes)
    )


def review_comparators(base: Any, mass: Any) -> None:
    """Admit at most two explicit historical comparators, preserving missing contracts."""
    import numpy as np
    import pandas as pd

    rows = list(csv.DictReader((DOCS / "BASELINE_REGISTRY.csv").open(encoding="utf-8")))
    result = []
    for label in admitted()["comparator_review_ids"]:
        row = next(r for r in rows if r["id"] == label)
        paths, hashes = json.loads(row["evidence_paths"]), json.loads(row["evidence_sha256"])
        for name, sha in zip(paths, hashes, strict=True):
            checked(Path(name), sha)
        manifest = read(Path(paths[0]))
        reader: Any = pd.read_csv  # Installed Pandas supports round_trip; its stub omits it.
        f = pd.concat(
            [reader(path, float_precision="round_trip") for path in paths[1:]],
            ignore_index=True,
        )
        findings = dict(
            id=label,
            evidence_paths=paths,
            evidence_sha256=hashes,
            queries=len(f),
            published_MiD=float(row["MiD"]),
            classification="INSUFFICIENT_CONTRACT_EVIDENCE",
            scientific_commit=row["scientific_commit"],
            source_manifest_type=manifest["artifact_type"],
            context="not established",
            ROI="not established",
            training_groups="not fully bound",
            missing=[
                "frozen producer source commit",
                "complete TRAIN genealogy and checkpoint selection",
                "ROI/modality/availability contract",
            ],
            no_automatic_comparator_admission=True,
        )
        findings["columns"] = list(f.columns)
        if "sample_token" in f:
            f = f.sort_values("sample_token").reset_index(drop=True)
            findings["identical_query_tokens"] = bool(
                np.array_equal(f.sample_token, base.sample_token)
            )
            for field in ("sequence_id", "track_id", "outer_fold", "target_ttc"):
                findings["identical_" + field] = bool(
                    field in f and np.array_equal(f[field], base[field])
                )
            if "target_ttc" in f:
                findings["max_target_difference"] = (
                    float(np.max(np.abs(f.target_ttc.to_numpy() - base.target_ttc.to_numpy())))
                    if len(f) == len(base)
                    else None
                )
        result.append(findings)
    save(
        OUT / "COMPARATOR_REVIEW.json",
        dict(
            status="BOUNDED_REVIEW_COMPLETE",
            comparisons=result,
            model_selection=False,
            additional_archeology=False,
        ),
    )


def profile() -> None:
    """Measure three matched frozen-head blocks with recovery every 25 calls/model."""
    p = admitted()
    checked(NIGHT / "PROFILE_INPUT_EXPORT.json", p["profile_index_sha256"])
    checked(NIGHT / "profiling/SELECTION.json", p["selection_sha256"])
    guard()
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.simplex_t_cost_context.model import build_model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    index = read(NIGHT / "PROFILE_INPUT_EXPORT.json")["models"]
    models, inputs, audit = {}, {}, {}
    for label in MODELS:
        row = index[label]
        w = checked(NIGHT / row["weights_path"], row["weights_sha256"])
        x = checked(NIGHT / row["inputs_path"], row["inputs_sha256"])
        constructor = row["constructor"]
        model = (
            TemporalRefiner(TemporalConfig(**constructor["config"]))
            if constructor["kind"] == "historical_temporal"
            else build_model(constructor["arm"])
        )
        model.cpu().float().eval()
        with np.load(w, allow_pickle=False) as z:
            model.load_state_dict(
                {key: torch.from_numpy(z[key].copy()) for key in z.files}, strict=True
            )
        with np.load(x, allow_pickle=False) as z:
            if set(z.files) != {"features", "times", "valid", "experts"}:
                raise ValueError("profile inputs must be target-free")
            xs = tuple(
                torch.from_numpy(z[key].copy()) for key in ("features", "times", "valid", "experts")
            )
        if any(len(value) != 64 for value in xs):
            raise ValueError("fixed 64 TRAIN inputs required")
        models[label], inputs[label] = model, xs
        audit[label] = dict(
            weights_sha256=row["weights_sha256"],
            inputs_sha256=row["inputs_sha256"],
            endpoint_sha256=row["endpoint_sha256"],
            shapes=[list(v.shape) for v in xs],
            dtypes=[str(v.dtype) for v in xs],
            strides=[list(v.stride()) for v in xs],
            parameters=sum(v.numel() for v in model.parameters()),
            device="cpu",
            floating_parameters="float32",
        )
    runtime = dict(
        host=platform.node(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        threads=torch.get_num_threads(),
        interop=torch.get_num_interop_threads(),
        mkldnn_available=torch.backends.mkldnn.is_available(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        torch_config=torch.__config__.show(),
        clock=dict(implementation=time_info().implementation, resolution=time_info().resolution),
        timer_scope=p["profile"]["timer"],
        shapes=audit,
        observed_utc=now(),
    )
    fingerprint = {key: value for key, value in runtime.items() if key != "observed_utc"}
    if (OUT / "RUNTIME_AUDIT.json").exists():
        previous = read(OUT / "RUNTIME_AUDIT.json")
        if {k: v for k, v in previous.items() if k != "observed_utc"} != fingerprint:
            raise ValueError("runtime changed within matched study; preserve prior measurements")
    else:
        save(OUT / "RUNTIME_AUDIT.json", runtime)
    pin = digest(OUT / "PROTOCOL.json")
    reused, committed = 0, 0
    selection = read(NIGHT / "profiling/SELECTION.json")["queries"]
    with torch.inference_mode():
        for block in range(3):
            paths = [
                OUT / "timings" / f"b{block}_f{frag:02d}_{label}.json"
                for frag in range(20)
                for label in MODELS
            ]
            if any(not path.exists() for path in paths):
                for label in order(block, 0):
                    for i in range(25):
                        guard()
                        batch = tuple(v[i : i + 1] for v in inputs[label])
                        emission = phase_to_ttc(
                            models[label](*batch)["point_phase"].to(torch.float64)
                        )
                        if not bool(torch.isfinite(emission).all()):
                            raise ValueError("nonfinite warmup")
            for frag in range(20):
                for label in order(block, frag):
                    binding = dict(
                        protocol_sha256=pin,
                        block=block,
                        fragment=frag,
                        model=label,
                        runtime_sha256=digest(OUT / "RUNTIME_AUDIT.json"),
                        weights_sha256=index[label]["weights_sha256"],
                        inputs_sha256=index[label]["inputs_sha256"],
                        selection_sha256=p["selection_sha256"],
                        measurement_start=frag * 25,
                        queries=[
                            selection[i % 64]["sample_token"]
                            for i in range(frag * 25, (frag + 1) * 25)
                        ],
                    )
                    path = OUT / "timings" / f"b{block}_f{frag:02d}_{label}.json"

                    def produce(
                        label: str = label, frag: int = frag, binding: dict = binding
                    ) -> list[float]:
                        values = []
                        resources = guard()
                        save(
                            OUT / "INFLIGHT.json",
                            dict(
                                binding=binding,
                                started_utc=now(),
                                resources=resources,
                                maximum_uncommitted_timing_calls=25,
                                optimizer_updates=0,
                            ),
                        )
                        for i in range(frag * 25, (frag + 1) * 25):
                            guard()
                            batch = tuple(v[i % 64 : i % 64 + 1] for v in inputs[label])

                            def call(batch: tuple = batch) -> tuple[Any, Any]:
                                output = models[label](*batch)
                                return output, phase_to_ttc(output["point_phase"].to(torch.float64))

                            (output, emission), elapsed = timed(call)
                            if not all(
                                bool(torch.isfinite(v).all()) for v in output.values()
                            ) or not bool(torch.isfinite(emission).all()):
                                raise ValueError("nonfinite measured head output")
                            values.append(elapsed)
                        return values

                    _, was_reused = fragment(path, binding, produce)
                    reused += int(was_reused)
                    committed += 1
                    save(
                        OUT / "PROFILE_PROGRESS.json",
                        dict(
                            protocol_sha256=pin,
                            fragments=committed,
                            measured_calls=25 * committed,
                            total_fragments=540,
                            last=path.name,
                            reused_fragments=reused,
                            optimizer_updates=0,
                            observed_utc=now(),
                        ),
                    )
                progress(
                    "P2",
                    "RUNNING",
                    fragments=committed,
                    measured_calls=25 * committed,
                    block=block + 1,
                )
    rows = []
    for block in range(3):
        for label in MODELS:
            raw = [
                v
                for frag in range(20)
                for v in read(OUT / "timings" / f"b{block}_f{frag:02d}_{label}.json")["raw_ms"]
            ]
            rows.append(
                dict(
                    block=block + 1,
                    model=label,
                    measurements=len(raw),
                    p50_ms=float(np.median(raw)),
                    p95_ms=float(np.quantile(raw, 0.95, method="linear")),
                    scope="prepared_input_head_only",
                    batch=1,
                    canonical_emission=True,
                )
            )
    table("MATCHED_HEAD_COST.csv", rows)
    save(
        OUT / "PROFILE_RECEIPT.json",
        dict(
            status="COMPLETE",
            fragments=540,
            measured_calls=13500,
            warmup_design_calls=675,
            reused_fragments=reused,
            optimizer_updates=0,
            source_protocol_sha256=pin,
            timing_is_host_dependent=True,
            sensor_to_decision=False,
            models_are_fold0_seed7_only=True,
        ),
    )
    progress("P2", "COMPLETE", fragments=540, measured_calls=13500, optimizer_updates=0)


def time_info() -> Any:
    import time

    return time.get_clock_info("perf_counter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "diagnose", "profile"))
    args = parser.parse_args()
    with Owner():
        try:
            {"prepare": prepare, "diagnose": diagnostics, "profile": profile}[args.task]()
        except (InterruptedError, PermissionError) as exc:
            save(
                OUT / "PAUSE.json",
                dict(
                    task=args.task,
                    reason=str(exc),
                    observed_utc=now(),
                    checkpoints_modified=False,
                    optimizer_updates=0,
                ),
            )
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
