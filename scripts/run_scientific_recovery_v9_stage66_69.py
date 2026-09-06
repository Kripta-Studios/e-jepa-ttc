"""Single foreground, resumable runner for the fixed Stage66–69 campaign."""
# ruff: noqa: E402 -- repository script bootstraps src before project imports

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.risk_geometry_decision_v10 import replication_decision, stage_decision
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    CampaignOwner,
    PhaseLedger,
    atomic_json,
    binding,
    digest,
    object_digest,
    verify,
)
from e_jepa_ttc.data.event_geometry_cache_v10 import build_geometry
from e_jepa_ttc.data.frozen_expert_tables_v10 import (
    FrozenTable,
    arm_inputs,
    create_record,
    feature_mask,
    load_table,
    pair_permutation,
)
from e_jepa_ttc.evaluation.risk_geometry_v10 import diagnostics, loss_frame, verify_coverage
from e_jepa_ttc.models.full_regret_router import FullRegretRidge
from e_jepa_ttc.training.risk_router_v10 import fit, predict, scaler


def read(path: Path) -> Any:  # noqa: ANN401 -- JSON boundary
    return json.loads(path.read_text(encoding="utf-8-sig"))


def legacy_fit(path: Path) -> FullRegretRidge:
    with np.load(path, allow_pickle=False) as z:
        return FullRegretRidge(
            z["mean"], z["scale"], z["coefficient"], z["intercept"], float(z["ridge"])
        )


def preflight(root: Path, local: dict[str, Any]) -> None:
    """Recheck exact local sources; bind controls before any scientific fitting."""
    frozen = root / "frozen_audit/extracted_input/run"
    provenance = read(frozen / "stage65/STAGE65_SOURCE_PROVENANCE.json")
    sources = list(provenance["input_bindings"].values())
    access = AllowlistedSources(sources, root / "SOURCE_ACCESS.jsonl")
    for record in sources:
        access.access(Path(record["path"]))
    atomic_json(root / "ALLOWLIST.json", sources)
    if read(root / "NESTED_ANCESTRY_AUDIT.json")["status"] != "passed":
        raise ValueError("nested producer prerequisite missing")
    equivalents = read(root / "LEGACY_EQUIVALENCE_ATTEMPT.json")
    if len(equivalents) != 12 or any(
        not r["ttc_matches"] or r["selected_matches"] is False for r in equivalents
    ):
        raise ValueError("legacy replay failed")
    router_root = Path(local["reference_worktree"]) / "artifacts/scientific_recovery_v8/results"
    original = pd.read_csv(
        router_root / "router/aggregate_seed7/router_oof_predictions.csv"
    ).set_index("token_id")
    aggregate = read(router_root / "router/aggregate_seed7/router_seed7_aggregate.json")
    for fold in range(3):
        signature = router_root / f"runs/router_fold{fold}_seed7/router_signature.json"
        if digest(signature) != aggregate["checkpoint_sha256"][str(fold)]:
            raise ValueError("RouterR signature changed")
        replay = pd.read_csv(root / f"tables/outer{fold}_RouterR_replay.csv")
        if not np.array_equal(
            original.loc[replay.sample_token].choose_c2f.astype(int), replay.selected_expert
        ):
            raise ValueError("RouterR index mismatch")
    atomic_json(
        root / "LEGACY_EQUIVALENCE.json",
        dict(
            passed=True,
            per_row=equivalents,
            router_indices_exact=True,
            input_zip=digest(Path(local["source_bundle"])),
        ),
    )
    records = {}
    maps = {}
    for fold in range(3):
        for role in ("inner_oof", "outer_dev"):
            key = f"outer{fold}_{role}"
            record = create_record(
                root / f"tables/{key}.npz", root / f"tables/{key}.csv", fold, role
            )
            table = load_table(record, role=role)
            records[key] = record
            donor = pair_permutation(table, 70 + fold)
            path = root / f"tables/{key}_donors.npy"
            np.save(path, donor, allow_pickle=False)
            maps[key] = binding(path)
    for fold in range(3):
        if set(records[f"outer{fold}_inner_oof"]["sequences"]) & set(
            records[f"outer{fold}_outer_dev"]["sequences"]
        ):
            raise ValueError("outer sequence leakage")
    atomic_json(root / "TABLE_INDEX.json", records)
    atomic_json(root / "CONTROL_MAPS.json", maps)
    atomic_json(
        root / "PREFLIGHT.json",
        dict(
            passed=True,
            legacy=binding(root / "LEGACY_EQUIVALENCE.json"),
            nested=binding(root / "NESTED_ANCESTRY_AUDIT.json"),
            tables=binding(root / "TABLE_INDEX.json"),
            source_allowlist=binding(root / "ALLOWLIST.json"),
            protected_data_opened=False,
            prior_exposure="LEVEL_0_METADATA_ONLY",
        ),
    )


def freeze(root: Path, protocol: Path) -> dict[str, Any]:
    if (root / "IMPLEMENTATION_LOCK.json").exists():
        lock = read(root / "IMPLEMENTATION_LOCK.json")
        for record in lock["files"]:
            verify(record)
        for key in ("protocol", "tables", "controls", "qa"):
            verify(lock[key])
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            != lock["training_commit"]
        ):
            raise ValueError("training commit drifted")
        return lock
    qa = read(root / "QA_ACCEPTANCE.json")
    if not qa.get("passed") or qa.get("new_failure_ids"):
        raise ValueError("QA incomplete")
    for record in qa["files"]:
        verify(record)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("scientific lock requires clean worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "scripts",
            "tests",
            "configs/protocol/scientific_recovery_v9_stage66_69.json",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    lock = dict(
        training_commit=head,
        protocol=binding(protocol),
        files=[binding(ROOT / p) for p in files],
        tables=binding(root / "TABLE_INDEX.json"),
        controls=binding(root / "CONTROL_MAPS.json"),
        qa=binding(root / "QA_ACCEPTANCE.json"),
    )
    atomic_json(root / "IMPLEMENTATION_LOCK.json", lock)
    atomic_json(
        root / "TRAINING_AUTHORIZATION.json",
        dict(
            training_authorized=True,
            prompt=binding(root / "CONTINUATION_AUTHORIZATION.json"),
            protocol=binding(protocol),
            training_commit=head,
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True
            ).strip(),
            allowed_stages="67 and conditional 68/69 per fixed gates",
            owner=read(root / "ACTIVE_OWNER.json"),
            inputs=binding(root / "TABLE_INDEX.json"),
            excluded_splits_opened=False,
        ),
    )
    (root / "TRAINING_COMMIT.txt").write_text(head + "\n")
    return lock


def references(
    root: Path, indexes: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, list[float]]]]:
    frames = {}
    quantiles = {}
    for arm in ("S65-RISK17", "S65-RISK8"):
        parts = []
        train_regret = {}
        for fold in range(3):
            ridge = legacy_fit(
                root / f"frozen_audit/extracted_input/run/stage65/outer{fold}/{arm}.npz"
            )
            for role in ("inner_oof", "outer_dev"):
                t = load_table(indexes[f"outer{fold}_{role}"], role=role)
                x = t.inputs.features[:, :8] if arm.endswith("8") else t.inputs.features
                costs = ridge.predict_regret(x)
                selected = costs.argmin(1)
                frame = loss_frame(
                    t.metadata, t.inputs.expert_ttc, costs, selected, arm=arm, seed=7, fold=fold
                )
                if role == "inner_oof":
                    train_regret[str(fold)] = np.quantile(frame.regret, [0.5, 0.9, 0.95]).tolist()
                else:
                    parts.append(frame)
        frames[arm] = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        quantiles[arm] = train_regret
    return frames, quantiles


def execute_phase(
    root: Path,
    stage: str,
    seeds: list[int],
    protocol: dict[str, Any],
    lock: dict[str, Any],
    owner: CampaignOwner,
    geometry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    phase = root / f"stage{stage}_seeds{'_'.join(map(str, seeds))}"
    phase.mkdir(exist_ok=True)
    ledger = PhaseLedger(phase / "LEDGER.json")
    prerequisites = [
        ("INPUTS_VERIFIED", [root / "PREFLIGHT.json"]),
        ("BASELINES_REPLAYED", [root / "LEGACY_EQUIVALENCE.json"]),
        ("IMPLEMENTATION_TESTED", [root / "QA_ACCEPTANCE.json"]),
        ("SCIENTIFIC_LOCKED", [root / "IMPLEMENTATION_LOCK.json", root / "CONTROL_MAPS.json"]),
        ("FITTING", [root / "TRAINING_AUTHORIZATION.json"]),
    ]
    for state, paths in prerequisites:
        if ledger.state == state:
            continue
        if ledger.state == (
            "NEW"
            if state == "INPUTS_VERIFIED"
            else prerequisites[[s for s, _ in prerequisites].index(state) - 1][0]
        ):
            ledger.advance(state, paths)
    indexes = read(root / "TABLE_INDEX.json")
    maps = read(root / "CONTROL_MAPS.json")
    arms = protocol["stages"][stage]["arms"]
    gvalues = None
    gtokens = {}
    if geometry:
        gvalues = np.load(verify(geometry["array"]), allow_pickle=False)
        gtokens = {t: i for i, t in enumerate(read(verify(geometry["tokens"])))}

    def gtable(t: FrozenTable) -> np.ndarray | None:
        return (
            None if gvalues is None else gvalues[[gtokens[tok] for tok in t.metadata.sample_token]]
        )

    freeze_path = phase / "ALL_ENDPOINTS_FROZEN.json"
    endpoints = {}
    if freeze_path.exists():
        endpoints = read(freeze_path)
        for record in endpoints.values():
            verify(record)
    else:
        for seed in seeds:
            for fold in range(3):
                t = load_table(indexes[f"outer{fold}_inner_oof"], role="inner_oof")
                g = gtable(t)
                donor = np.load(verify(maps[f"outer{fold}_inner_oof"]), allow_pickle=False)
                original = arm_inputs(t, arms[0], donor, g)
                # S68 ridge standardizes each masked arm. Neural arms share true-input scaler.
                mean, std = scaler(original.features, t.supervision.global_mass)
                for arm in arms:
                    owner.check()
                    key = f"{arm}_seed{seed}_outer{fold}"
                    inputs = arm_inputs(t, arm, donor, g)
                    directory = phase / key
                    if stage == "68":
                        ledger.require("FITTING")
                        t.validate("inner_oof")
                        directory.mkdir(exist_ok=True)
                        path = directory / "ridge.npz"
                        receipt_path = directory / "RIDGE_FIT.json"
                        identity = dict(
                            arm=arm,
                            seed=seed,
                            fold=fold,
                            table=t.seal,
                            implementation=object_digest(lock),
                        )
                        if receipt_path.exists():
                            receipt = read(receipt_path)
                            if receipt["identity"] != identity:
                                raise ValueError("ridge resume identity changed")
                            verify(receipt["endpoint"])
                            endpoints[key] = receipt["endpoint"]
                            continue
                        if path.exists():
                            raise ValueError("unreceipted partial ridge fit")
                        x = inputs.features * feature_mask(arm, 45)
                        losses = 10000 * np.abs(inputs.phases - t.supervision.target_phase[:, None])
                        ridge = FullRegretRidge.fit(x, losses, t.supervision.global_mass)
                        temporary = directory / "ridge.tmp.npz"
                        np.savez(
                            temporary,
                            mean=ridge.mean,
                            scale=ridge.scale,
                            coefficient=ridge.coefficient,
                            intercept=ridge.intercept,
                            ridge=np.array(0.01),
                        )
                        with temporary.open("rb") as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                        endpoints[key] = binding(path)
                        atomic_json(receipt_path, dict(identity=identity, endpoint=endpoints[key]))
                    else:
                        endpoints[key] = fit(
                            t,
                            inputs,
                            ledger=ledger,
                            mean=mean,
                            std=std,
                            arm=arm,
                            seed=seed,
                            outer=fold,
                            output=directory,
                            identity=dict(
                                purpose="scientific",
                                implementation=object_digest(lock),
                                protocol=object_digest(protocol),
                                table=t.seal,
                                control=maps[f"outer{fold}_inner_oof"]["sha256"],
                            ),
                            resume=True,
                            check=owner.check,
                        )
        atomic_json(freeze_path, endpoints)
    if ledger.state == "FITTING":
        ledger.advance("ENDPOINTS_FROZEN", [freeze_path])
    ledger.require("ENDPOINTS_FROZEN")
    expected = {
        f"{arm}_seed{seed}_outer{fold}" for arm in arms for seed in seeds for fold in range(3)
    }
    if set(endpoints) != expected:
        raise ValueError("endpoint coverage mismatch")
    evaluation_path = phase / "ALL_EVALUATIONS.json"
    frames = {}
    quantiles = {}
    eval_records = {}
    if evaluation_path.exists():
        cached = read(evaluation_path)
        frames = {k: pd.read_csv(verify(v)) for k, v in cached["frames"].items()}
        quantiles = cached["quantiles"]
    else:
        for seed in seeds:
            for arm in arms:
                label = arm if len(seeds) == 1 else f"{arm}_seed{seed}"
                parts = []
                train_regret = {}
                for fold in range(3):
                    for role in ("inner_oof", "outer_dev"):
                        t = load_table(indexes[f"outer{fold}_{role}"], role=role)
                        g = gtable(t)
                        donor = np.load(verify(maps[f"outer{fold}_{role}"]), allow_pickle=False)
                        inputs = arm_inputs(t, arm, donor, g)
                        key = f"{arm}_seed{seed}_outer{fold}"
                        if stage == "68":
                            ridge = legacy_fit(verify(endpoints[key]))
                            costs = ridge.predict_regret(inputs.features * feature_mask(arm, 45))
                            selected = costs.argmin(1)
                        else:
                            costs, selected, _, clipping = predict(endpoints[key], inputs)
                            atomic_json(phase / f"{key}_{role}_CLIPPING.json", clipping.tolist())
                        frame = loss_frame(
                            t.metadata,
                            inputs.expert_ttc,
                            costs,
                            selected,
                            arm=arm,
                            seed=seed,
                            fold=fold,
                        )
                        if g is not None:
                            for j in range(28):
                                frame[f"geometry{j}"] = g[:, j]
                        frame.to_csv(phase / f"{key}_{role}.csv", index=False)
                        if role == "inner_oof":
                            train_regret[str(fold)] = np.quantile(
                                frame.regret, [0.5, 0.9, 0.95]
                            ).tolist()
                        else:
                            parts.append(frame)
                merged = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
                path = phase / f"{label}_OOF.csv"
                merged.to_csv(path, index=False)
                frames[label] = merged
                eval_records[label] = binding(path)
                quantiles[label] = train_regret
        atomic_json(evaluation_path, dict(frames=eval_records, quantiles=quantiles))
    if ledger.state == "ENDPOINTS_FROZEN":
        ledger.advance("OUTER_EVALUATED", [evaluation_path])
    refs, ref_quantiles = references(root, indexes)
    frames.update(refs)
    quantiles.update(ref_quantiles)
    if stage == "69":
        s68 = root / "stage68_seeds7/S68-GTRUE45_OOF.csv"
        frames["S68-GTRUE45"] = pd.read_csv(s68)
        quantiles["S68-GTRUE45"] = read(root / "stage68_seeds7/ALL_EVALUATIONS.json")["quantiles"][
            "S68-GTRUE45"
        ]
    if stage == "68":
        zero = frames["S68-GZERO45"].sort_values("sample_token")
        risk = frames["S65-RISK17"].sort_values("sample_token")
        if (
            not np.array_equal(zero.selected_expert, risk.selected_expert)
            or np.max(np.abs(zero.prediction_ttc_s - risk.prediction_ttc_s)) > 1e-10
        ):
            raise ValueError("GZERO45 baseline equivalence failed")
    primary = protocol["stages"][stage]["candidate"]
    decision_path = phase / "DECISION.json"
    if decision_path.exists():
        return read(decision_path), frames
    legacy_index = []
    for legacy_name in ("S65-CE17-REPLAY", "RouterR"):
        legacy = (
            pd.concat(
                [pd.read_csv(root / f"tables/outer{f}_{legacy_name}_replay.csv") for f in range(3)]
            )
            .sort_values("sample_token")
            .reset_index(drop=True)
        )
        for name, frame in frames.items():
            if not frame.sample_token.equals(legacy.sample_token):
                raise ValueError("legacy comparison row identity mismatch")
            comparison = frame[
                ["sample_token", "sequence_id", "track_id", "selected_expert"]
            ].copy()
            comparison["reference_selected"] = legacy.selected_expert
            comparison["delta_loss"] = frame.loss - legacy.loss
            path = phase / f"{name}_vs_{legacy_name}_SELECTION.csv"
            comparison.to_csv(path, index=False)
            legacy_index.append(binding(path))
    atomic_json(phase / "LEGACY_SELECTION_COMPARISONS.json", legacy_index)
    if len(seeds) == 1:
        evidence = diagnostics(frames, primary, phase / "diagnostics", quantiles, owner.check)
    else:
        seed_evidence = {}
        for seed in seeds:
            sub = {arm: frames[f"{arm}_seed{seed}"] for arm in arms}
            sub.update(refs)
            if stage == "69":
                sub["S68-GTRUE45"] = frames["S68-GTRUE45"]
            seed_evidence[str(seed)] = diagnostics(
                sub,
                primary,
                phase / f"diagnostics_seed{seed}",
                {arm: quantiles[f"{arm}_seed{seed}"] for arm in arms},
                owner.check,
            )
        average = {}
        for arm in arms:
            prior = (
                pd.read_csv(root / f"stage{stage}_seeds7/{arm}_OOF.csv")
                .sort_values("sample_token")
                .reset_index(drop=True)
            )
            avg = prior.copy()
            for column in ("loss", "wrong_sign", "oracle_loss", "regret"):
                avg[column] = (
                    prior[column].to_numpy()
                    + sum(frames[f"{arm}_seed{s}"][column].to_numpy() for s in seeds)
                ) / 3
            average[arm] = avg
        average.update(refs)
        if stage == "69":
            average["S68-GTRUE45"] = frames["S68-GTRUE45"]
        averaged = diagnostics(
            average,
            primary,
            phase / "diagnostics_seed_averaged_losses",
            {arm: quantiles[f"{arm}_seed{seeds[0]}"] for arm in arms},
            owner.check,
        )
    coverages = list(phase.glob("diagnostics*/DIAGNOSTIC_COVERAGE.json"))
    if not coverages or any(not read(p)["completed"] for p in coverages):
        raise ValueError("diagnostic coverage incomplete")
    for coverage in coverages:
        verify_coverage(coverage)
    if ledger.state == "OUTER_EVALUATED":
        ledger.advance("DIAGNOSTICS_MATERIALIZED", coverages)
    ledger.require("DIAGNOSTICS_MATERIALIZED")
    if len(seeds) == 1:
        decision = stage_decision(stage, evidence, protocol)
    else:
        decision = replication_decision(stage, seed_evidence, averaged, protocol)
        decision["averaged_predictions"] = False
    atomic_json(decision_path, decision)
    ledger.advance("DECIDED", [decision_path])
    return decision, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--local-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["preflight", "full", "analyze", "package"], required=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    local = read(args.local_inputs)
    if (
        Path(local["output_root"]).resolve() != root
        or Path(local["new_worktree"]).resolve() != ROOT
    ):
        raise ValueError("local root/campaign identity mismatch")
    authorization = read(root / "CONTINUATION_AUTHORIZATION.json")
    if authorization.get("user_authorized") is not True:
        raise ValueError("scoped user authorization missing")
    if (root / "IMPLEMENTATION_LOCK.json").exists() and args.mode == "full" and not args.resume:
        raise ValueError("nonempty campaign requires explicit matching resume")
    atomic_json(
        root / f"COMMAND_{os.getpid()}.json",
        dict(argv=sys.argv, executable=sys.executable, cwd=str(ROOT)),
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = ROOT / "configs/protocol/scientific_recovery_v9_stage66_69.json"
    protocol = read(protocol_path)
    if digest(protocol_path) != digest(args.handoff_root / "PROTOCOL.json"):
        raise ValueError("scientific protocol changed")
    try:
        with CampaignOwner(root) as owner:
            if args.mode == "preflight":
                preflight(root, local)
                return 0
            if args.mode in ("analyze", "package"):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/package_scientific_recovery_v9_stage66_69.py"),
                        "--output-root",
                        str(root),
                    ],
                    check=True,
                )
                return 0
            if not (root / "PREFLIGHT.json").exists():
                preflight(root, local)
            lock = freeze(root, protocol_path)
            decision, _ = execute_phase(root, "67", [7], protocol, lock, owner)
            if decision["next_action"] == "REPLICATE_STAGE67":
                decision, _ = execute_phase(root, "67", [13, 23], protocol, lock, owner)
            elif decision["next_action"] == "RUN_STAGE68":
                geom_path = root / "GEOMETRY_INPUT_LOCK.json"
                if geom_path.exists():
                    geometry = read(geom_path)
                else:
                    geometry = build_geometry(
                        root / "frozen_audit/extracted_input/run/stage63/X3_RAW_BINDING_V2.csv",
                        Path(local["eap_train_root"]),
                        root / "geometry",
                        owner.check,
                    )
                    atomic_json(geom_path, geometry)
                matrix = np.load(verify(geometry["array"]), allow_pickle=False)
                tokens = {t: i for i, t in enumerate(read(verify(geometry["tokens"])))}
                support = {}
                for fold in range(3):
                    table = load_table(
                        read(root / "TABLE_INDEX.json")[f"outer{fold}_inner_oof"], role="inner_oof"
                    )
                    g = matrix[[tokens[t] for t in table.metadata.sample_token]]
                    support[str(fold)] = float(np.mean((g[:, 11] > 0) | (g[:, 23] > 0)))
                atomic_json(root / "GEOMETRY_SUPPORT.json", support)
                if min(support.values()) < 0.2:
                    decision = dict(
                        next_action="GEOMETRY_INSUFFICIENT_SUPPORT",
                        scientific_negative=False,
                        support=support,
                    )
                else:
                    decision, _ = execute_phase(root, "68", [7], protocol, lock, owner, geometry)
                    if decision["next_action"] == "RUN_STAGE69":
                        decision, _ = execute_phase(
                            root, "69", [7], protocol, lock, owner, geometry
                        )
                        if decision["next_action"] == "REPLICATE_STAGE69":
                            decision, _ = execute_phase(
                                root, "69", [13, 23], protocol, lock, owner, geometry
                            )
            atomic_json(
                root / "NEXT_DECISION_V3.json",
                {
                    **decision,
                    "training_commit": lock["training_commit"],
                    "historical_acceptance": "INTEGRITY_BLOCKED",
                    "execution_status": "COMPLETE",
                    "prior_exposure": "LEVEL_0_METADATA_ONLY",
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/package_scientific_recovery_v9_stage66_69.py"),
                    "--output-root",
                    str(root),
                ],
                check=True,
            )
        return 0
    except Exception as error:
        atomic_json(
            root / f"EXECUTION_FAILURE_{os.getpid()}.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                scientific_negative=False,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
