"""Read-only preflight of the exact H16 TRAIN recipe, lineage and historical controls."""

from __future__ import annotations

import gc
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from common import (
    ARM,
    HISTORICAL_FREEZE,
    OUT,
    ROOT,
    Resources,
    atomic_json,
    digest,
    historical_spec,
    ids,
    inventory,
    record,
)


def main() -> None:
    import numpy as np
    import torch
    from e_jepa_ttc.simplex_t.arms import resolve_arm
    from e_jepa_ttc.simplex_t.configuration_preflight import open_acknowledged_source_configuration
    from e_jepa_ttc.simplex_t.training import load_checkpoint, state_digest

    check = Resources()
    check.check()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    launch = record(ROOT / "artifacts/simplex_t/scientific_campaign/launches/T6.json")
    freeze_path = Path(launch["freeze"])
    if digest(freeze_path) != HISTORICAL_FREEZE:
        raise ValueError("historical scientific freeze differs")
    freeze = record(freeze_path)
    if freeze["code_commit"] != "0c1a7285b6b5af869b0bf5a13a9629f08997e7a2":
        raise ValueError("historical code identity differs")
    paths = [freeze_path, Path(launch["local_paths"]), Path(launch["source_configuration"])]
    static = record(OUT / "STATIC_TRAIN_AUDIT.json")
    paths += [Path(r["path"]) for r in static["files"]]
    # Verify all frozen scientific source bytes, without invoking any historical stage.
    for row in freeze["files"]:
        if row["category"] == "code":
            path = Path(launch["roots"][row["root"]]) / row["relative_path"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"historical scientific code changed: {path}")
            paths.append(path)
    s, authority = open_acknowledged_source_configuration(
        Path(launch["local_paths"]),
        Path(launch["source_configuration"]),
        launch["source_configuration_sha256"],
    )
    # Source manifests disclose paths; only inspect acknowledged TRAIN/cache lineage.
    paths += [
        s.index_root / "INDEX_MANIFEST.json",
        s.index_root / "query_context_index.npz",
        s.historical_root / "NESTED_ANCESTRY_AUDIT.json",
    ]
    for fold in range(3):
        paths += list(s.folds[fold].path.glob("*.npy")) + [
            s.folds[fold].path / "COMPILED.json",
            s.dedup_root / f"outer{fold}.npz",
        ]
        b = s.expansion_folds[fold]
        paths += list(b.compiled.glob("*.npy")) + [
            b.compiled / "COMPILED.json",
            b.index_manifest,
            b.dedup,
            b.pool,
            b.metadata,
            b.labels,
        ]
        paths += [
            s.historical_root / "tables" / f"outer{fold}_{role}.{suffix}"
            for role in ("inner_oof", "outer_dev")
            for suffix in ("csv", "npz")
        ]
    historical = {}
    for stage in ("T2", "T3", "T5"):
        binding = launch["publications"][stage]
        for field in ("publication", "endpoints"):
            path = Path(binding[field])
            if digest(path) != binding[field + "_sha256"]:
                raise ValueError("historical control seal changed")
            paths.append(path)
        endpoints = record(Path(binding["endpoints"]))["fits"]
        pub = record(Path(binding["publication"]))
        for row in endpoints:
            fit = row["fit"]
            if fit["name"] not in {ARM, "TPR-D1-H8-C160"}:
                continue
            path = Path(binding["checkpoint_root"]) / row["checkpoint"]
            if digest(path) != row["checkpoint_sha256"]:
                raise ValueError("historical control checkpoint changed")
            state = load_checkpoint(path)
            if state["completed_updates"] != 2500 or state["status"] != "COMPLETED":
                raise ValueError("historical control is not a valid endpoint")
            if state["identity"]["torch_version"] != str(torch.__version__):
                raise ValueError("Torch differs from executed historical recipe")
            prediction = Path(binding["publication"]).parent / pub["fits"][row["key"]]["path"]
            if digest(prediction) != pub["fits"][row["key"]]["sha256"]:
                raise ValueError("historical prediction changed")
            historical[row["key"]] = dict(
                endpoint=row,
                identity=state["identity"],
                checkpoint=str(path),
                prediction=str(prediction),
                prediction_sha256=digest(prediction),
            )
            paths += [path, prediction]
            del state
    source_rows = {}
    for fold in range(3):
        check.check()
        spec = historical_spec(s, fold)
        train = s.train(spec)
        dev = s.source(spec, "outer_dev")
        h8 = s.train(historical_spec(s, fold, 8))
        if {"inner_oof": train.identity_sha256, "outer_dev": dev.identity_sha256} != freeze[
            "source_identities"
        ][f"T3/{ARM}/fold{fold}/seed7"]:
            raise ValueError("H16 source identity differs from historical execution")
        for field in ("features", "history", "target_phase", "mass", "anchor_us", "available_us"):
            if not np.array_equal(getattr(train, field), getattr(h8, field)):
                raise ValueError(f"H8/H16 underlying pool differs: {field}")
        if not np.array_equal(train.normalizer.mean, h8.normalizer.mean) or not np.array_equal(
            train.normalizer.scale, h8.normalizer.scale
        ):
            raise ValueError("H8/H16 normalizers differ")
        normalizer_sha = state_digest(
            dict(
                mean=torch.from_numpy(train.normalizer.mean),
                scale=torch.from_numpy(train.normalizer.scale),
                ids=train.normalizer.consumed_ids_sha256,
            )
        )
        # Historical bindings deliberately encode history length; no new arm enters that graph.
        source_rows[str(fold)] = dict(
            train_sha256=train.identity_sha256,
            dev_sha256=dev.identity_sha256,
            h8_train_sha256=h8.identity_sha256,
            normalizer_sha256=normalizer_sha,
            train_population=train.population,
            dev_population=dev.population,
            model=asdict(resolve_arm(spec, s.graph).model),
            history=16,
            full_history_fraction=float(np.mean((train.history >= 0).sum(1) == 16)),
        )
        print(
            json.dumps(dict(status="TRAIN_H16_VERIFIED", fold=fold, **source_rows[str(fold)])),
            flush=True,
        )
        del train, dev, h8
        s.release()
        gc.collect()
    s.release()
    # Bind frozen authority metadata in permitted roots; leave Stage70 files unopened.
    for row in freeze["files"]:
        if row["root"] in {"work", "historical"} and row["category"] != "code":
            path = Path(launch["roots"][row["root"]]) / row["relative_path"]
            if path.suffix in {".json", ".yaml", ".yml"}:
                if digest(path) != row["sha256"]:
                    raise ValueError(f"frozen lineage metadata changed: {path}")
                paths.append(path)
    result: dict = dict(
        status="PREFLIGHT_VERIFIED_ZERO_UPDATES",
        launch=launch,
        authority=authority,
        sources=source_rows,
        historical_controls=historical,
        fits=ids(),
        input_inventory=inventory(paths),
        current_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        current_branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        preserved_git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        scientific_optimizer_updates=0,
        technical_optimizer_updates=0,
    )
    atomic_json(OUT / "PREFLIGHT.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                files=len(result["input_inventory"]),
                controls=len(historical),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
