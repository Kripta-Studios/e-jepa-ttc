"""Verify physical TRAIN cache and frozen producer bytes without importing Torch."""

from __future__ import annotations

import json
from pathlib import Path

from common import HISTORICAL_FREEZE, OUT, ROOT, atomic_json, digest, ids, inventory, record


def main() -> None:
    launch = record(ROOT / "artifacts/simplex_t/scientific_campaign/launches/T6.json")
    freeze = Path(launch["freeze"])
    if digest(freeze) != HISTORICAL_FREEZE:
        raise ValueError("historical freeze changed")
    config = record(Path(launch["source_configuration"]))
    roots = {k: Path(v) for k, v in launch["roots"].items()}
    paths = [freeze, Path(launch["source_configuration"])]
    folds = []
    for fold in range(3):
        for name, row, pin in (
            (
                "original",
                config["original"]["folds"][str(fold)]["path"],
                config["original"]["folds"][str(fold)]["sha256"],
            ),
            (
                "expansion",
                config["expansion"][str(fold)]["compiled"],
                config["expansion"][str(fold)]["compiled_sha256"],
            ),
        ):
            root = roots[row["root"]] / row["relative_path"]
            manifest = root / "COMPILED.json"
            if digest(manifest) != pin:
                raise ValueError(f"compiled manifest changed: {manifest}")
            m = record(manifest)
            paths.append(manifest)
            for key, sha in m["arrays"].items():
                path = root / (key + ".npy")
                if digest(path) != sha:
                    raise ValueError(f"compiled cache changed: {path}")
                paths.append(path)
            folds.append(
                dict(
                    fold=fold,
                    pool=name,
                    queries=m["queries"],
                    observations=m.get("observations"),
                    manifest_sha256=pin,
                    arrays=len(m["arrays"]),
                )
            )
            print(json.dumps(dict(status="CACHE_BYTES_VERIFIED", **folds[-1])), flush=True)
    producer_inventory = (
        ROOT
        / "artifacts/simplex_t/closure_20261002/essential_delivery/supplement"
        / "EXPERT_CHECKPOINT_INVENTORY.json"
    )
    paths.append(producer_inventory)
    producers = record(producer_inventory)["experts"]
    for row in producers:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"frozen producer checkpoint changed: {path}")
        paths.append(path)
    # Resolve TRAIN supervision only through declared source bindings.
    for binding in config["expansion"].values():
        for key in ("metadata", "labels", "index_manifest", "pool"):
            ref = binding[key]
            path = roots[ref["root"]] / ref["relative_path"]
            if digest(path) != binding[key + "_sha256"]:
                raise ValueError(f"declared TRAIN dependency changed: {path}")
            paths.append(path)
    atomic_json(
        OUT / "STATIC_TRAIN_AUDIT.json",
        dict(
            status="CACHE_AND_PRODUCER_BYTES_VERIFIED_NOT_FULL_SOURCE_ADMISSION",
            folds=folds,
            producers=len(producers),
            files=inventory(paths),
            fits=ids(),
            scientific_optimizer_updates=0,
            technical_optimizer_updates=0,
            pending="Torch-based normalizer/source/endpoint recipe identity admission",
        ),
    )
    print("STATIC_TRAIN_AUDIT_COMPLETE_ZERO_UPDATES", flush=True)


if __name__ == "__main__":
    main()
