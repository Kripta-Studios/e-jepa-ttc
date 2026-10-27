"""Zero-update proof for byte-identical local RGB cache routing on Runpod."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.inventory import digest
from operational.runpod_migration.local_cache import prove_rgb_route


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    parser.add_argument("--budget-bytes", type=int, default=31_000_000_000)
    args = parser.parse_args()

    import runpod_technical_overlay_v3 as technical

    setup = technical.prepare(args.bundle, args.run, args.budget_bytes)
    config = setup["config"]
    from operational.final_refit.fold_sources import load_fold_rgb_source

    source = load_fold_rgb_source(
        Path(config["rgb_source"]),
        Path(config["teacher"]),
        Path(config["fold_contract"]),
        args.rotation,
    )
    try:
        proof = prove_rgb_route(
            source,
            logical_root=args.bundle,
            stage_root=args.stage,
            readiness_path=args.readiness,
        )
    finally:
        source.close()
    repository = Path(__file__).resolve().parents[1]
    result = {
        **proof,
        "schema": "ejepa_runpod_local_cache_preflight_v1",
        "rotation": args.rotation,
        "fit": "R_A5",
        "migration_binding_sha256": digest(args.run / "MIGRATION_BINDING.json"),
        "runtime_sources": {
            "local_cache": digest(
                repository / "operational/runpod_migration/local_cache.py"
            ),
            "preflight": digest(Path(__file__)),
        },
        "training_called": False,
        "optimizer_updates": 0,
        "scientific_execution_allowed": False,
    }
    result.pop("proof_sha256", None)
    from operational.final_refit.recipe import canonical_sha256

    result["proof_sha256"] = canonical_sha256(result)
    output = args.run / f"LOCAL_CACHE_NUMERIC_PROOF_R{args.rotation}.json"
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
