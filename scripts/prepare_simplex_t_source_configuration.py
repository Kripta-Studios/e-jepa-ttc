"""Bind all registered D0/D1/density sources without reading target values or fitting."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.configuration_preflight import open_acknowledged_source_configuration
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.expansion_authority import verify_expansion_authority
from e_jepa_ttc.simplex_t.lifecycle import admitted


def configuration(work: Path, ack: dict) -> dict:
    """Require all nine canonical compiled manifests; no optional fallback selection."""
    temporal = "artifacts/simplex_t/T1/"

    def reference(path: str, root: str = "work") -> dict:
        return {"root": root, "relative_path": path}

    def original_fold(outer: int) -> dict:
        folder = temporal + f"compiled_context/outer{outer}"
        return {"path": reference(folder), "sha256": sha256(work / folder / "COMPILED.json")}

    roles = ack["interfaces"]["role_manifest"]["roles"]
    result = {
        "schema": "simplex_t_campaign_sources_v1",
        "original_sequences": sorted(roles["original"]),
        "expansion_sequences": sorted(roles["expansion"]),
        "original": {
            "index_root": reference(temporal + "query_context_index"),
            "dedup_root": reference(temporal + "query_context_dedup"),
            "historical_root": reference(".", "historical"),
            "ancestry_sha256": ack["producers"]["authoritative_historical_manifest"]["sha256"],
            "folds": {str(outer): original_fold(outer) for outer in range(3)},
        },
    }
    pools = {
        "expansion": (
            "expansion",
            "EXPANSION_POOL_PLAN.json",
            "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849",
        ),
        "dense": (
            "dense",
            "MATCHED_CONTROL_POOL_PLAN.json",
            "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c",
        ),
    }
    amendment_path = work / "configs/experiment/simplex_t_throughput_amendment.json"
    selected_binding = None
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text("utf-8"))
        if amendment["id"] != "SIMPLEX_T_THROUGHPUT_2026-09-08":
            raise ValueError("unknown throughput amendment")
        amended_pool = "MATCHED_CONTROL_POOL_DENSITY_20260908.json"
        amended_hash = "00322ff06093aa5dc46c2aaed26e42961dc3349fc85823001672d0c593cca122"
        amended_path = work / "artifacts/simplex_t/T0" / amended_pool
        if sha256(amended_path) != amended_hash:
            raise ValueError("amended matched pool missing or changed; no legacy fallback")
        amended = json.loads(amended_path.read_text("utf-8"))
        if not amended["dense_membership_unchanged"]:
            raise ValueError("amendment changes acknowledged DENSE membership")
        selected_binding = amended["query_selection"]
        if sha256(Path(selected_binding["path"])) != selected_binding["sha256"]:
            raise ValueError("amended D1 selection changed")
        pools["dense"] = ("dense", amended_pool, amended_hash)
    for name, (prefix, pool_file, pool_hash) in pools.items():
        pool_path = "artifacts/simplex_t/T0/" + pool_file
        if sha256(work / pool_path) != pool_hash:
            raise ValueError("registered TRAIN pool changed")
        folds = {}
        for outer in range(3):
            folder = temporal + f"compiled_{prefix}_context/outer{outer}"
            manifest_path = work / folder / "COMPILED.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["outer"] != outer or manifest["pool"] != (
                "D1" if name == "expansion" else "DENSE_OLD"
            ):
                raise ValueError("compiled TRAIN pool/fold differs")
            if name == "expansion" and selected_binding is not None:
                if manifest.get("query_selection") != selected_binding:
                    raise ValueError("D1 compiled selection and matched controls differ")
            folds[str(outer)] = {
                "compiled": reference(folder),
                "compiled_sha256": sha256(manifest_path),
                "index_manifest": reference(
                    temporal + f"{prefix}_query_context_index/INDEX_MANIFEST.json"
                ),
                "index_manifest_sha256": sha256(
                    work / temporal / f"{prefix}_query_context_index/INDEX_MANIFEST.json"
                ),
                "dedup": reference(temporal + f"{prefix}_query_context_dedup/outer{outer}.npz"),
                "pool": reference(pool_path),
                "pool_sha256": pool_hash,
                "metadata": reference("data/train.parquet", "garl"),
                "metadata_sha256": (
                    "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6"
                ),
                "labels": reference("annotations/train.parquet", "garl"),
                "labels_sha256": "64e9a81232643ecce1d2f66e906c7cff6ea1e9e6958f81588b95e112693b11ad",
                "cache_identity_sha256": manifest["cache_identity_sha256"],
            }
        result[name] = folds
    result["matched"] = {
        "pool": reference("artifacts/simplex_t/T0/" + pools["dense"][1]),
        "pool_sha256": pools["dense"][2],
        "original_index_manifest_sha256": (
            "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if args.other_reserved_bytes < 0 or not output.is_relative_to(work / "artifacts"):
        raise ValueError("nonnegative reservation and companion artifact destination required")
    local_hash = sha256(args.local_paths)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    if Path(paths["worktree"]).resolve() != work:
        raise ValueError("local paths name another worktree")

    def resources() -> bool:
        state = admitted([work])
        return state["has_headroom"] and shared_write_admission(
            state["written_volume_free_bytes"][0], args.other_reserved_bytes + 2_097_152
        )

    if not resources():
        return 3
    ack = verified_ack(
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318",
    )
    verify_expansion_authority(args.local_paths, resource_ok=resources)
    required = [
        work / f"artifacts/simplex_t/T1/{folder}/outer{outer}/COMPILED.json"
        for folder in ("compiled_context", "compiled_expansion_context", "compiled_dense_context")
        for outer in range(3)
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(
            json.dumps(
                {"status": "SOURCE_CONFIGURATION_PREREQUISITES_INCOMPLETE", "missing": missing}
            )
        )
        return 10
    config = configuration(work, ack)
    if args.verify_only:
        if not output.is_file():
            return 10
        if output.stat().st_size > 1_048_576 or json.loads(output.read_bytes()) != config:
            raise ValueError("source configuration differs from registered bindings")
        candidate = output
    else:
        if output.exists():
            raise FileExistsError("preserve source configuration; use --verify-only")
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate = output.with_name(output.stem + ".candidate_" + uuid.uuid4().hex + ".json")
        write_new_json(candidate, config)
    candidate_hash = sha256(candidate)
    sources, inspection = open_acknowledged_source_configuration(
        args.local_paths, candidate, candidate_hash
    )
    sources.release()
    if (
        sha256(args.local_paths) != local_hash
        or configuration(work, ack) != config
        or sha256(candidate) != candidate_hash
    ):
        raise ValueError("source binding changed during inspection")
    if not resources():
        return 3
    if not args.verify_only:
        write_new_json(output, config)
    print(
        json.dumps(
            {
                "status": "SOURCE_CONFIGURATION_BOUND_NOT_PAYLOAD_VERIFIED_OR_FROZEN",
                "sha256": sha256(output),
                "inspection": inspection,
                "optimizer_updates": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InterruptedError, RuntimeError) as error:
        if not str(error).startswith(("PAUSED_RESOURCE:", "RESOURCE_PAUSE:")):
            raise
        print(
            json.dumps({"status": "PAUSED_RESOURCE", "reason": str(error), "optimizer_updates": 0})
        )
        raise SystemExit(3) from error
