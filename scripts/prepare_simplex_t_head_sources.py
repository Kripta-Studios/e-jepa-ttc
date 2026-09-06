"""Prepare all D0 source identities from complete compiled folds; never fit or score."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources, CompiledFold
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.registry import registered_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dedup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if args.output.exists():
        raise FileExistsError("source preparation receipt already exists")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(args.coordination.read_text(encoding="utf-8"))
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    snapshot = admitted([args.output.parent])
    if not snapshot["has_headroom"] or snapshot["host_available_bytes"] < 12 * 1024**3:
        raise RuntimeError("RESOURCE_PAUSE: reserve4 GiB while retaining8 GiB available")
    folds = {}
    for outer in range(3):
        folder = args.compiled_root / f"outer{outer}"
        folds[outer] = CompiledFold(folder, compute_file_hash(str(folder / "COMPILED.json")))
    # Prepare possible branches now. This is NOT a practical-gate decision and
    # cannot authorize optional fits; queue prerequisite checks remain mandatory.
    graph = registered_graph(
        d1=False,
        density=False,
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    sources = CampaignSources(
        graph,
        folds,
        index_root=args.index,
        dedup_root=args.dedup,
        historical_root=Path(ancestry["path"]).parent,
        ancestry_sha256=ancestry["sha256"],
        allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
    )
    identities = sources.identities()
    write_new_json(
        args.output,
        {
            "schema": "simplex_t_d0_source_preparation_v1",
            "status": "SOURCES_PREPARED_NOT_SCIENTIFIC_FREEZE",
            "namespace": "SIMPLEX_T_QUERY_CONTEXT_AMENDMENT",
            "ack_sha256": config["ack_sha256"],
            "ancestry_sha256": ancestry["sha256"],
            "compiled_folds": {
                str(outer): {"path": str(pin.path.resolve()), "sha256": pin.sha256}
                for outer, pin in folds.items()
            },
            "possible_graph_not_gate_authorization": [asdict(spec) for spec in graph],
            "source_identities": identities,
            "optimizer_updates": 0,
            "scores_read": False,
            "resources_at_start": snapshot,
        },
    )
    print(json.dumps({"prepared_fit_identities": len(identities), "scientific_freeze": False}))


if __name__ == "__main__":
    main()
