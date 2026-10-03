"""Verify every included full/partial checkpoint from an independently extracted bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    sys.path[:0] = [str(root / "provenance"), str(root / "provenance/src")]
    from operational.simplex_t_cost_context.regenerate import record, resource_guard, sha

    resource_guard(root)
    import torch

    from e_jepa_ttc.simplex_t.training import load_checkpoint

    index = record(root / "ENDPOINT_INVENTORY.json")
    manifest = record(root / "CONTENT_MANIFEST.json")
    rows = []
    for fit in index["fits"]:
        if fit["checkpoint"] is None:
            continue
        resource_guard(root)
        prefix = "h16" if fit["family"] == "N1" else "execution"
        name = prefix + "/fits/" + fit["id"] + "/checkpoint_last.pt"
        path = root / name
        if (
            sha(path) != fit["checkpoint_sha256"]
            or sha(path) != manifest["members"][name]["sha256"]
        ):
            raise ValueError("included checkpoint bytes differ from frozen inventory")
        state = load_checkpoint(path)
        protocol_path = root / (
            "h16/PROTOCOL.json" if fit["family"] == "N1" else "PROTOCOL_COST_CONTEXT.json"
        )
        p = record(protocol_path)
        expected = p["sources"][str(fit["fold"])]
        source = (
            expected["train_sha256"]
            if fit["family"] == "N1"
            else expected["wrapped"][fit["arm"]]["train_sha256"]
        )
        identity = state["identity"]
        if (
            state["completed_updates"] != fit["saved_updates"]
            or identity["seed"] != fit["seed"]
            or identity["source"] != source
            or identity["freeze"] != sha(protocol_path)
            or identity["device"] != "cpu"
            or identity["batch"] != 128
            or identity["endpoint"] != 2500
        ):
            raise ValueError("checkpoint count, source or scientific recipe identity differs")
        if any(v.dtype != torch.float32 for v in state["model"].values()):
            raise ValueError("scientific model parameters are not FP32")
        rows.append(
            dict(
                id=fit["id"],
                saved_updates=fit["saved_updates"],
                status=state["status"],
                sha256=sha(path),
                full_state_integrity_verified=True,
            )
        )
        del state
    report = dict(
        status="EXTRACTED_COMPLETE_STATE_CHECKPOINTS_VERIFIED",
        checkpoints=rows,
        optimizer_updates=0,
        raw_reconstruction=False,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
