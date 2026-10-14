"""Admit strong historical RGB/fusion references without selecting on V outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.infer_experts import _parents

from .common import RUN, owner


def admit() -> dict:
    """Verify immutable ancestry, H supervision and complete V query coverage, read-only."""
    root = owner()
    parent_path = root / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    parent_record = read(parent_path)
    parent_hashes = _parents(root, sorted(parent_record["endpoints"]))
    if parent_hashes != {
        name: value["checkpoint_sha256"] for name, value in parent_record["endpoints"].items()
    }:
        raise ValueError("Historical expert ancestry changed")
    manifest_hashes = {role: sha(root / f"{role}_MANIFEST.json") for role in ("H", "V")}
    rows = {}
    for role in ("H", "V"):
        manifest = read(root / f"{role}_MANIFEST.json")
        if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
            raise ValueError("Historical query metadata changed")
        rows[role] = pd.read_parquet(manifest["rows_path"], columns=["sample_token", "group_id"])
        if len(rows[role]) != manifest["population_size"] or not rows[role].sample_token.is_unique:
            raise ValueError("Historical population coverage changed")
    v_receipt_path = root / "predictions/V/HEAD_PREDICTIONS.json"
    v_receipt = read(v_receipt_path)
    if (
        v_receipt["status"] != "COMPLETE"
        or v_receipt["manifest_v_sha256"] != manifest_hashes["V"]
        or v_receipt["population"] != len(rows["V"])
    ):
        raise ValueError("Historical V receipt changed")
    heads = {}
    for fit, cache_name, norm_name in (
        ("R_CTX", "R_HISTORY.npz", "R_NORMALIZER.json"),
        ("F_TRUE", "FUSION.npz", "FUSION.json"),
    ):
        directory = root / "fits" / fit
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        journal = read(directory / "UPDATE_JOURNAL.json")
        checkpoint = directory / "checkpoint_last.pt"
        if (
            receipt["status"] != "COMPLETE"
            or not receipt["scientific_endpoint"]
            or receipt["completed_updates"] != 2500
            or journal["completed_updates"] != 2500
            or journal["durable_updates"] != 2500
            or journal["pending_update_upper"] != 0
            or sha(checkpoint) != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Historical head endpoint is incomplete or changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = payload["identity"]
        cache = root / "features/H" / cache_name
        normalizer = root / "features/H" / norm_name
        if (
            payload["completed_updates"] != 2500
            or payload["identity_sha256"] != receipt["identity_sha256"]
            or identity["identity_sha256"] != receipt["identity_sha256"]
            or journal["identity_sha256"] != receipt["identity_sha256"]
            or identity["role_manifest_sha256"] != manifest_hashes["H"]
            or identity["parent_sha256"] != sha(parent_path)
            or identity["cache_sha256"] != sha(cache)
            or identity["normalizer_sha256"] != sha(normalizer)
        ):
            raise ValueError("Historical head ancestry or H training inputs changed")
        with np.load(cache, allow_pickle=False) as archive:
            if set(archive["role"].astype(str)) != {"H"} or not np.array_equal(
                archive["sample_token"].astype(str), rows["H"].sample_token.to_numpy()
            ):
                raise ValueError("Historical head training cache is not the complete H cohort")
        endpoint = {
            key: identity[key] for key in ("identity_sha256", "normalizer_sha256", "parent_sha256")
        }
        endpoint["checkpoint_sha256"] = receipt["checkpoint_sha256"]
        if v_receipt["endpoint_bindings"][fit] != endpoint:
            raise ValueError("Historical V output belongs to another head endpoint")
        artifact = v_receipt["artifacts"][fit]
        csv = root / "predictions/V" / f"{fit}.csv"
        if (
            artifact["rows"] != len(rows["V"])
            or not artifact["native_unfiltered"]
            or artifact["csv_sha256"] != sha(csv)
        ):
            raise ValueError("Historical V output bytes or coverage changed")
        # Deliberately do not load prediction or target columns for this inventory.
        cohort = pd.read_csv(csv, usecols=["sample_token", "group_id"])
        if any(
            not np.array_equal(cohort[key].to_numpy(), rows["V"][key].to_numpy()) for key in cohort
        ):
            raise ValueError("Historical V table changes query order or groups")
        heads[fit] = {
            **endpoint,
            "H_cache_sha256": identity["cache_sha256"],
            "H_population": len(rows["H"]),
            "V_population": len(rows["V"]),
            "V_prediction_sha256": artifact["csv_sha256"],
            "H_supervised": True,
            "H_metrics_would_be_in_sample": True,
        }
    normalizers = {}
    for name in ("E_NORMALIZER.json", "R_NORMALIZER.json"):
        path = root / "features/H" / name
        if read(path)["fit_role"] != "H":
            raise ValueError("Historical normalization role changed")
        normalizers[name] = sha(path)
    result = {
        "status": "ADMITTED_AS_HISTORICAL_COMPARATORS",
        "heads": heads,
        "expert_endpoints": parent_hashes,
        "normalizers": normalizers,
        "manifest_hashes": manifest_hashes,
        "V_receipt_sha256": sha(v_receipt_path),
        "source_sha256": sha(Path(__file__)),
        "new_parent_selected": False,
        "V_prediction_values_loaded": False,
        "V_metrics_computed": False,
        "new_scientific_recipe_registered": False,
        "optimizer_updates": 0,
        "interpretation": (
            "R_CTX/F_TRUE trained on H. Any new H head using them is sequential supervised "
            "fitting, not OOF or independent H evaluation. Rebuild new-producer features."
        ),
        "cost_scope": "historical HEAD_ONLY profiles are not full-pipeline costs",
    }
    freeze(RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json", result)
    return result


if __name__ == "__main__":
    print(admit())
