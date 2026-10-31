"""Materialize label-blind Test12 deployment states and causal geometry for audit."""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_startup.geometry import startup_geometry
from scripts.build_phase_safe_source_audit import causal_history
from scripts.final_train40_transfer import HEAD


def query_history(
    query: dict[str, Any], observations: dict[str, Any], owners: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Collect only supplied history of the same object, preserving causal availability."""
    row: dict[str, Any] = {
        "query_time_us": query["query_time_us"],
        "cutoff_us": query["query_time_us"],
        "producer_times_us": [],
        "producer_available_us": [],
        "producer_boxes_xyxy": [],
        "observation_times_us": [],
        "observation_available_us": [],
        "observation_boxes_xyxy": [],
    }
    for identity in query["rgb_history"]:
        item = observations[identity]
        owner = owners.get(item["sample_token"])
        if owner is None or any(owner[k] != query[k] for k in ("sequence_id", "track_id")):
            raise ValueError("History crosses an object or sequence boundary")
        for output, source in (
            ("observation_times_us", "producer_times_us"),
            ("observation_available_us", "producer_available_us"),
            ("observation_boxes_xyxy", "producer_boxes_xyxy"),
        ):
            row[output].append(item[source])
    return causal_history(row)


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    import json

    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"Missing archive member: {name}")
    return json.loads(stream.read())


def build(args: argparse.Namespace) -> None:
    """Verify original evidence and save no targets and no new-candidate predictions."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths = verified_upstream_paths(args.upstream)
    parent = read(args.upstream / "BUNDLE.json")["parent"]
    if file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Changed trusted label-blind Test12 inventory")
    inputs = read(args.inputs)
    if _contains_private_key(inputs):
        raise ValueError("Forbidden target field in Test12 inventory")
    queries = {row["sample_token"]: row for row in inputs["queries"]}
    aliases = inputs["physical_aliases"]
    if len(queries) != 6762 or set(aliases) != set(queries):
        raise ValueError("Expected exact official logical population")
    canonical = sorted(set(aliases.values()))
    if len(canonical) != 5805 or not set(canonical) <= set(queries):
        raise ValueError("Unexpected canonical physical population")
    for token, original in aliases.items():
        if any(
            queries[token][k] != queries[original][k]
            for k in ("sequence_id", "track_id", "query_time_us", "rgb_history")
        ):
            raise ValueError("Alias changes physical query identity")
    closure = read(args.preservation)
    sources = [
        Path(__file__),
        Path("scripts/build_phase_safe_source_audit.py"),
        Path("operational/phase_startup/geometry.py"),
        Path("operational/final_refit/training.py"),
        Path("operational/final_refit/test12.py"),
        Path("operational/phase_robust/bundle.py"),
        Path("src/e_jepa_ttc/simplex_t/phase.py"),
        Path("operational/phase_adapt_cache.py"),
    ]
    source_hashes = {str(p): file_sha(p) for p in sources}
    records: dict[str, dict[str, Any]] = {}
    archive_inputs = {}
    for key, entry in sorted(closure["groups"].items()):
        if not key.startswith("test12_predictions_"):
            continue
        archive_path = args.archive_root / Path(entry["archive"]).name
        if file_sha(archive_path) != entry["sha256"]:
            raise ValueError(f"Changed Test12 archive: {key}")
        archive_inputs[key] = {"path": str(archive_path.resolve()), "sha256": entry["sha256"]}
        with tarfile.open(archive_path) as archive:
            names = archive.getnames()
            admission = _json_member(
                archive, next(n for n in names if n.endswith(".admission.json"))
            )
            complete = _json_member(archive, next(n for n in names if n.endswith(".complete.json")))
            if (
                admission["model_seal_sha256"] != parent["model_seal_sha256"]
                or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
                or complete["status"] != "COMPLETE"
                or complete["run_binding_sha256"] != admission["run_binding_sha256"]
            ):
                raise ValueError("Test12 deployment or sequence completion binding differs")
            sequence_tokens = set()
            for name in names:
                if "/diagnostics/" not in name or not name.endswith(".json"):
                    continue
                receipt = _json_member(archive, name)
                token = receipt["sample_token"]
                if token in records or token not in canonical:
                    raise ValueError("Duplicated or unknown canonical Test12 token")
                if (
                    receipt["private_targets_read"] is not False
                    or receipt["run_binding_sha256"] != admission["run_binding_sha256"]
                    or receipt["fallback_source"] != "PHASE_ONLY"
                ):
                    raise ValueError("Unexpected diagnostic lineage or fallback")
                query = queries[token]
                if any(
                    receipt[k] != query[k]
                    for k in ("sequence_id", "track_id", "query_time_us", "rgb_history")
                ):
                    raise ValueError("Diagnostic and public sensor identity differ")
                stream = archive.extractfile(name.removesuffix(".json") + ".npz")
                if stream is None:
                    raise ValueError("Missing diagnostic NPZ")
                raw = stream.read()
                if hashlib.sha256(raw).hexdigest() != receipt["npz_sha256"]:
                    raise ValueError("Changed diagnostic arrays")
                with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
                    features = stored[f"{HEAD}.input_0"]
                    widths = stored[f"{HEAD}.input_1"]
                    if features.shape != (1, 340) or widths.shape != (1, 2):
                        raise ValueError("Unexpected frozen head input dimensions")
                    f0 = float(stored[f"{HEAD}.output_ttc"][0])
                    if f0 != receipt["prediction"]:
                        raise ValueError("Archived native prediction differs")
                    context = np.array(
                        [
                            stored[f"{name}.output_raw_location"][0]
                            for name in ("R_CTX", "E_CTX_MATCHED")
                        ],
                        np.float32,
                    )
                    np.testing.assert_array_equal(features[0, [160, 330]], context)
                    records[token] = {
                        "features": features[0],
                        "rgb_width_logits": widths[0],
                        "F0": f0,
                        "phase": stored[f"{HEAD}.output_raw_location"][0],
                        "context": context,
                    }
                sequence_tokens.add(token)
            if sequence_tokens != set(complete["canonical_tokens"]):
                raise ValueError("Sequence diagnostics do not cover completion receipt")
        print(f"Verified {key}: {len(sequence_tokens)} physical queries", flush=True)
    if set(records) != set(canonical):
        raise ValueError("Missing canonical diagnostic states")
    features = np.stack([records[t]["features"] for t in canonical])
    widths = np.stack([records[t]["rgb_width_logits"] for t in canonical])
    f0 = np.array([records[t]["F0"] for t in canonical], np.float32)
    phases = np.array([records[t]["phase"] for t in canonical], np.float32)
    context = np.stack([records[t]["context"] for t in canonical])
    if not all(np.isfinite(a).all() for a in (features, widths, f0, phases, context)):
        raise ValueError("Nonfinite original head state")
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(
        torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    )
    with torch.inference_mode():
        replay = baseline(torch.from_numpy(features), torch.from_numpy(widths))
        # CPU batched replay may differ slightly from original batch1 GPU execution.
        torch.testing.assert_close(
            replay["raw_location"], torch.from_numpy(phases), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(replay["ttc"], torch.from_numpy(f0), rtol=1e-4, atol=1e-4)
        native = np.column_stack((f0, phase_to_ttc(torch.from_numpy(context)).numpy())).astype(
            np.float32
        )
    geometry, qualities, reasons, history_counts = [], [], [], []
    for token in canonical:
        query = queries[token]
        boxes, clocks = query_history(query, inputs["rgb_observations"], queries)
        physical = startup_geometry(boxes, clocks, query["query_time_us"])
        geometry.append(np.clip(physical["ttc"], -60, 60) if physical["valid"] else np.nan)
        qualities.append(physical["quality"])
        reasons.append(physical["reason"])
        history_counts.append(len(clocks))
    index = {t: i for i, t in enumerate(canonical)}
    logical = [r["sample_token"] for r in inputs["queries"]]
    expanded = np.array([index[aliases[t]] for t in logical], np.int64)
    arrays = {
        "native": native[expanded],
        "geometry": np.array(geometry, np.float32)[expanded],
        "quality": np.array(qualities, np.float32)[expanded],
        "features": features[expanded],
        "rgb_width_logits": widths[expanded],
        "token": np.array(logical),
        "canonical_token": np.array([aliases[t] for t in logical]),
        "group": np.array([queries[t]["sequence_id"] for t in logical]),
        "track": np.array([queries[t]["track_id"] for t in logical]),
        "clocks": np.array([queries[t]["query_time_us"] for t in logical], np.int64),
        "geometry_reason": np.array(reasons)[expanded],
    }
    if any(file_sha(Path(p)) != digest for p, digest in source_hashes.items()):
        raise ValueError("Audit implementation changed during extraction")
    args.output.mkdir(parents=True, exist_ok=True)
    save_npz(args.output / "POPULATION.npz", arrays)
    q = arrays["quality"]
    reference = arrays["native"][:, 0]
    drift = {}
    for name, pred in (
        ("RGB_context", arrays["native"][:, 1]),
        ("event_context", arrays["native"][:, 2]),
        ("geometry", arrays["geometry"]),
    ):
        valid = np.isfinite(pred)
        drift[name] = {
            "finite": int(valid.sum()),
            "mean_absolute_difference_seconds": float(
                np.abs(pred[valid] - reference[valid]).mean()
            ),
            "sign_disagreement_with_F0": int(
                np.sum(np.sign(pred[valid]) != np.sign(reference[valid]))
            ),
        }
    atomic_receipt(
        args.output / "COMPLETE.json",
        {
            "schema": "phase_safe_test12_labelblind_audit_v1",
            "status": "COMPLETE",
            "logical_rows": len(logical),
            "physical_rows": len(canonical),
            "sequences": 12,
            "active_history_rows": int(((q[:, 9] > 0) & (q[:, 11] <= 0.15)).sum()),
            "valid_geometry_rows": int(np.isfinite(arrays["geometry"]).sum()),
            "canonical_history_counts": dict(Counter(map(str, history_counts))),
            "geometry_reasons": dict(Counter(arrays["geometry_reason"].tolist())),
            "native_replay": {
                "rows": len(canonical),
                "within_registered_tolerance": True,
                "phase_rtol": 1e-5,
                "phase_atol": 1e-6,
                "ttc_rtol": 1e-4,
                "ttc_atol": 1e-4,
                "max_absolute_phase_difference": float(
                    np.max(np.abs(replay["raw_location"].numpy() - phases))
                ),
                "max_absolute_ttc_difference_seconds": float(
                    np.max(np.abs(replay["ttc"].numpy() - f0))
                ),
                "bitwise_ttc_equal": bool(np.array_equal(replay["ttc"].numpy(), f0)),
            },
            "original_expert_drift_against_F0_NOT_ACCURACY": drift,
            "parent": parent,
            "archives": archive_inputs,
            "sources_sha256": source_hashes,
            "inputs_sha256": TRUSTED_INPUTS_SHA256,
            "preservation_sha256": file_sha(args.preservation),
            "population_sha256": file_sha(args.output / "POPULATION.npz"),
            "feature_domain": "SAVED_TEST12_DEPLOYMENT_PRODUCER_STATES_NOT_OOF",
            "scope": (
                "Label-blind independent Test12 inputs; no targets, accuracy, tuning or selection. "
                "Expand 5805 physical queries to 6762 official logical queries with frozen aliases."
            ),
            "private_targets_read": False,
            "optimizer_updates": 0,
            "new_candidate_evaluated": False,
            "raw_media_read": False,
        },
    )
    print(f"COMPLETE: {len(logical)} label-blind logical queries", flush=True)


def main() -> None:
    """Read explicit archive/inventory locations and write a separate audit cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "inputs", "preservation", "archive-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
