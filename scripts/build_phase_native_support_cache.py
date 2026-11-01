"""Build target-blind native clock/count tables from already sealed inference states."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_native_support.features import (
    HEAD_INPUT,
    INDICES,
    from_context_arrays,
    from_phase340,
    validate_support,
)
from scripts.build_phase_native_context_augmentations import verify_artifacts
from scripts.train_phase_native_context_fusion import validate_source_closure

CONTEXT_FIELDS = {
    HEAD_INPUT,
    "R_CTX.input_1",
    "R_CTX.input_2",
    "E_CTX_MATCHED.input_1",
    "E_CTX_MATCHED.input_2",
}
ALLOWED_FIELDS = {
    "features",
    "sample_token",
    "group_id",
    "domain",
    "token",
    "group",
    "keep",
    "status",
    "reason",
    "canonical_token",
} | CONTEXT_FIELDS


def read_fields(path: str | Path, fields: list[str]) -> dict[str, np.ndarray]:
    """Materialize only explicit non-target fields, even from supervised containers."""
    if not set(fields) <= ALLOWED_FIELDS:
        raise ValueError("Only explicitly allowed label-blind fields may be read")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in fields}


def exact_indices(source: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Join unique exact tokens and reject missing or ambiguous identities."""
    if (
        source.ndim != 1
        or desired.ndim != 1
        or len(set(source)) != len(source)
        or len(set(desired)) != len(desired)
    ):
        raise ValueError("Exact token join requires unique one-dimensional identities")
    lookup = {str(token): i for i, token in enumerate(source)}
    if any(str(token) not in lookup for token in desired):
        raise ValueError("Required token is missing from source state")
    return np.array([lookup[str(token)] for token in desired], np.int64)


def context_support(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only the head input and context clock/mask tensors from a sealed shard."""
    with np.load(path, allow_pickle=False) as stored:
        fields = sorted(set(stored.files) & CONTEXT_FIELDS)
    return from_context_arrays(read_fields(path, fields))


def table(
    token: np.ndarray,
    group: np.ndarray,
    quality: np.ndarray,
    known: np.ndarray,
    reasons: np.ndarray | None = None,
) -> dict:
    """Construct aligned support and explicit route metadata without predictions or labels."""
    validate_support(quality, known)
    if token.shape != (len(quality),) or group.shape != token.shape:
        raise ValueError("Support table identities are not aligned")
    status = np.where(known.all(1), "fused", np.where(known[:, 1], "event_only", "unavailable"))
    return {
        "token": token.copy(),
        "group": group.copy(),
        "quality10": quality,
        "knownmask2": known,
        "status": status,
        "reason": status.copy() if reasons is None else reasons.copy(),
    }


def historical_slider_support(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the same ten coordinates from original accepted-anchor diagnostics."""
    quality, known = np.zeros((1, 10), np.float32), np.zeros((1, 2), bool)
    if row["runtime"]["status"] != "ok":
        return quality, known  # Prepared RGB count is not an executed RGB context.
    query_us = row["input"]["query_us"]
    for index, modality in enumerate(("rgb", "event")):
        diagnostics = row["runtime"]["producer_diagnostics"][modality]
        anchors = diagnostics["accepted_anchor_us"]
        if not anchors:
            continue
        by_anchor = {item["anchor_us"]: item for item in row["input"][modality + "_observations"]}
        anchor = anchors[-1]
        available = by_anchor[anchor]["available_us"]
        if len(anchors) != row["runtime"][modality + "_history_count"]:
            raise ValueError("Historical accepted context counts differ")
        quality[0, index * 5 : index * 5 + 5] = [
            (query_us - anchor) / 1e6,
            (query_us - available) / 1e6,
            0 if len(anchors) == 1 else (anchor - anchors[-2]) / 1e6,
            (available - anchor) / 1e6,
            len(anchors) / 8,
        ]
        known[0, index] = True
    validate_support(quality, known)
    return quality, known


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "population": "output/phase_submission/training/POPULATION.npz",
        "original-config": "output/phase_submission/training/CONFIG.json",
        "data": "output/phase_adapt_final/data_v3",
        "expanded": "output/phase_observation_cache/expanded149",
        "source-closure": (
            "output/phase_observation_cache/expanded149_sourceclosure/SOURCE_CLOSURE.json"
        ),
        "source-audit": "output/phase_safe_fusion/source_audit",
        "test12-audit": "output/phase_safe_fusion/test12_audit",
        "slider": "output/phase_campaign/Slider500",
        "output": "output/phase_native_support/cache_v1",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable support-cache output directory")
    started = time.perf_counter()
    expanded = read(args.expanded / "COMPLETE.json")
    if (
        expanded.get("status") != "COMPLETE"
        or expanded.get("queries") != 149
        or expanded.get("augmentations") != 447
        or expanded.get("targets_read") is not False
        or expanded.get("clean_exact") is not True
        or expanded.get("disk_roundtrip_exact") is not True
    ):
        raise ValueError("Complete target-blind149/447 extraction required")
    verify_artifacts(args.expanded, expanded["artifacts"])
    if file_sha(args.expanded / "FREEZE.json") != expanded["freeze_sha256"]:
        raise ValueError("Expansion freeze changed")
    validate_source_closure(args.source_closure, args.expanded)
    for path in (args.population, args.original_config):
        if expanded["inputs"].get(str(path)) != file_sha(path):
            raise ValueError("Development population differs from sealed extraction")
    source_receipt = read(args.source_audit / "COMPLETE.json")
    test_receipt = read(args.test12_audit / "COMPLETE.json")
    data_receipt = read(args.data / "COMPLETE.json")
    if source_receipt["cache_receipt"] != data_receipt:
        raise ValueError("TRAIN40 audit uses a different original state cache")
    for directory, receipt in (
        (args.source_audit, source_receipt),
        (args.test12_audit, test_receipt),
    ):
        if receipt["status"] != "COMPLETE" or receipt["population_sha256"] != file_sha(
            directory / "POPULATION.npz"
        ):
            raise ValueError("Source/Test12 audit bytes changed")
    if (
        data_receipt["training_npz_sha256"] != file_sha(args.data / "training.npz")
        or test_receipt["private_targets_read"] is not False
        or test_receipt["feature_domain"] != "SAVED_TEST12_DEPLOYMENT_PRODUCER_STATES_NOT_OOF"
    ):
        raise ValueError("Original states or deployment provenance differ")
    archives = {item["path"]: item["sha256"] for item in test_receipt["archives"].values()}
    for path, digest in archives.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Original Test12 prediction archive changed")
    sources = [
        Path(__file__).resolve(),
        Path("operational/phase_native_support/features.py"),
        Path("tests/unit/test_phase_native_support_cache.py"),
        Path("scripts/build_phase_native_context_augmentations.py"),
        Path("scripts/train_phase_native_context_fusion.py"),
        Path("scripts/audit_phase_native_source_closure.py"),
        Path("operational/phase_observation_cache/integrity.py"),
        Path("operational/final_refit/features.py"),
        Path("operational/rgb_port/infer_experts.py"),
        Path("operational/phase_adapt_cache.py"),
        Path("operational/phase_campaign/data.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-native-support-fusion.md"),
    ]
    source_hashes = {str(p.resolve().relative_to(Path.cwd())): file_sha(p) for p in sources}
    inputs = [
        args.population,
        args.original_config,
        args.data / "training.npz",
        args.data / "COMPLETE.json",
        args.data / "TRAIN40_RESTORE.json",
        args.source_audit / "POPULATION.npz",
        args.source_audit / "COMPLETE.json",
        args.test12_audit / "POPULATION.npz",
        args.test12_audit / "COMPLETE.json",
        args.expanded / "COMPLETE.json",
        args.expanded / "FREEZE.json",
        args.expanded / "AUGMENTATIONS.npz",
        args.source_closure,
        args.source_closure.parent / "SNAPSHOT.json",
        args.slider / "DIAGNOSTICS.json",
        args.slider / "PREDICTIONS_SEALED.json",
        args.slider / "FREEZE.json",
    ]
    freeze = {
        "schema": "phase_native_support_cache_v1",
        "targets_read": False,
        "optimizer_updates": 0,
        "feature_indices": list(INDICES),
        "inputs": {str(p): file_sha(p) for p in inputs},
        "source_hashes": source_hashes,
        "test12_archive_sha256": archives,
        "clock_order": [
            "query_minus_anchor_s",
            "query_minus_available_s",
            "previous_anchor_interval_s",
            "available_minus_anchor_s",
            "valid_count_div8",
        ],
        "modality_order": ["RGB", "event"],
        "missing_semantics": (
            "Zeros are placeholders; knownmask2=false means no executed context, "
            "not measured zero clocks"
        ),
        "geometry_quality": "Not read or modified; join unchanged original values by token",
        "scopes": {
            "development": "Original road deployment states and matched fresh Slider contexts",
            "restarts": (
                "Actual retained-context replay; diagnostic only, not training augmentation"
            ),
            "source_train40": (
                "TRAIN40 active audit; five-fold physical OOF upstream states, not deployment"
            ),
            "test12": "6762 logical queries /5805 physical saved deployment states; label blind",
        },
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for relative in source_hashes:
        destination = args.output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)

    # Only the explicitly named identity/state fields are ever materialized.
    dev = read_fields(args.population, ["token", "group"])
    config = read(args.original_config)
    if (
        len(dev["token"]) != 2608
        or len(set(dev["token"])) != 2608
        or sorted(set(dev["group"])) != config["fit_groups"]
        or set(dev["group"]) & set(config["excluded_road"] + config["excluded_slider"])
    ):
        raise ValueError("Eligible development identity changed")
    original = read_fields(
        args.data / "training.npz", ["features", "sample_token", "group_id", "domain"]
    )
    original_q, original_known = from_phase340(original["features"])
    road = dev["group"] != "Slider-500"
    road_indices = exact_indices(original["sample_token"], dev["token"][road])
    if not np.array_equal(original["group_id"][road_indices], dev["group"][road]) or not np.all(
        original["domain"][road_indices] == "evttc"
    ):
        raise ValueError("Road group or domain mapping differs")
    dq, dk = np.zeros((2608, 10), np.float32), np.zeros((2608, 2), bool)
    dq[road], dk[road] = original_q[road_indices], original_known[road_indices]
    dev_reasons = np.full(2608, "fused", dtype="U96")
    dev_lookup = {str(t): i for i, t in enumerate(dev["token"])}
    restart = read_fields(
        args.expanded / "AUGMENTATIONS.npz", ["token", "group", "keep", "status", "reason"]
    )
    rq, rk = np.zeros((447, 10), np.float32), np.zeros((447, 2), bool)
    if (
        len(restart["token"]) != 447
        or len(set(zip(restart["token"], restart["keep"], strict=True))) != 447
    ):
        raise ValueError("Restart query/keep identities differ")
    diagnostic_rows = read(args.slider / "DIAGNOSTICS.json")["rows"]
    historical = {f"Slider-500:{row['query']}": row for row in diagnostic_rows}
    if set(historical) != {f"Slider-500:{i}" for i in range(77)}:
        raise ValueError("Historical Slider identities differ")
    parity = {
        "road_reextracted_exact": 0,
        "slider_historical_available_exact": 0,
        "slider_unavailable_explicit": 0,
        "restart_rows": 0,
    }
    for token in dict.fromkeys(restart["token"]):
        token = str(token)
        directory = args.expanded / "queries" / hashlib.sha256(token.encode()).hexdigest()[:24]
        receipt = read(directory / "COMPLETE.json")
        i = dev_lookup[token]
        if receipt["binding"]["token"] != token or receipt["binding"]["group"] != str(
            dev["group"][i]
        ):
            raise ValueError("Per-query acquisition identity differs")
        clean_q, clean_k = context_support(directory / "CLEAN_REPLAY.npz")
        if road[i]:
            np.testing.assert_array_equal(dq[i], clean_q[0])
            np.testing.assert_array_equal(dk[i], clean_k[0])
            parity["road_reextracted_exact"] += 1
        else:
            old_q, old_k = historical_slider_support(historical[token])
            np.testing.assert_array_equal(clean_q, old_q)
            np.testing.assert_array_equal(clean_k, old_k)
            dq[i], dk[i] = clean_q[0], clean_k[0]
            dev_reasons[i] = receipt["reason"]
            parity[
                "slider_historical_available_exact"
                if clean_k.all()
                else "slider_unavailable_explicit"
            ] += 1
        for j in np.flatnonzero(restart["token"] == token):
            if restart["group"][j] != dev["group"][i]:
                raise ValueError("Restart group differs from original query")
            if receipt["status"] != "unavailable":
                sq, sk = context_support(directory / f"LAST_{int(restart['keep'][j])}.npz")
                rq[j], rk[j] = sq[0], sk[0]
            parity["restart_rows"] += 1
    if parity != {
        "road_reextracted_exact": 72,
        "slider_historical_available_exact": 76,
        "slider_unavailable_explicit": 1,
        "restart_rows": 447,
    }:
        raise ValueError("Incomplete original/fresh support parity")
    development = table(dev["token"], dev["group"], dq, dk, dev_reasons)
    restarts = table(restart["token"], restart["group"], rq, rk, restart["reason"])
    restarts["keep"] = restart["keep"].copy()
    if not np.array_equal(restarts["status"], restart["status"]):
        raise ValueError("Restart executed-context status differs")
    source = read_fields(args.source_audit / "POPULATION.npz", ["token", "group"])
    source_indices = exact_indices(original["sample_token"], source["token"])
    if (
        len(source_indices) != 88744
        or not np.array_equal(source["group"], original["group_id"][source_indices])
        or not np.all(original["domain"][source_indices] == "eap")
    ):
        raise ValueError("TRAIN40 OOF token/group/domain mapping differs")
    source_table = table(
        source["token"], source["group"], original_q[source_indices], original_known[source_indices]
    )
    test = read_fields(
        args.test12_audit / "POPULATION.npz", ["token", "group", "features", "canonical_token"]
    )
    if (
        len(test["token"]) != 6762
        or len(set(test["token"])) != 6762
        or len(set(test["canonical_token"])) != 5805
    ):
        raise ValueError("Test12 logical/physical alias identities differ")
    tq, tk = from_phase340(test["features"])
    test_table = table(test["token"], test["group"], tq, tk)
    tables = {
        "DEVELOPMENT": development,
        "RESTARTS": restarts,
        "SOURCE_TRAIN40": source_table,
        "TEST12": test_table,
    }
    for name, values in tables.items():
        save_npz(args.output / f"{name}.npz", values)
    for path, digest in {**freeze["inputs"], **source_hashes}.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Frozen source or input changed during cache construction")
    atomic_receipt(
        args.output / "COMPLETE.json",
        {
            "schema": freeze["schema"],
            "status": "COMPLETE",
            "targets_read": False,
            "optimizer_updates": 0,
            "producer_calls": 0,
            "feature_indices": list(INDICES),
            "counts": {
                "DEVELOPMENT": 2608,
                "RESTARTS": 447,
                "SOURCE_TRAIN40": 88744,
                "TEST12": 6762,
            },
            "parity": parity,
            "inputs": freeze["inputs"],
            "freeze_sha256": file_sha(args.output / "FREEZE.json"),
            "source_closure_sha256": file_sha(args.source_closure),
            "elapsed_seconds": time.perf_counter() - started,
            "known_context_counts": {
                name: values["knownmask2"].sum(0).tolist() for name, values in tables.items()
            },
            "artifacts": {
                str(p.relative_to(args.output)): file_sha(p)
                for p in args.output.rglob("*")
                if p.is_file()
            },
        },
    )
    print(
        "COMPLETE support2608/restart447/TRAIN40OOF88744/Test12deployment6762; "
        "no targets or producer calls",
        flush=True,
    )


if __name__ == "__main__":
    main()
