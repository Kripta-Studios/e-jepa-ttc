"""Audit frozen canonical PAIR on sealed current-query eAP inputs without labels.

This reports output drift, coverage and emission policy; it does not reconstruct
histories or evaluate context/head accuracy or eAP preservation.
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair, CanonicalPhaseDomainError
from scripts.final_train40_transfer import load_models

SCOPE = "Current-query PAIR only; no context/head accuracy or eAP preservation claim"
BATCH_SIZE = 512


def output_policy(raw_phase: np.ndarray) -> dict[str, int]:
    """Count the exact FP32 inverse floor and raw-phase policy boundaries."""
    finite = np.isfinite(raw_phase)
    raw = raw_phase[finite]
    inverse = -torch.expm1(-torch.from_numpy(raw).clamp(MIN_PHASE, MAX_PHASE)) / 0.1
    floor = inverse.abs() <= 1 / 60
    return {
        "finite_rows": int(finite.sum()),
        "nonfinite_raw_phase": int((~finite).sum()),
        "raw_zero_phase": int((raw == 0).sum()),
        "raw_negative_zero_phase": int(((raw == 0) & np.signbit(raw)).sum()),
        "raw_negative_phase": int((raw < 0).sum()),
        "raw_phase_below_min": int((raw < MIN_PHASE).sum()),
        "raw_phase_above_max": int((raw > MAX_PHASE).sum()),
        "inverse_magnitude_floor_rows": int(floor.sum()),
        "positive_floor": int((floor & (inverse >= 0)).sum()),
        "negative_floor": int((floor & (inverse < 0)).sum()),
    }


def verify_inputs(inputs: Path, upstream: Path) -> tuple[dict, dict[str, Path], dict]:
    """Verify extraction, historical wrapper/source seals and deployment ancestry."""
    complete = read(inputs / "INPUTS_COMPLETE.json")
    original = read(inputs / "INPUT_FREEZE.json")
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("targets_read") is not False
        or original.get("targets_read") is not False
        or complete["freeze_sha256"] != file_sha(inputs / "INPUT_FREEZE.json")
        or original["source_sha256"] != file_sha(inputs / "extract_inputs.py")
        or original["upstream_bundle_sha256"] != file_sha(upstream / "BUNDLE.json")
    ):
        raise ValueError("Input extraction seal, source or model identity differs")
    expected = {"EVENT_INPUTS.npz", "RGB_INPUTS.npz"}
    if set(complete["artifacts"]) != expected:
        raise ValueError("Input artifact inventory differs")
    for name, digest in complete["artifacts"].items():
        if file_sha(inputs / name) != digest:
            raise ValueError(f"Changed input cache: {name}")
    historical = read(inputs / "RUN_FREEZE.json")
    historical_complete = read(inputs / "RUN_COMPLETE.json")
    if historical_complete["freeze_sha256"] != file_sha(inputs / "RUN_FREEZE.json") or historical[
        "inputs_complete_sha256"
    ] != file_sha(inputs / "INPUTS_COMPLETE.json"):
        raise ValueError("Historical wrapper run seal differs")
    for name, digest in historical["source_hashes"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f"Historical numerical source changed: {name}")
    wrapper_path = "operational/phase_timewarp/canonical_pair.py"
    if not any(Path(name) == Path(wrapper_path) for name in historical["source_hashes"]):
        raise ValueError("Historical source seal lacks canonical wrapper")
    paths = verified_upstream_paths(upstream)
    for modality, parent in (("event", "E_A5"), ("rgb", "R_A5")):
        info = complete["modalities"][modality]
        receipt = info["receipt"]
        if (
            receipt["parents"][parent] != file_sha(paths[parent])
            or receipt["output_sha256"] != info["member_sha256"]
            or receipt["targets_used_for_feature_construction"] is not False
            or receipt["rows"] != 88_744
        ):
            raise ValueError("Current-query source provenance differs from deployment")
    return complete, paths, historical


def audit_modality(
    model: torch.nn.Module, features: np.ndarray, tokens: np.ndarray
) -> tuple[dict, dict]:
    """Execute paired full batches and retain failed-domain traces without fabricated outputs."""
    wrapper = CanonicalPair(model).eval()
    chunks: list[dict[str, np.ndarray]] = []
    rejected: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(features), BATCH_SIZE):
            batch = PairFeatureBatch(torch.from_numpy(features[start : start + BATCH_SIZE]))
            try:
                wrapper(batch)
            except CanonicalPhaseDomainError as error:
                rejected.append(
                    {
                        "start": start,
                        "stop": min(start + BATCH_SIZE, len(features)),
                        "reason": str(error),
                    }
                )
            trace = {k: v.cpu().numpy() for k, v in wrapper.last_trace.items()}
            baseline = emitted_phase(torch.from_numpy(trace["original_raw_phase"]))
            valid = trace["domain_valid"]
            corrected = np.full(len(baseline), np.nan, np.float32)
            ttc = np.full(len(baseline), np.nan, np.float32)
            if valid.any():
                emitted = emitted_phase(torch.from_numpy(trace["raw_corrected_phase"][valid]))
                corrected[valid] = emitted.numpy()
                ttc[valid] = phase_to_ttc(emitted).numpy()
            trace.update(
                baseline_emitted_phase=baseline.numpy(),
                baseline_ttc=phase_to_ttc(baseline).numpy(),
                corrected_emitted_phase=corrected,
                corrected_ttc=ttc,
            )
            chunks.append(trace)
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    arrays["token"] = tokens
    identity, valid = arrays["identity_mask"], arrays["domain_valid"]
    exact = {}
    for label, new, old in (
        ("raw", "raw_corrected_phase", "original_raw_phase"),
        ("emitted", "corrected_emitted_phase", "baseline_emitted_phase"),
        ("ttc", "corrected_ttc", "baseline_ttc"),
    ):
        exact[label + "_identity_exact"] = bool(
            np.array_equal(
                arrays[new][identity].view(np.uint32), arrays[old][identity].view(np.uint32)
            )
        )
    if not all(exact.values()):
        raise AssertionError("Canonical 100 ms identity changed output bytes")
    difference = np.abs(arrays["corrected_ttc"][valid] - arrays["baseline_ttc"][valid])
    report = {
        "rows": len(tokens),
        "identity_rows": int(identity.sum()),
        "domain_valid_rows": int(valid.sum()),
        "rejected_batches": rejected,
        **exact,
        "ttc_unchanged_rows": int((arrays["corrected_ttc"] == arrays["baseline_ttc"]).sum()),
        "sign_changes": int(
            (
                np.signbit(arrays["corrected_ttc"][valid])
                != np.signbit(arrays["baseline_ttc"][valid])
            ).sum()
        ),
        "ttc_abs_drift_mean": float(difference.mean()) if len(difference) else None,
        "ttc_abs_drift_max": float(difference.max()) if len(difference) else None,
        "ttc_abs_drift_quantiles": np.quantile(difference, [0, 0.5, 0.9, 0.99, 1]).tolist()
        if len(difference)
        else None,
        "output_policy": {
            "baseline": output_policy(arrays["original_raw_phase"]),
            "corrected": output_policy(arrays["raw_corrected_phase"]),
        },
        "domain_valid_is_row_diagnostic": True,
        "batch_rejection_policy": (
            "A failed wrapper call rejects the full original batch; "
            "finite row diagnostics do not override this"
        ),
    }
    return arrays, report


def main() -> None:
    """Run the sealed current-query audit in a fresh output directory on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("output/phase_timewarp/eap_current_pair_audit"),
        help="Sealed extraction and original wrapper-run receipts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh evidence directory; existing nonempty directories are rejected",
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("output/phase_submission/FINAL_MODEL/upstream"),
        help="Verified frozen deployment bundle",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="CPU-only diagnostic; no producer extraction",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Audit output must be fresh; existing evidence is immutable")
    input_receipt, paths, historical = verify_inputs(args.inputs, args.upstream)
    sources = {Path(name) for name in historical["source_hashes"]} | {
        Path(__file__),
        Path("tests/unit/test_phase_canonical_pair_source_audit.py"),
    }
    sources = {p.resolve() for p in sources}
    hashes = {str(p.relative_to(Path.cwd())): file_sha(p) for p in sorted(sources)}
    freeze = {
        "source_hashes": hashes,
        "inputs": str(args.inputs),
        "input_artifacts": input_receipt["artifacts"],
        "inputs_complete_sha256": file_sha(args.inputs / "INPUTS_COMPLETE.json"),
        "historical_run_freeze_sha256": file_sha(args.inputs / "RUN_FREEZE.json"),
        "upstream_bundle_sha256": file_sha(args.upstream / "BUNDLE.json"),
        "model_components": {key: file_sha(path) for key, path in paths.items()},
        "device": args.device,
        "threads": 2,
        "batch_size": BATCH_SIZE,
        "scope": SCOPE,
        "targets_read": False,
        "optimizer_updates": 0,
        "floor_predicate": "abs(-expm1(-clamp(raw_phase,MIN_PHASE,MAX_PHASE))/0.1)<=FP32(1/60)",
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for name in hashes:
        destination = args.output / "executed_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    torch.set_num_threads(2)
    try:
        models, _ = load_models(paths, args.device)
        reports = {}
        started = time.perf_counter()
        for modality, model_name in (("event", "PAIR_E_MATCHED"), ("rgb", "PAIR_R")):
            with np.load(
                args.inputs / (modality.upper() + "_INPUTS.npz"), allow_pickle=False
            ) as stored:
                features, tokens = stored["features"], stored["token"]
            if (
                features.shape != (88_744, 133)
                or features.dtype != np.float32
                or not np.isfinite(features).all()
                or tokens.shape != (88_744,)
            ):
                raise ValueError("Invalid current-query PAIR133 cache")
            arrays, reports[modality] = audit_modality(models[model_name], features, tokens)
            save_npz(args.output / (modality.upper() + "_CANONICAL.npz"), arrays)
        for name, digest in hashes.items():
            if file_sha(Path(name)) != digest:
                raise ValueError(f"Execution source changed: {name}")
        for name, digest in input_receipt["artifacts"].items():
            if file_sha(args.inputs / name) != digest:
                raise ValueError("Input cache changed during audit")
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "targets_read": False,
                "optimizer_updates": 0,
                "scope": SCOPE,
                "elapsed_seconds": time.perf_counter() - started,
                "modalities": reports,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "artifacts": {
                    str(p.relative_to(args.output)): file_sha(p)
                    for p in args.output.rglob("*")
                    if p.is_file()
                },
            },
        )
    except Exception as error:
        atomic_receipt(
            args.output / "FAILED.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "targets_read": False},
        )
        raise
    print("COMPLETE", args.output / "COMPLETE.json", flush=True)


if __name__ == "__main__":
    main()
