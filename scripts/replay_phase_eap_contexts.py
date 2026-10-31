"""Replay original deployment contexts and head from sealed Test12 diagnostics."""

import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.features import frozen_context_state
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from scripts.final_train40_transfer import HEAD, load_models


def main() -> None:
    """Use batch1 CPU controls only, retaining every numerical mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh replay directory")
    if file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Unexpected Test12 inventory")
    inputs = read(args.inputs)
    if _contains_private_key(inputs):
        raise ValueError("Unexpected private input field")
    queries = {q["sample_token"]: q for q in inputs["queries"]}
    canonical = set(inputs["physical_aliases"].values())
    baseline_path = args.baseline_audit / "COMPLETE.json"
    baseline = read(baseline_path)
    parent = read(args.upstream / "BUNDLE.json")["parent"]
    if (
        baseline["status"] != "COMPLETE"
        or baseline["parent"] != parent
        or baseline["inputs_sha256"] != TRUSTED_INPUTS_SHA256
        or len(canonical) != 5805
    ):
        raise ValueError("Diagnostic baseline ancestry/population differs")
    paths = verified_upstream_paths(args.upstream)
    sources = [
        Path(__file__),
        Path("scripts/final_train40_transfer.py"),
        Path("operational/final_refit/features.py"),
        Path("operational/final_refit/training.py"),
        Path("src/e_jepa_ttc/simplex_t/model.py"),
        Path("src/e_jepa_ttc/simplex_t/phase.py"),
    ]
    freeze = {
        "schema": "phase_original_eap_context_replay_v1",
        "device": "cpu",
        "batch_size": 1,
        "float_rtol": 1e-5,
        "float_atol": 1e-6,
        "ttc_rtol": 1e-4,
        "ttc_atol": 1e-4,
        "categorical": "exact",
        "torch_version": torch.__version__,
        "inputs_sha256": TRUSTED_INPUTS_SHA256,
        "baseline_sha256": file_sha(baseline_path),
        "upstream_sha256": file_sha(args.upstream / "BUNDLE.json"),
        "sources": {str(p): file_sha(p) for p in sources},
        "model_files": {str(p): file_sha(p) for p in paths.values()},
        "archives": baseline["archives"],
        "targets_read": False,
        "candidate_evaluated": False,
        "optimizer_updates": 0,
        "scope": "Original contexts/head from saved inputs; no sensor producer replay",
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models, _ = load_models(paths, "cpu")
    started = time.perf_counter()
    records, fields = {}, {}

    def compare(name: str, actual: torch.Tensor, expected: np.ndarray) -> bool:
        reference = torch.from_numpy(expected)
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError(f"Shape/dtype differs: {name}")
        if actual.is_floating_point():
            rtol, atol = (1e-4, 1e-4) if name.endswith("output_ttc") else (1e-5, 1e-6)
            close = torch.isclose(actual, reference, rtol=rtol, atol=atol)
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise ValueError(f"Nonfinite original replay: {name}")
            maximum = float((actual - reference).abs().max())
        else:
            close = actual == reference
            maximum = float((actual.to(torch.float64) - reference).abs().max())
        row = fields.setdefault(name, {"comparisons": 0, "failed_queries": 0, "max_abs": 0.0})
        row["comparisons"] += 1
        row["failed_queries"] += int(not bool(close.all()))
        row["max_abs"] = max(row["max_abs"], maximum)
        return bool(close.all())

    try:
        for entry in baseline["archives"].values():
            path = Path(entry["path"])
            if file_sha(path) != entry["sha256"]:
                raise ValueError("Diagnostic archive changed")
            with tarfile.open(path) as archive:
                names = archive.getnames()

                def member(name: str) -> bytes:
                    stream = archive.extractfile(name)
                    if stream is None:
                        raise ValueError(f"Missing diagnostic: {name}")
                    return stream.read()

                admission = json.loads(
                    member(next(n for n in names if n.endswith(".admission.json")))
                )
                if (
                    admission["model_seal_sha256"] != parent["model_seal_sha256"]
                    or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
                ):
                    raise ValueError("Archived sequence belongs to another model/input")
                for name in sorted(names):
                    if "/diagnostics/" not in name or not name.endswith(".json"):
                        continue
                    receipt = json.loads(member(name))
                    token = receipt["sample_token"]
                    if token not in canonical or token in records:
                        raise ValueError("Duplicate or unexpected physical query")
                    if (
                        receipt["private_targets_read"]
                        or receipt["fallback_source"] != "PHASE_ONLY"
                        or receipt["run_binding_sha256"] != admission["run_binding_sha256"]
                    ):
                        raise ValueError("Unexpected original diagnostic binding")
                    for key in (
                        "event_history",
                        "rgb_history",
                        "query_time_us",
                        "sequence_id",
                        "track_id",
                    ):
                        if receipt[key] != queries[token][key]:
                            raise ValueError("Archived query identity/history differs")
                    payload = member(name.removesuffix(".json") + ".npz")
                    if hashlib.sha256(payload).hexdigest() != receipt["npz_sha256"]:
                        raise ValueError("Diagnostic array bytes changed")
                    failed = []
                    with (
                        np.load(io.BytesIO(payload), allow_pickle=False) as stored,
                        torch.inference_mode(),
                    ):
                        blocks, widths = {}, {}
                        for stage in ("E_CTX_MATCHED", "R_CTX"):
                            arguments = [
                                torch.from_numpy(stored[f"{stage}.input_{i}"]) for i in range(4)
                            ]
                            blocks[stage], widths[stage], output = frozen_context_state(
                                models[stage], *arguments
                            )
                            for key, value in output.items():
                                label = f"{stage}.output_{key}"
                                if not compare(label, value, stored[label]):
                                    failed.append(label)
                        features = torch.cat((blocks["R_CTX"], blocks["E_CTX_MATCHED"]), -1)
                        for label, value in (
                            (f"{HEAD}.input_0", features),
                            (f"{HEAD}.input_1", widths["R_CTX"]),
                        ):
                            if not compare(label, value, stored[label]):
                                failed.append(label)
                        output = models[HEAD](features, widths["R_CTX"])
                        for key, value in output.items():
                            label = f"{HEAD}.output_{key}"
                            if not compare(label, value, stored[label]):
                                failed.append(label)
                    records[token] = {"failed_fields": failed}
            print(f"{len(records)}/5805 original context/head queries replayed", flush=True)
        if set(records) != canonical:
            raise ValueError("Replay misses physical queries")
        for path, expected in {**freeze["sources"], **freeze["model_files"]}.items():
            if file_sha(Path(path)) != expected:
                raise ValueError("Runtime source/model changed during replay")
        atomic_receipt(args.output / "RESULTS.json", records)
        failures = sum(bool(r["failed_fields"]) for r in records.values())
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "parity_status": "FAIL" if failures else "PASS",
                "physical_queries": len(records),
                "failed_queries": failures,
                "fields": fields,
                "elapsed_seconds": time.perf_counter() - started,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "results_sha256": file_sha(args.output / "RESULTS.json"),
                "targets_read": False,
                "candidate_evaluated": False,
                "producer_calls": 0,
                "scope": freeze["scope"],
            },
        )
        print({"parity_failed_queries": failures}, flush=True)
    except Exception as exc:
        atomic_receipt(args.output / "FAILED.json", {"error": repr(exc), "queries": len(records)})
        raise


if __name__ == "__main__":
    main()
