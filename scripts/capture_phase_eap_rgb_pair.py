"""Recapture original Test12 RGB PAIR133, admitting no candidate or target access."""

import argparse
import hashlib
import io
import json
import platform
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader, decode_query
from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.compact_capture import capture_compact
from operational.rgb_port import transfer as engine
from scripts.final_train40_transfer import load_models


def select_observations(inputs: dict, mode: str) -> list[str]:
    """Fix pilot endpoints by sequence/time without consulting predictions or targets."""
    if mode not in {"pilot", "full"}:
        raise ValueError("Unknown capture mode")
    physical = set(inputs["physical_aliases"].values())
    queries = [q for q in inputs["queries"] if q["sample_token"] in physical]
    if len({q["sample_token"] for q in queries}) != len(queries):
        raise ValueError("Duplicate physical query")
    if mode == "pilot":
        selected = []
        for sequence in sorted({q["sequence_id"] for q in queries}):
            rows = sorted(
                (q for q in queries if q["sequence_id"] == sequence),
                key=lambda q: (q["query_time_us"], q["sample_token"]),
            )
            selected.extend((rows[0], rows[-1]))
        identities = {q["rgb_history"][-1] for q in selected}
    else:
        identities = {i for q in queries for i in q["rgb_history"]}
    return sorted(identities)


def reference_observations(baseline: dict, inputs: dict, selected: list[str]) -> dict:
    """Authenticate historical diagnostics and collect original RGB raw17/expert3."""
    wanted = set(selected)
    result = {}
    queries = {q["sample_token"]: q for q in inputs["queries"]}
    for entry in baseline["archives"].values():
        path = Path(entry["path"])
        if file_sha(path) != entry["sha256"]:
            raise ValueError("Diagnostic archive changed")
        with tarfile.open(path) as archive:
            names = archive.getnames()

            def member(name: str) -> bytes:
                stream = archive.extractfile(name)
                if stream is None:
                    raise ValueError("Missing diagnostic member")
                return stream.read()

            admission = json.loads(member(next(n for n in names if n.endswith(".admission.json"))))
            if (
                admission["model_seal_sha256"] != baseline["parent"]["model_seal_sha256"]
                or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
            ):
                raise ValueError("Archived model/input ancestry differs")
            for name in names:
                if "/diagnostics/" not in name or not name.endswith(".json"):
                    continue
                receipt = json.loads(member(name))
                query = queries[receipt["sample_token"]]
                if (
                    receipt["private_targets_read"]
                    or receipt["run_binding_sha256"] != admission["run_binding_sha256"]
                    or receipt["rgb_history"] != query["rgb_history"]
                ):
                    raise ValueError("Diagnostic history/binding differs")
                if not wanted.intersection(query["rgb_history"]):
                    continue
                payload = member(name.removesuffix(".json") + ".npz")
                if hashlib.sha256(payload).hexdigest() != receipt["npz_sha256"]:
                    raise ValueError("Diagnostic array hash differs")
                with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                    for i, identity in enumerate(arrays["rgb_new_ids"]):
                        identity = str(identity)
                        if identity not in wanted:
                            continue
                        values = {
                            "raw17": arrays["rgb_new_raw"][i : i + 1].copy(),
                            "expert3": arrays["rgb_new_experts"][i : i + 1].copy(),
                        }
                        if identity in result:
                            for key in values:
                                np.testing.assert_array_equal(result[identity][key], values[key])
                        result[identity] = values
    if set(result) != wanted:
        raise ValueError("Missing original RGB observations")
    return result


def parity(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Use predeclared FP32 producer tolerances; never relax them after a failure."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Original array shape/dtype differs")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite original producer output")
    return {
        "pass": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs": float(np.max(np.abs(actual - expected))),
    }


def main() -> None:
    """Seal original-only capture and retain all parity failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--population-freeze", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Require fresh output and trusted Test12 inventory")
    inputs = read(args.inputs)
    if _contains_private_key(inputs):
        raise ValueError("Unexpected private input field")
    baseline = read(args.baseline_audit / "COMPLETE.json")
    if (
        baseline["status"] != "COMPLETE"
        or baseline["inputs_sha256"] != TRUSTED_INPUTS_SHA256
        or baseline["parent"] != read(args.upstream / "BUNDLE.json")["parent"]
    ):
        raise ValueError("Baseline ancestry differs from delivered upstream")
    if args.mode == "full":
        if args.pilot is None:
            raise ValueError("Full recapture requires the original-only pilot")
        pilot = read(args.pilot / "COMPLETE.json")
        if (
            pilot["parity_status"] != "PASS"
            or pilot["mode"] != "pilot"
            or pilot["script_sha256"] != file_sha(Path(__file__))
            or pilot["freeze_sha256"] != file_sha(args.pilot / "FREEZE.json")
            or pilot["results_sha256"] != file_sha(args.pilot / "RESULTS.json")
        ):
            raise ValueError("Pilot is failed, altered, or from different code")
    paths = verified_upstream_paths(args.upstream)
    selected = select_observations(inputs, args.mode)
    if len(selected) != (24 if args.mode == "pilot" else 5805):
        raise ValueError("Unexpected fixed selection cardinality")
    sources = {
        **read(args.population_freeze)["source_hashes"],
        **{str(p): file_sha(p) for p in (Path(__file__), Path("src/e_jepa_ttc/rgb_port/data.py"))},
    }
    for path, digest in sources.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Runtime differs from population source freeze")
    configure_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    freeze = {
        "schema": "phase_original_eap_rgb_capture_v1",
        "mode": args.mode,
        "selected": selected,
        "selection": "pilot:first/last physical query per sequence; full:all RGB histories",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "batch_size": 1,
        "tf32": False,
        "rtol": 1e-5,
        "atol": 1e-6,
        "sources": sources,
        "models": {str(p): file_sha(p) for p in paths.values()},
        "inputs_sha256": TRUSTED_INPUTS_SHA256,
        "baseline_sha256": file_sha(args.baseline_audit / "COMPLETE.json"),
        "archives": baseline["archives"],
        "targets_read": False,
        "candidate_evaluated": False,
        "optimizer_updates": 0,
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    started = time.perf_counter()
    results = {}
    try:
        reference = reference_observations(baseline, inputs, selected)
        models, _ = load_models(paths, "cuda")
        reader = TarFrameReader(args.media_root, max_open=2, max_frame_cache_bytes=128 * 1024**2)
        verified_shards = {}
        try:
            for index, identity in enumerate(selected):
                item = inputs["rgb_observations"][identity]
                for relative in set(item["producer_shards"]):
                    if relative not in verified_shards:
                        path = (args.media_root / relative).resolve()
                        if not path.is_relative_to(args.media_root.resolve()):
                            raise ValueError("RGB shard escapes media root")
                        stat = path.stat()
                        pin = inputs["rgb_sources"][relative]
                        if stat.st_size != pin["size_bytes"] or file_sha(path) != pin["sha256"]:
                            raise ValueError("RGB shard differs from trusted inputs")
                        verified_shards[relative] = (stat.st_size, stat.st_mtime_ns)
                decoded = decode_query(item, eap_root=args.media_root, reader=reader)
                with capture_compact(models) as (calls, arrays), torch.inference_mode():
                    engine._expert_block(
                        models,
                        "R",
                        torch.from_numpy(decoded["rgb"])[None].to("cuda"),
                        torch.from_numpy(decoded["delta_t_s"])[None].to("cuda"),
                    )
                if len(calls) != 1 or calls[0]["status"] != "accepted":
                    raise ValueError("Expected one unchanged batch1 RGB observation")
                checks = {
                    key: parity(arrays[f"call0.{key}"], reference[identity][key])
                    for key in ("raw17", "expert3")
                }
                target = args.output / "observations" / f"{identity}.npz"
                save_npz(target, arrays)
                results[identity] = {"checks": checks, "sha256": file_sha(target)}
                if (index + 1) % 20 == 0 or index + 1 == len(selected):
                    elapsed = time.perf_counter() - started
                    print(
                        f"{index + 1}/{len(selected)} original RGB elapsed={elapsed:.1f}s",
                        flush=True,
                    )
        finally:
            reader.close()
        for relative, expected in verified_shards.items():
            stat = (args.media_root / relative).stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError("RGB shard changed during inference")
        for path, digest in {**sources, **freeze["models"]}.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Source/model changed during capture")
        atomic_receipt(args.output / "RESULTS.json", results)
        failed = sum(not all(c["pass"] for c in row["checks"].values()) for row in results.values())
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "mode": args.mode,
                "parity_status": "FAIL" if failed else "PASS",
                "failed_observations": failed,
                "observations": len(results),
                "elapsed_seconds": time.perf_counter() - started,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "results_sha256": file_sha(args.output / "RESULTS.json"),
                "script_sha256": file_sha(Path(__file__)),
                "targets_read": False,
                "candidate_evaluated": False,
                "optimizer_updates": 0,
            },
        )
        print({"failed_original_observations": failed}, flush=True)
    except Exception as exc:
        atomic_receipt(
            args.output / "FAILED.json", {"error": repr(exc), "observations": len(results)}
        )
        raise


if __name__ == "__main__":
    main()
