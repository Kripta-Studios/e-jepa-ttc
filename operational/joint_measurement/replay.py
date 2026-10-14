"""Replay included V18 predictions and portable weights from an independent extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bundle import MANIFEST, safe_name
from .verify_predictions import recalculate


def file_hash(path: Path) -> str:
    """Stream file bytes without depending on a campaign's external source paths."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_extraction(root: Path) -> dict:
    """Verify the extracted bytes and reject files escaping the extraction root."""
    root = root.resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest["schema"] != "v18_essential_bundle_v1":
        raise ValueError("Unknown bundle schema")
    expected = manifest["files"]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected) | {MANIFEST}:
        raise ValueError("Extracted file set differs from the bundle manifest; run Python with -B")
    for name, receipt in expected.items():
        path = root / safe_name(name)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Extracted file escapes the bundle root")
        if {"bytes": path.stat().st_size, "sha256": file_hash(path)} != receipt:
            raise ValueError(f"Extracted file differs: {name}")
    return {"status": "PASS", "payload_members": len(expected)}


def replay_predictions(folder: Path) -> dict:
    """Recalculate explicit scalar fields using NumPy, without the production scorer."""
    receipt = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(folder / "predictions.parquet")
    if (
        receipt["status"] != "COMPLETE"
        or file_hash(folder / "predictions.parquet") != receipt["sha256"]
        or len(frame) != receipt["rows"]
        or frame.sample_token.duplicated().any()
    ):
        raise ValueError("Incomplete or altered prediction population")
    original = json.loads((folder / "METRICS.json").read_text(encoding="utf-8"))
    expected = recalculate(frame)
    differences = {}
    for key, value in expected.items():
        found = original
        for part in key.split("."):
            found = found[part]
        if value is None or found is None:
            if value is not None or found is not None:
                raise ValueError(f"Defined metric status changed: {key}")
            differences[key] = None
        elif isinstance(value, (int, bool)):
            if value != found:
                raise ValueError(f"Metric count/flag changed: {key}")
            differences[key] = 0.0
        else:
            if not np.isclose(value, found, atol=1e-9, rtol=1e-10):
                raise ValueError(f"Metric scalar changed: {key}")
            differences[key] = abs(float(value) - float(found))
    return {"status": "PASS", "rows": len(frame), "absolute_differences": differences}


def replay_export(folder: Path) -> dict:
    """Load actual portable bytes on CPU and replay the included label-free example."""
    receipt = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
    for name, digest in receipt["files"].items():
        if file_hash(folder / safe_name(name)) != digest:
            raise ValueError("Portable artifact differs from its receipt")
    binding = json.loads((folder / "BINDING.json").read_text(encoding="utf-8"))
    if "input_names" in binding:
        if (
            receipt["status"] != "PASS"
            or binding["input_names"]
            != ["rgb_features", "event_features", "rgb_ttc", "event_ttc", "rgb_available"]
            or binding.get("includes_encoders_or_sensor_preprocessing") is not False
            or set(receipt["files"])
            != {
                "BINDING.json",
                "model.torchscript.pt",
                "example_inputs.npz",
                "example_outputs.json",
            }
        ):
            raise ValueError("Unknown or incomplete portable fusion contract")
        # The standalone verifier must import no project Python models. Its local
        # replay receipt belongs to a disposable copy, never to the extracted bundle.
        with tempfile.TemporaryDirectory(prefix="v18-fusion-replay-") as temporary:
            copied = Path(temporary)
            for name in (*receipt["files"], "COMPLETE.json"):
                shutil.copyfile(folder / name, copied / name)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("replay_fusion_export.py").resolve()),
                    str(copied),
                ],
                cwd=copied,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise ValueError(f"Independent fusion replay failed: {completed.stderr[-4000:]}")
            return json.loads((copied / "REPLAY.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE":
        raise ValueError("Portable export is incomplete")
    with (folder / "model.torchscript.pt").open("rb") as stream:
        model = torch.jit.load(stream, map_location="cpu").eval()
    with np.load(folder / "example_input.npz", allow_pickle=False) as archive:
        pixel_key = binding.get("pixel_input_key", "events")
        if pixel_key not in {"events", "rgb"}:
            raise ValueError("Unknown portable pixel input contract")
        inputs = (torch.from_numpy(archive[pixel_key]), torch.from_numpy(archive["delta_t_s"]))
    expected = json.loads((folder / "example_output.json").read_text(encoding="utf-8"))[
        "signed_ttc_seconds"
    ]
    with torch.inference_mode():
        output = model(*inputs).numpy()
    if not np.isfinite(output).all() or not np.isfinite(expected).all():
        raise ValueError("Portable example contains nonfinite predictions")
    np.testing.assert_allclose(
        output, expected, atol=binding["atol_seconds"], rtol=binding["rtol"], equal_nan=False
    )
    return {
        "status": "PASS",
        "maxabs_seconds": float(np.max(np.abs(output - np.asarray(expected)))),
        "input_example_count": len(output),
        "rtol": binding["rtol"],
        "atol_seconds": binding["atol_seconds"],
        "bit_exact": bool(np.array_equal(output.astype(np.float64), np.asarray(expected))),
    }


def replay_encoder_change(initial_path: Path, trained_path: Path, witness_path: Path) -> dict:
    """Recompute the retained real encoder-weight change without a dataset or optimizer."""
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    if (
        witness["status"] != "PASS"
        or file_hash(initial_path) != witness["initial_checkpoint_sha256"]
        or file_hash(trained_path) != witness["trained_checkpoint_sha256"]
    ):
        raise ValueError("Encoder witness checkpoint bytes differ")
    initial, trained = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (initial_path, trained_path)
    ]
    if (
        initial["identity"] != trained["identity"]
        or initial["identity_sha256"] != trained["identity_sha256"]
        or trained["identity_sha256"] != witness["identity_sha256"]
        or [initial["completed_updates"], trained["completed_updates"]]
        != witness["updates_compared"]
        or initial["completed_updates"] != 0
        or trained["completed_updates"] <= 0
    ):
        raise ValueError("Encoder witness scientific identity or update interval differs")
    a, b = initial["model_state_dict"], trained["model_state_dict"]
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("Encoder witness model topology differs")
    if not all(torch.isfinite(value).all() for value in b.values()):
        raise ValueError("Encoder witness contains nonfinite trained weights")
    deltas = {key: float((b[key] - a[key]).abs().max()) for key in a if key.startswith("encoder.")}
    if (
        not deltas
        or deltas != witness["per_tensor_maxabs"]
        or max(deltas.values()) != witness["encoder_maxabs"]
        or len(deltas) != witness["encoder_tensors_checked"]
        or sum(value > 0 for value in deltas.values()) != witness["encoder_tensors_changed"]
        or not all(value > 0 for value in deltas.values())
    ):
        raise ValueError("Encoder change claim does not match retained weights")
    updates = [row["update"] for row in trained["cursor"]["curves"]]
    if updates != list(range(1, trained["completed_updates"] + 1)):
        raise ValueError("Encoder witness curve has a missing or duplicated update")
    return {
        "status": "PASS",
        "updates_compared": witness["updates_compared"],
        "encoder_tensors_changed": len(deltas),
        "encoder_maxabs": max(deltas.values()),
        "optimizer_updates": 0,
        "generalization_claim": False,
    }


def replay_frozen_encoder(initial_path: Path, trained_path: Path, witness_path: Path) -> dict:
    """Verify unchanged encoder bytes and learned readout from retained CPU weights."""
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    if (
        witness["status"] != "PASS"
        or file_hash(initial_path) != witness["initial_checkpoint_sha256"]
        or file_hash(trained_path) != witness["trained_checkpoint_sha256"]
    ):
        raise ValueError("Frozen encoder witness checkpoint bytes differ")
    initial, trained = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (initial_path, trained_path)
    ]
    updates = trained["completed_updates"]
    if (
        initial["identity"] != trained["identity"]
        or initial["identity_sha256"] != trained["identity_sha256"]
        or trained["identity_sha256"] != witness["identity_sha256"]
        or initial["completed_updates"] != 0
        or updates <= 0
        or updates != witness["completed_updates"]
    ):
        raise ValueError("Frozen encoder witness identity or update interval differs")
    before, after = initial["model_state_dict"], trained["model_state_dict"]
    if before.keys() != after.keys() or any(
        before[k].shape != after[k].shape or before[k].dtype != after[k].dtype for k in before
    ):
        raise ValueError("Frozen encoder witness model topology differs")
    if not all(torch.isfinite(value).all() for value in [*before.values(), *after.values()]):
        raise ValueError("Frozen encoder witness contains nonfinite weights")
    changed = {
        key
        for key in before
        if not torch.equal(
            before[key].contiguous().reshape(-1).view(torch.uint8),
            after[key].contiguous().reshape(-1).view(torch.uint8),
        )
    }
    encoder = {key for key in before if key.startswith("encoder.")}
    if (
        not encoder
        or encoder & changed
        or len(encoder) != witness["encoder_tensors_bitwise_unchanged"]
        or not changed
        or sorted(changed) != sorted(witness["changed_readout_tensors"])
    ):
        raise ValueError("Frozen encoder or learned readout claim differs from retained weights")
    if [row["update"] for row in trained["cursor"]["curves"]] != list(range(1, updates + 1)):
        raise ValueError("Frozen encoder witness curve has a missing or duplicated update")
    steps = sorted({int(row["step"]) for row in trained["optimizer_state_dict"]["state"].values()})
    if steps != [updates] or steps != witness["optimizer_steps"]:
        raise ValueError("Frozen encoder witness optimizer steps differ")
    return {
        "status": "PASS",
        "updates_compared": [0, updates],
        "encoder_tensors_bitwise_unchanged": len(encoder),
        "readout_tensors_changed": len(changed),
        "optimizer_updates": 0,
        "generalization_claim": False,
    }


def replay(root: Path) -> dict:
    """Check included evidence only; never reopen training datasets or run an optimizer."""
    integrity = verify_extraction(root)
    torch.set_num_threads(1)
    campaign = root / "campaign"
    predictions = {
        path.parent.relative_to(campaign)
        .as_posix()
        .removeprefix("predictions/"): replay_predictions(path.parent)
        for path in sorted(campaign.glob("**/predictions/*/*/COMPLETE.json"))
    }
    exports = {
        path.parent.relative_to(campaign).as_posix().removeprefix("exports/"): replay_export(
            path.parent
        )
        for path in sorted(campaign.glob("**/exports/*/COMPLETE.json"))
    }
    witnesses = {}
    verification = campaign / "verification"
    for variant in ("DIRECT", "CORR"):
        witness = verification / f"PAIR_{variant}_ENCODER_CHANGE_100.json"
        if witness.exists():
            witnesses[f"PAIR-{variant}-S7@100"] = replay_encoder_change(
                verification / f"PAIR_{variant}_INITIAL_STATE.pt",
                verification / f"PAIR_{variant}_STATE_100.pt",
                witness,
            )
    frozen_witnesses = {}
    frozen_witness = verification / "PAIR_CORR_FROZEN_INVARIANT_100.json"
    if frozen_witness.exists():
        frozen_witnesses["PAIR-CORR-FROZEN-S7@100"] = replay_frozen_encoder(
            verification / "PAIR_CORR_FROZEN_INITIAL_STATE.pt",
            verification / "PAIR_CORR_FROZEN_STATE_100.pt",
            frozen_witness,
        )
    runtime_witnesses = {}
    if (campaign / "runtime/optimization_20261013/RUNTIME_BINDING.json").exists():
        from .replay_runtime import replay_runtime

        runtime_witnesses["native_cuda_graph_resume"] = replay_runtime(root)
    return {
        "integrity": integrity,
        "prediction_tables": predictions,
        "portable_exports": exports,
        "encoder_adaptation_witnesses": witnesses,
        "frozen_encoder_witnesses": frozen_witnesses,
        "optimized_runtime_witnesses": runtime_witnesses,
        "rows_recalculated": sum(result["rows"] for result in predictions.values()),
        "scientific_replay_performed": bool(
            predictions or exports or witnesses or frozen_witnesses or runtime_witnesses
        ),
        "scope": (
            "Included prediction scalars, portable examples, encoder and runtime witnesses; "
            "not raw producer replay"
        ),
        "optimizer_updates": 0,
        "torch_version": str(torch.__version__),
        "SOTA_confirmation": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Save independent replay output outside the original extraction")
    result = replay(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
