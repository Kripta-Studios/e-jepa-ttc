"""Train three fixed F1 refits, export seed7, and replay real sensor input."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from operational.evttc_transfer_audit.common import canonical_sha, file_sha, write_receipt
from operational.evttc_transfer_audit.provenance import assert_sources_unchanged, snapshot_sources
from operational.phase_adapt import ROLE, validate_population
from operational.phase_adapt_bundle import AdaptedPredictor, HeadExport
from operational.phase_adapt_training import configure_cuda, fit_head, load_head
from scripts.final_train40_transfer import HEAD, verify_candidate


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the declared recipe and save a portable final inference bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    cuda_config = configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    paths, parent = verify_candidate(args.candidate)
    receipt = read(args.data / "COMPLETE.json")
    if receipt["status"] != "COMPLETE" or receipt["role"] != ROLE:
        raise ValueError("Only explicit completed final refit population may train")
    for filename, key in (
        ("training.npz", "training_npz_sha256"),
        ("BINDING.json", "binding_sha256"),
        ("POPULATION.json", "population_sha256"),
        ("TARGET_LEDGER.json", "targets_sha256"),
        ("TRAIN40_RESTORE.json", "train40_restore_sha256"),
    ):
        if file_sha(args.data / filename) != receipt[key]:
            raise ValueError(f"Training input hash mismatch: {filename}")
    if read(args.data / "BINDING.json")["candidate"] != parent:
        raise ValueError("Feature cache was produced with a different parent")
    with np.load(args.data / "training.npz", allow_pickle=False) as stored:
        values = {k: stored[k] for k in stored.files}
    validate_population(values["sample_token"].tolist(), values["role"].tolist())
    if len(values["features"]) != receipt["rows"]:
        raise ValueError("Training population differs")
    sources = snapshot_sources(ROOT)
    sources["scripts/train_phase_adapt.py"] = file_sha(Path(__file__))
    binding = {
        "parent": parent,
        "data_receipt_sha256": file_sha(args.data / "COMPLETE.json"),
        "source_sha256": sources,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "cublas_workspace_config": cuda_config,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        },
        "role": ROLE,
        "scope": "training_resubstitution_not_independent_confirmation",
    }
    initial = torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    recipe = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 30,
        "max_updates": 2000,
        "grad_clip": 1.0,
        "selection": "last_endpoint",
        "domain_weight": "equal",
        "group_weight": "equal_within_domain",
    }
    write_receipt(
        args.output / "RECIPE.json",
        {
            "recipe": recipe,
            "seeds": [7, 13, 21],
            "delivery_seed": 7,
            "reason": "predeclared_before_training",
            "binding": binding,
        },
    )
    reports = {}
    for seed in (7, 13, 21):
        destination = args.output / f"seed_{seed}"
        if (destination / "COMPLETE.json").exists():
            report = read(destination / "COMPLETE.json")
            loaded = torch.load(
                destination / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            if loaded["binding"] != binding or loaded["recipe"] != {**recipe, "seed": seed}:
                raise ValueError("Refusing to resume a different training recipe")
            load_head(destination / "checkpoint.pt", report["checkpoint_sha256"])
        else:
            report = fit_head(
                initial,
                values,
                {**recipe, "seed": seed},
                destination,
                device=args.device,
                binding=binding,
            )
        reports[str(seed)] = report
    bundle = args.output / "FINAL_MODEL"
    bundle.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in {**paths, "head": args.output / "seed_7/checkpoint.pt"}.items():
        relative = f"components/{name}{source.suffix}"
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_sha(destination) != file_sha(source):
                raise ValueError("Existing bundle member differs")
        else:
            shutil.copyfile(source, destination)
        files[name] = {"path": relative, "sha256": file_sha(destination)}
    head = load_head(bundle / files["head"]["path"], files["head"]["sha256"])
    exported = cast(
        torch.jit.ScriptModule,
        torch.jit.trace(
            HeadExport(head),
            (
                torch.from_numpy(values["features"][:1].copy()),
                torch.from_numpy(values["rgb_width_logits"][:1].copy()),
            ),
        ),
    )
    export_path = bundle / "head.torchscript.pt"
    exported.save(str(export_path))
    replay = torch.jit.load(str(export_path))
    sample = torch.from_numpy(values["features"][:: max(1, len(values["features"]) // 32)].copy())
    widths = torch.from_numpy(
        values["rgb_width_logits"][:: max(1, len(values["features"]) // 32)].copy()
    )
    with torch.inference_mode():
        expected = HeadExport(head)(sample, widths)
        actual = replay(sample, widths)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.savez_compressed(
        bundle / "example_input.npz", features=sample.numpy(), rgb_width_logits=widths.numpy()
    )
    write_receipt(
        bundle / "example_output.json",
        {"columns": ["ttc", "phase", "q10_phase", "q90_phase"], "values": actual.tolist()},
    )
    files["torchscript_head"] = {"path": export_path.name, "sha256": file_sha(export_path)}
    write_receipt(
        bundle / "BUNDLE.json",
        {
            "schema": "phase_adapt_bundle_v1",
            "files": files,
            "parent_binding": parent,
            "new_head_seed": 7,
            "adapted_recipe": {**recipe, "seed": 7},
            "data_receipt_sha256": file_sha(args.data / "COMPLETE.json"),
            "train_receipt": receipt,
            "recipe_sha256": file_sha(args.output / "RECIPE.json"),
            "independent_confirmation": False,
            "modality": "RGB+events",
        },
    )
    population = read(args.data / "POPULATION.json")["rows"]
    target_ledger = read(args.data / "TARGET_LEDGER.json")["rows"]
    used = {row["query_id"]: row for row in target_ledger if row["status"] == "used"}
    row = next(row for row in population if row["query_id"] in used)
    sensor_row = {k: v for k, v in row.items() if k != "ttc_csv_path"}
    runtime = AdaptedPredictor(bundle, args.device)
    live = runtime.predict(sensor_row)
    shard = args.data / "shards" / used[row["query_id"]]["shard"]
    with np.load(shard, allow_pickle=False) as stored, torch.inference_mode():
        cached = head(
            torch.from_numpy(stored["features"]), torch.from_numpy(stored["rgb_width_logits"])
        )
    np.testing.assert_allclose(
        live["predictions"]["PHASE_ADAPT_FINAL_S7"], float(cached["ttc"][0]), rtol=1e-5, atol=1e-5
    )
    write_receipt(
        bundle / "RAW_REPLAY.json",
        {
            "query_id": row["query_id"],
            "prediction": live,
            "cached_ttc": float(cached["ttc"][0]),
            "rtol": 1e-5,
            "atol": 1e-5,
        },
    )
    assert_sources_unchanged(ROOT, sources)
    if verify_candidate(args.candidate)[1] != parent:
        raise ValueError("Sealed parent changed during adaptation")
    write_receipt(
        args.output / "RESULTS.json",
        {
            "reports": reports,
            "final_bundle_sha256": file_sha(bundle / "BUNDLE.json"),
            "recipe_hash": canonical_sha(recipe),
            "export_exact_parity": True,
            "raw_sensor_replay": "passed",
            "selected_seed": 7,
            "scope": "training_resubstitution_only",
        },
    )
    print(f"FINAL MODEL: {bundle.resolve()}", flush=True)


if __name__ == "__main__":
    main()
