"""Rebuild PHASE17 and H-only normalization for each frozen A5 residual."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from operational.range_transfer.cache import HeadCache
from operational.rgb_port.infer_experts import save_npz
from operational.rgb_port.prepare_features import (
    build_history,
    build_observations,
    fit_h_normalizer,
)

from .common import OWNER, RUN, freeze, read, sha, write
from .train_readout import VectorCache, load_endpoint


def prepare(bound: float, role: str, seed: int = 7) -> Path:
    """Keep C2F/PAIR/base/clock fixed; regenerate all A5-dependent coordinates."""
    if role == "V" and not (RUN / "SCREEN_ENDPOINTS.json").exists():
        raise ValueError("V features wait for the complete frozen head queue")
    family = f"R-C{round(bound * 100):02d}-S{seed}"
    directory = RUN / "families" / family
    directory.mkdir(parents=True, exist_ok=True)
    receipt = read(RUN / "fits" / family / "CHECKPOINT_RECEIPT.json")
    if sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]:
        raise ValueError("Readout checkpoint changed")
    if role == "V" and seed != 7 and not (RUN / "REPLICATION_ENDPOINTS.json").exists():
        raise ValueError("Replica queue must be frozen before V")
    completed = directory / role / "REBUILD.json"
    if completed.exists():
        previous = read(completed)
        history = directory / role / "HISTORY.npz"
        if (
            sha(history) != previous["history_sha256"]
            or sha(directory / "H_NORMALIZER.json") != previous["normalizer_sha256"]
            or sha(directory / "PARENTS.json") != previous["parent_sha256"]
        ):
            raise ValueError("Completed family features changed")
        return history
    if receipt["status"] != "COMPLETE" or receipt["completed_updates"] != 6000:
        raise ValueError("A5 readout incomplete")
    parents = read(OWNER / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json")
    parents["endpoints"]["E_A5_MATCHED"] = receipt
    parent_path = directory / "PARENTS.json"
    freeze(parent_path, parents)
    cache = VectorCache(role)
    model = load_endpoint(Path(receipt["checkpoint_path"]), ResidualSpec(bound))
    index = {str(t): i for i, t in enumerate(cache.values["sample_token"])}
    raw = OWNER / "features" / role / "E_RAW"
    with np.load(raw / "blocks.npz", allow_pickle=False) as archive:
        blocks = {k: archive[k].copy() for k in archive.files}
    if not set(map(str, blocks["observation_id"])).issubset(index):
        raise ValueError("Original observation has no new frozen vector input")
    original_c2f = blocks["expert_phase"][:, 1].copy()
    original_pair = blocks["expert_phase"][:, 2].copy()
    original_clock = {k: blocks[k].copy() for k in ("observation_id", "anchor_us", "available_us")}
    with torch.inference_mode():
        for begin in range(0, len(blocks["observation_id"]), 128):
            ids = torch.tensor(
                [index[str(t)] for t in blocks["observation_id"][begin : begin + 128]]
            )
            inputs = cache.batch(ids)
            if any(k.startswith(("target_", "dino_")) for k in inputs):
                raise ValueError("H/V inference cache has training target fields")
            out = model(inputs)
            phase = expert_phase_from_ttc(out.ttc_mean_seconds.numpy())
            margin = torch.minimum(
                out.log_height_ratio[:, -1].abs() / 0.002, out.sensor_support[:, -1] / 0.0001
            )
            diag = torch.stack(
                (out.diagnostics["transport_flow_magnitude"][:, -1], margin, out.ttc_log_variance),
                -1,
            )
            blocks["expert_phase"][begin : begin + len(ids), 0] = phase
            blocks["a5_diagnostics3"][begin : begin + len(ids)] = diag.numpy()
    assert np.array_equal(blocks["expert_phase"][:, 1], original_c2f)
    assert np.array_equal(blocks["expert_phase"][:, 2], original_pair)
    assert all(np.array_equal(blocks[k], v) for k, v in original_clock.items())
    if read(RUN / "qa/READOUT_PARITY.json")["checks"]["pair133_invariant"] is not True:
        raise ValueError("PAIR reuse not admitted")
    role_dir = directory / role
    role_dir.mkdir(exist_ok=True)
    save_npz(role_dir / "blocks.npz", blocks)
    with np.load(raw / "history.npz", allow_pickle=False) as archive:
        history = {k: archive[k].copy() for k in archive.files}
    lookup = {str(t): i for i, t in enumerate(blocks["observation_id"])}
    history["current_expert_phase"] = np.stack(
        [blocks["expert_phase"][lookup[str(t)]] for t in history["observation_id"][:, -1]]
    )
    save_npz(role_dir / "history.npz", history)
    build_observations(
        role_dir / "blocks.npz",
        role_dir / "OBSERVATIONS.npz",
        role=role,
        modality="event",
        freeze=parent_path,
    )
    normalizer = directory / "H_NORMALIZER.json"
    if role == "H":
        fit_h_normalizer(role_dir / "OBSERVATIONS.npz", normalizer, modality="event")
    destination = role_dir / "HISTORY.npz"
    build_history(
        role_dir / "OBSERVATIONS.npz",
        role_dir / "history.npz",
        normalizer,
        destination,
        role=role,
        modality="event",
        freeze=parent_path,
    )
    if role == "H":
        base = read(RUN / "ADMISSION.json")
        base["cache_path"] = str(destination.resolve())
        base["normalizer_path"] = str(normalizer.resolve())
        base["fit_prefix"] = f"R-C{round(bound * 100):02d}-CTX"
        base["family"] = family
        base["files"] = {
            **base["files"],
            **{
                str(p.resolve()): sha(p)
                for p in (destination, normalizer, parent_path, Path(receipt["checkpoint_path"]))
            },
        }
        freeze(directory / "HEAD_ADMISSION.json", base)
        head = HeadCache(destination)
        if head.population != 18422:
            raise ValueError("H population changed")
    write(
        role_dir / "REBUILD.json",
        {
            "status": "COMPLETE",
            "role": role,
            "seed": seed,
            "readout": family,
            "parent_sha256": sha(parent_path),
            "normalizer_sha256": sha(normalizer),
            "history_sha256": sha(destination),
            "C2F_PAIR_clocks_unchanged": True,
            "PAIR_input_identity_qa": sha(RUN / "qa/READOUT_PARITY.json"),
            "source_blocks_sha256": sha(raw / "blocks.npz"),
            "source_history_sha256": sha(raw / "history.npz"),
        },
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["H", "V"])
    parser.add_argument("--bound", type=float, choices=[0.05, 0.20], required=True)
    parser.add_argument("--seed", type=int, choices=[7, 13, 23], default=7)
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = prepare(args.bound, args.role, args.seed)
    if args.fit:
        if args.role != "H":
            raise ValueError("Only H fits")
        from .train_context import fit

        binding = read(path.parent.parent / "HEAD_ADMISSION.json")
        print(fit(HeadCache(path), RUN, binding, mass_id=0, loss_id=0, seed=args.seed), flush=True)


if __name__ == "__main__":
    main()
