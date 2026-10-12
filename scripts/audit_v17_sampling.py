"""Replay every saved sampler draw from its original seed without optimizer steps."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import read, sha, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    partial = parser.parse_args().allow_partial
    torch.set_num_threads(1)
    run = Path(__file__).resolve().parents[1] / "artifacts/representation_audit_20261012"
    folder = run / "sampling_audit"
    folder.mkdir(exist_ok=True)
    results = {}
    for directory in sorted((run / "fits").iterdir()):
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        if receipt["status"] != "COMPLETE" and not partial:
            raise ValueError("Complete every registered fit before final sampler audit")
        if sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        payload = torch.load(
            directory / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        arm, seed_text = directory.name.split("-S")
        population = 18422 if arm in ("Z", "G", "D") else 53890
        expected_endpoint = 2500 if arm in ("Z", "G", "D") else 6000
        endpoint = receipt["completed_updates"]
        if endpoint > expected_endpoint or (
            receipt["status"] == "COMPLETE" and endpoint != expected_endpoint
        ):
            raise ValueError("Endpoint count differs from protocol")
        cursor = payload["cursor"]
        curves = cursor.get("curve", cursor.get("curves"))
        saved = cursor.get("draws", cursor.get("draw_sha"))
        if len(saved) != endpoint or [r["update"] for r in curves] != list(range(1, endpoint + 1)):
            raise ValueError("Missing, duplicated or reordered optimizer/draw records")
        generator = torch.Generator().manual_seed(int(seed_text))
        counts = np.zeros(population, np.int64)
        for expected in saved:
            indices = torch.randint(population, (128,), generator=generator).numpy()
            if hashlib.sha256(indices.tobytes()).hexdigest() != expected:
                raise ValueError("Sampler reset or altered draws: " + directory.name)
            np.add.at(counts, indices, 1)
        if not torch.equal(generator.get_state(), payload["sampler_generator_state"]):
            raise ValueError("Final sampler state differs from uninterrupted draw sequence")
        journal = read(directory / "UPDATE_JOURNAL.json")
        if journal["completed_updates"] != endpoint or journal["durable_updates"] != endpoint:
            raise ValueError("Journal and endpoint differ")
        steps = [int(v["step"]) for v in payload["optimizer_state_dict"]["state"].values()]
        if not steps or set(steps) != {endpoint}:
            raise ValueError("Optimizer clocks differ from endpoint")
        if payload["scheduler_state_dict"]["last_epoch"] != endpoint:
            raise ValueError("Scheduler clock differs from endpoint")
        pauses = []
        for path in sorted((run / "resume_audit" / directory.name).glob("pause_*.pt")):
            earlier = torch.load(path, map_location="cpu", weights_only=False)
            count = earlier["completed_updates"]
            previous = earlier["cursor"]
            if earlier["identity_sha256"] != payload["identity_sha256"]:
                raise ValueError("Pause belongs to another fit")
            if previous.get("draws", previous.get("draw_sha")) != saved[:count]:
                raise ValueError("Pause draw prefix changed")
            if previous.get("curve", previous.get("curves")) != curves[:count]:
                raise ValueError("Pause loss/gradient prefix changed")
            pauses.append({"updates": count, "sha256": sha(path), "prefix_exact": True})
        np.savez_compressed(folder / f"{directory.name}_EXPOSURE.npz", draws_per_position=counts)
        results[directory.name] = {
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "status": receipt["status"],
            "expected_endpoint": expected_endpoint,
            "updates": endpoint,
            "population": population,
            "sampler_every_draw_exact_from_initial_seed": True,
            "sampler_final_rng_state_exact": True,
            "curve_ordinals_complete": True,
            "optimizer_and_scheduler_clocks": endpoint,
            "observed_unique_positions": int((counts > 0).sum()),
            "total_draws": int(counts.sum()),
            "recovery_upper": journal["recovery_upper"],
            "retained_pause_prefix_checks": pauses,
        }
    write(
        run / "SAMPLING_CONTINUITY_AUDIT.json",
        {
            "status": "PASS_WITH_PENDING_FITS"
            if any(r["status"] != "COMPLETE" for r in results.values())
            else "PASS",
            "models": results,
            "optimizer_updates_executed": 0,
            "source_sha256": sha(Path(__file__)),
            "scope": "sampling/clock replay; no retraining of full scientific trajectories",
            "full_state_resume_equality": "covered by separate synthetic 10 versus 5+restore+5 QA",
            "pause_checkpoint_retention": "native writer keeps only its latest two versions",
        },
    )
    print(
        f"Verified uninterrupted sampler sequences through durable states for {len(results)} fits"
    )


if __name__ == "__main__":
    main()
