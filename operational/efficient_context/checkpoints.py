"""Offline complete-state inventory and next-sampler proof, with zero optimizer work."""

from .common import Campaign, atomic_bytes, atomic_json, digest, npz, read


def inventory(c: Campaign) -> list:
    """Export durable partial weights/curves without scoring partial OLD_DEV endpoints."""
    import hashlib

    import torch

    from e_jepa_ttc.simplex_t.training import learning_rate, load_checkpoint, state_digest
    from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal

    c.freeze()
    ledger = read(c.out / "PHYSICAL_WORK.json")
    records = []
    for path in sorted((c.out / "fits").glob("seed*/fold*/checkpoint_last.pt")):
        state = load_checkpoint(path)
        seed = state["identity"]["seed"]
        fold = int(path.parent.name.removeprefix("fold"))
        key = f"WIDE/fold{fold}/seed{seed}"
        if state["identity"]["freeze"] != digest(c.out / "PROTOCOL.json"):
            raise ValueError("durable checkpoint parent differs")
        binding = ledger["fits"][key]
        if state["completed_updates"] != binding["completed"]:
            raise ValueError("durable checkpoint and physical ledger progress differ")
        canonical = EngineWorkJournal._training_state_digest(path)
        if canonical != binding["checkpoint_state_sha256"]:
            raise ValueError("durable optimizer/RNG/sampler/model state changed")
        restored = load_checkpoint(path)
        if state_digest(restored) != state_digest(state):
            raise ValueError("complete-state independent reload differs")
        population = c.freeze()["sources"][str(fold)]["population"]
        draws = []
        for checkpoint in (state, restored):
            generator = torch.Generator().set_state(checkpoint["sampler_rng"])
            sample = torch.randint(population, (128,), generator=generator)
            draws.append(hashlib.sha256(sample.numpy().tobytes()).hexdigest())
        if draws[0] != draws[1]:
            raise ValueError("next sampler draw differs after reload")
        npz(path.parent / "WEIGHTS.npz", **{k: v.numpy() for k, v in state["model"].items()})
        atomic_bytes(
            path.parent / "TRAINING_CURVE.csv",
            (
                "update,loss,learning_rate,sampler_sha256\n"
                + "".join(
                    f"{i + 1},{value:.17g},{learning_rate(i + 1):.17g},"
                    f"{state['sampler_hashes'][i]}\n"
                    for i, value in enumerate(state["losses"])
                )
            ).encode(),
        )
        record = {
            "key": key,
            "checkpoint": str(path),
            "sha256": digest(path),
            "state_sha256": state_digest(state),
            "canonical_training_state_sha256": canonical,
            "journal_archive_sha256": binding["checkpoint_sha256"],
            "archive_difference_allowed_by_canonical_engine": digest(path)
            != binding["checkpoint_sha256"],
            "completed_updates": state["completed_updates"],
            "status": state["status"],
            "next_sampler_sha256": draws[0],
            "extra_optimizer_updates": 0,
            "partial_OLD_DEV_scored": False,
        }
        records.append(record)
    atomic_json(c.out / "CHECKPOINT_INVENTORY.json", {"fits": records})
    atomic_json(
        c.out / "RESUME_PROOF.json",
        {"status": "COMPLETE_STATE_AND_NEXT_SAMPLER_VERIFIED", "fits": records, "extra_updates": 0},
    )
    return records
