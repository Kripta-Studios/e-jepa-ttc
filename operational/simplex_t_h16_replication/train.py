"""Execute only the six independent H16 fits; never evaluate during training."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

from common import (
    OUT,
    Lease,
    Resources,
    atomic_json,
    digest,
    historical_spec,
    ids,
    protocol,
    publish_json,
    record,
    sources,
    validate_pins,
)


def prove_restore(path: Path, population: int) -> dict:
    """Check complete state, restored RNG and next sampler without optimizer updates."""
    import torch
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.training import (
        atomic_checkpoint,
        learning_rate,
        load_checkpoint,
        state_digest,
    )

    state = load_checkpoint(path)
    copy = OUT / "verification/resume_checkpoint100.pt"
    copy.parent.mkdir(exist_ok=True)
    atomic_checkpoint(copy, state)
    loaded = load_checkpoint(copy)
    if state_digest(state) != state_digest(loaded):
        raise ValueError("complete checkpoint roundtrip changed")
    model = TemporalRefiner(TemporalConfig(**state["identity"]["config"])).float()
    model.load_state_dict(loaded["model"])
    if state_digest(dict(model.state_dict())) != state_digest(state["model"]):
        raise ValueError("model state restore changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    optimizer.load_state_dict(loaded["optimizer"])
    if state_digest(optimizer.state_dict()) != state_digest(state["optimizer"]):
        raise ValueError("optimizer state restore changed")
    a, b = torch.Generator(), torch.Generator()
    a.set_state(state["sampler_rng"])
    b.set_state(loaded["sampler_rng"])
    ia, ib = (torch.randint(population, (128,), generator=g) for g in (a, b))
    if not torch.equal(ia, ib) or not torch.equal(state["torch_rng"], loaded["torch_rng"]):
        raise ValueError("RNG/sampler restore changed")
    result = dict(
        status="EXACT_COMPLETE_STATE_RESTORE_NO_EXTRA_UPDATES",
        completed=100,
        next_update=101,
        restored_state_sha256=state_digest(state),
        checkpoint100_sha256=digest(copy),
        next_sampler_sha256=hashlib.sha256(ia.numpy().tobytes()).hexdigest(),
        next_learning_rate=learning_rate(101),
        optimizer_updates_for_probe=0,
        continuous_vs_interrupted_training_reexecuted=False,
        limit=(
            "Full continuous duplicate fit is not authorized; "
            "historical deterministic engine retained."
        ),
    )
    publish_json(OUT / "verification/RESUME_ROUNDTRIP.json", result)
    return result


def main() -> int:
    from e_jepa_ttc.simplex_t.arms import resolve_arm
    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.physical_accounting import audit_physical_work
    from e_jepa_ttc.simplex_t.training import fit, learning_rate, load_checkpoint
    from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget

    p, pin = protocol()
    resources = Resources(training=True)
    resources.check()
    validate_pins(p, full=True)
    with Lease():
        if (OUT / "ENDPOINTS.json").exists():
            print("ALL_SIX_ENDPOINTS_ALREADY_SEALED_NO_TRAINING", flush=True)
            return 0
        s = sources(p)
        graph = {v["key"]: 2500 for v in ids()}
        budget = WorkBudget(OUT / "PHYSICAL_WORK.json", graph, technical_reserved=0)
        endpoints = []
        for row in ids():
            resources.check()
            validate_pins(p, full=False)
            spec = historical_spec(s, row["fold"])
            binding = resolve_arm(spec, s.graph)
            source = s.train(spec)
            if source.identity_sha256 != p["sources"][str(row["fold"])]["train_sha256"]:
                raise ValueError("H16 TRAIN source changed after protocol")
            folder = OUT / "fits" / row["key"]
            path = folder / "checkpoint_last.pt"
            prior = load_checkpoint(path) if path.exists() else None
            if prior is None or prior["status"] != "COMPLETED":

                class ProgressJournal(EngineWorkJournal):
                    """Persist actual update boundaries25 without changing the fit recipe."""

                    def before_update(self, completed: int) -> None:
                        super().before_update(completed)
                        if completed % 25 == 0:
                            atomic_json(
                                OUT / "UPDATE_PROGRESS.json",
                                dict(
                                    current_fit=self.key,
                                    confirmed_completed_updates=completed,
                                    durable_checkpoint_updates=self.saved,
                                    protocol_sha256=pin,
                                    endpoint_count=len(endpoints),
                                ),
                            )

                journal = ProgressJournal(budget, row["key"])
                first_pause = (
                    row == ids()[0] and not (OUT / "verification/RESUME_ROUNDTRIP.json").exists()
                )

                def admitted(
                    pause: bool = first_pause, active: EngineWorkJournal = journal
                ) -> bool:
                    # Stop at the first safe100 checkpoint, never create a technical fit.
                    if pause and active.saved == 100:
                        return False
                    return resources()

                result = fit(
                    source,
                    binding.model,
                    folder,
                    seed=row["seed"],
                    freeze_sha256=pin,
                    resource_ok=admitted,
                    resume=prior is not None,
                    journal=journal,
                    device="cpu",
                )
                atomic_json(
                    OUT / "STATUS.json",
                    dict(
                        status=result["status"],
                        current_fit=row,
                        completed_updates=result["completed_updates"],
                        endpoint_count=len(endpoints),
                        scientific_saved_updates=record(budget.path)["accounting"][
                            "scientific_saved_updates"
                        ],
                    ),
                )
                if result["status"] != "COMPLETED":
                    if first_pause and result["completed_updates"] == 100:
                        prove_restore(path, source.population)
                        print("AUTHORIZED_FIRST_FIT_PAUSED_AT_100_RESUME_PROVEN", flush=True)
                        return 4
                    return 3
            model = load_endpoint(
                path,
                binding.model,
                seed=row["seed"],
                freeze_sha256=pin,
                train_source_sha256=source.identity_sha256,
                endpoint_sha256=digest(path),
            )
            EngineWorkJournal(budget, row["key"]).start(2500, path)
            state = load_checkpoint(path)
            if row == ids()[0]:
                proof = record(OUT / "verification/RESUME_ROUNDTRIP.json")
                if state["sampler_hashes"][100] != proof["next_sampler_sha256"]:
                    raise ValueError("actual resumed update101 sampler differs")
                publish_json(
                    OUT / "verification/ACTUAL_RESUME.json",
                    dict(
                        status="ACTUAL_AUTHORIZED_FIT_RESUMED_FROM_100_TO_2500",
                        sample101_matches=True,
                        scientific_saved_updates=2500,
                        additional_or_replayed_optimizer_updates=0,
                    ),
                )
            curves = "update,loss,learning_rate,sampler_sha256\n" + "".join(
                f"{i + 1},{loss:.17g},{learning_rate(i + 1):.17g},{state['sampler_hashes'][i]}\n"
                for i, loss in enumerate(state["losses"])
            )
            from common import atomic_bytes

            atomic_bytes(folder / "TRAINING_CURVE.csv", curves.encode())
            endpoints.append(
                dict(
                    **row,
                    checkpoint=str(path.relative_to(OUT)),
                    checkpoint_sha256=digest(path),
                    identity_sha256=state["identity_sha256"],
                    train_source_sha256=source.identity_sha256,
                    model=state["identity"]["config"],
                )
            )
            print(
                json.dumps(
                    dict(
                        status="ENDPOINT2500_VERIFIED",
                        key=row["key"],
                        endpoint_count=len(endpoints),
                        saved_updates=len(endpoints) * 2500,
                    )
                ),
                flush=True,
            )
            del model, state, source, prior
            s.release()
            gc.collect()
        s.release()
        validate_pins(p, full=True)
        accounting = audit_physical_work(
            record(budget.path), expected_graph=graph, technical_reserved=0, resource_ok=resources
        )
        if accounting["saved_updates"] != 15000:
            raise ValueError("six-fit scientific accounting differs")
        publish_json(
            OUT / "ENDPOINTS.json",
            dict(
                schema="independent_h16_endpoints_v1",
                protocol_sha256=pin,
                fits=endpoints,
                all_six_frozen_before_evaluation=True,
                old_dev_scores_read=False,
                scientific_saved_updates=15000,
            ),
        )
        atomic_json(OUT / "ACCOUNTING.json", accounting)
        atomic_json(
            OUT / "STATUS.json",
            dict(
                status="ALL_SIX_ENDPOINTS_SEALED_NOT_EVALUATED",
                endpoint_count=6,
                scientific_saved_updates=15000,
                technical_optimizer_updates=0,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
