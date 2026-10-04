"""Execute the independent efficient-context queue and its resumable endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .common import (
    ROOT,
    Campaign,
    Lease,
    atomic_bytes,
    atomic_json,
    digest,
    npz,
    read,
    release,
    spec,
)


def prepare(c: Campaign) -> None:
    """Audit TRAIN contracts and preregister WIDE before new OLD_DEV evaluation."""
    import torch

    from e_jepa_ttc.efficient_context.sparse_history import WideSource
    from e_jepa_ttc.simplex_t.arms import resolve_arm
    from e_jepa_ttc.simplex_t.training import state_digest

    c.require_resources()
    if (c.out / "PROTOCOL.json").exists():
        c.freeze()
        return
    qa = c.out / "TEST_RESULTS/QA.json"
    if not qa.exists() or read(qa)["status"] != "PASSED":
        raise ValueError("QA must pass before freeze and optimizer updates")
    s = c.sources()
    rows = {}
    for fold in range(3):
        c.require_resources()
        parent = s.train(spec(s, fold))
        expected = c.h16["sources"][str(fold)]["train_sha256"]
        if parent.identity_sha256 != expected:
            raise ValueError("historical H16 TRAIN identity differs")
        wide = WideSource(parent)
        x, t, v, e, y, m = wide.gather(torch.arange(min(128, wide.population)))
        if not v[:, -1].all() or not torch.isfinite(x[v]).all():
            raise ValueError("WIDE gather contract failed")
        rows[str(fold)] = {
            "parent_sha256": expected,
            "wide_sha256": wide.identity_sha256,
            "population": wide.population,
            "model": asdict(resolve_arm(spec(s, fold), s.graph).model),
            "normalizer_sha256": state_digest(
                {
                    "mean": torch.from_numpy(parent.normalizer.mean),
                    "scale": torch.from_numpy(parent.normalizer.scale),
                    "ids": parent.normalizer.consumed_ids_sha256,
                }
            ),
        }
        npz(
            c.out / f"normalizers/fold{fold}.npz",
            mean=parent.normalizer.mean,
            scale=parent.normalizer.scale,
        )
        del parent, wide, x, t, v, e, y, m
        release(s)
    code = [
        ROOT / "src/e_jepa_ttc/efficient_context/sparse_history.py",
        ROOT / "src/e_jepa_ttc/efficient_context/analytical.py",
        ROOT / "src/e_jepa_ttc/simplex_t/training.py",
        ROOT / "src/e_jepa_ttc/simplex_t/model.py",
        Path(__file__),
        Path(__file__).with_name("common.py"),
    ]
    p = {
        "schema": "efficient_context_freeze_v1",
        "config": c.config,
        "config_sha256": digest(c.config_path),
        "policy_sha256": digest(c.policy_path),
        "sources": rows,
        "historical_protocol_sha256": digest(c.h16_path),
        "science_files": [{"path": str(v), "sha256": digest(v)} for v in code],
        "qa_sha256": digest(qa),
        "optimizer_updates_at_freeze": 0,
        "roots": {
            "code_root": str(ROOT),
            "historical_artifact_root": str(c.historical / "artifacts"),
            "raw_train_root": str(c.raw),
            "new_output_root": str(c.out),
        },
        "torch_version": str(torch.__version__),
        "python_executable": sys.executable,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status_preserved": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "bootstrap_draws": c.h16["bootstrap_draws"],
        "historical_controls": c.h16["historical_controls"],
    }
    atomic_json(c.out / "PROTOCOL.json", p)
    atomic_bytes(
        c.out / "PROTOCOL.sha256", (digest(c.out / "PROTOCOL.json") + "  PROTOCOL.json\n").encode()
    )
    atomic_json(
        c.out / "SOURCE_ADMISSION.json",
        {"status": "TRAIN_CACHE_ADMITTED", "sources": rows, "roots": p["roots"], "updates": 0},
    )
    print("FROZEN_TRAIN_ONLY_ZERO_UPDATES", flush=True)


def train(c: Campaign, seed: int) -> bool:
    """Train three endpoints at fixed update2500 with canonical full-state engine."""
    from e_jepa_ttc.efficient_context.sparse_history import WideSource
    from e_jepa_ttc.simplex_t.model import TemporalConfig
    from e_jepa_ttc.simplex_t.training import fit, learning_rate, load_checkpoint, state_digest
    from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget

    p = c.freeze()
    if seed not in (7, 13, 23):
        raise ValueError("unregistered seed")
    if seed != 7 and not read(c.out / "H8_WIDE_RESULTS.json")["replication_authorized"]:
        raise ValueError("conditional replication rule not met")
    pin = digest(c.out / "PROTOCOL.json")
    graph = {f"WIDE/fold{fold}/seed{v}": 2500 for v in (7, 13, 23) for fold in range(3)}
    budget = WorkBudget(c.out / "PHYSICAL_WORK.json", graph, technical_reserved=0)
    s = c.sources()
    endpoints = []
    for fold in range(3):
        c.require_resources()
        parent = s.train(spec(s, fold))
        source = WideSource(parent)
        if source.identity_sha256 != p["sources"][str(fold)]["wide_sha256"]:
            raise ValueError("WIDE source changed")
        folder = c.out / f"fits/seed{seed}/fold{fold}"
        checkpoint = folder / "checkpoint_last.pt"
        prior = load_checkpoint(checkpoint) if checkpoint.exists() else None
        key = f"WIDE/fold{fold}/seed{seed}"

        class Journal(EngineWorkJournal):
            def before_update(self, completed: int) -> None:
                super().before_update(completed)
                if completed % 25 == 0:
                    atomic_json(
                        c.out / "UPDATE_PROGRESS.json",
                        {
                            "fit": self.key,
                            "confirmed": completed,
                            "durable": self.saved,
                            "seed": seed,
                        },
                    )

        if prior is None or prior["completed_updates"] != 2500:
            result = fit(
                source,
                TemporalConfig(**p["sources"][str(fold)]["model"]),
                folder,
                seed=seed,
                freeze_sha256=pin,
                resource_ok=c.check,
                resume=prior is not None,
                journal=Journal(budget, key),
                device="cpu",
            )
            if result["status"] != "COMPLETED":
                atomic_json(c.out / "STATUS.json", result | {"fit": key})
                release(s)
                return False
        state = load_checkpoint(checkpoint)
        if (
            state["identity"]["freeze"] != pin
            or state["identity"]["source"] != source.identity_sha256
        ):
            raise ValueError("endpoint binding mismatch")
        curve = "update,loss,learning_rate,sampler_sha256\n" + "".join(
            f"{i + 1},{loss:.17g},{learning_rate(i + 1):.17g},{state['sampler_hashes'][i]}\n"
            for i, loss in enumerate(state["losses"])
        )
        atomic_bytes(folder / "TRAINING_CURVE.csv", curve.encode())
        npz(folder / "WEIGHTS.npz", **{k: v.numpy() for k, v in state["model"].items()})
        endpoints.append(
            {
                "fold": fold,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "source_sha256": source.identity_sha256,
                "model": p["sources"][str(fold)]["model"],
                "updates": 2500,
            }
        )
        # Verify restored optimizer/RNG/sampler without any duplicate training.
        restored = load_checkpoint(checkpoint)
        if state_digest(restored) != state_digest(state):
            raise ValueError("resume roundtrip mismatch")
        atomic_json(
            c.out / "RESUME_PROOF.json",
            {
                "status": "COMPLETE_STATE_ROUNDTRIP_VERIFIED",
                "checkpoint": str(checkpoint),
                "extra_updates": 0,
                "state_sha256": state_digest(state),
            },
        )
        del state, restored, prior, source, parent
        release(s)
    atomic_json(
        c.out / f"ENDPOINTS_seed{seed}.json",
        {"fits": endpoints, "protocol_sha256": pin, "all_three_frozen_before_evaluation": True},
    )
    return True


def execute(c: Campaign) -> None:
    """Advance independent branches; never call historical training launchers."""
    from .analysis import evaluate
    from .audit import audit
    from .profile import run_profile

    audit(c)
    prepare(c)
    if train(c, 7):
        evaluate(c, 7)
        if read(c.out / "H8_WIDE_RESULTS.json")["replication_authorized"]:
            for seed in (13, 23):
                if not train(c, seed):
                    break
                evaluate(c, seed)
    run_profile(c)


def main() -> int:
    """Useful resumable CLI; failure status never masquerades as a scientific negative."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("status", "audit", "prepare", "all", "train", "evaluate", "profile", "package"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse sealed fragments and complete checkpoints"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    c = Campaign(args.protocol)
    atomic_bytes(
        c.out / "COMMAND_LOG.jsonl",
        (c.out / "COMMAND_LOG.jsonl").read_bytes()
        + (json.dumps({"argv": sys.argv, "executable": sys.executable}) + "\n").encode()
        if (c.out / "COMMAND_LOG.jsonl").exists()
        else (json.dumps({"argv": sys.argv, "executable": sys.executable}) + "\n").encode(),
    )
    if args.action == "status":
        print(
            json.dumps(
                {
                    "output": str(c.out),
                    "resources_admitted": c.check(),
                    "status": read(c.out / "STATUS.json")
                    if (c.out / "STATUS.json").exists()
                    else "PENDING",
                }
            )
        )
        return 0
    try:
        with Lease(c.out):
            if args.action == "audit":
                from .audit import audit

                audit(c)
            elif args.action == "prepare":
                prepare(c)
            elif args.action == "train":
                if not train(c, args.seed):
                    return 3
            elif args.action == "evaluate":
                from .analysis import evaluate

                evaluate(c, args.seed)
            elif args.action == "profile":
                from .profile import run_profile

                run_profile(c)
            elif args.action == "package":
                from .package import package

                package(c)
            else:
                execute(c)
                from .package import package

                package(c)
        return 0
    except (InterruptedError, OSError) as exc:
        atomic_json(
            c.out / "STATUS.json",
            {
                "status": "PAUSED_RESOURCE"
                if isinstance(exc, InterruptedError)
                else "BLOCKED_DEPENDENCY",
                "error": repr(exc),
            },
        )
        print(repr(exc), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
