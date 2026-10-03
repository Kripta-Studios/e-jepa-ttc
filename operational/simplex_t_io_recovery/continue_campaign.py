"""Apply a new operational window without rewriting the frozen scientific protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
AUTH = NIGHT / "CONTINUATION_AUTHORIZATION_20261003.json"
MODULES = {
    "operational.simplex_t_cost_context.engine",
    "operational.simplex_t_cost_context.analyze",
    "operational.simplex_t_cost_context.profile",
    "operational.simplex_t_cost_context.deliver",
}


def policy() -> dict[str, Any]:
    """Validate the explicit operational amendment and its historical binding."""
    p = json.loads(AUTH.read_bytes())
    if p["scientific_saved_update_cap"] != 60000 or p["fits_cap"] != 24:
        raise ValueError("scientific authorization changed")
    if p["own_artifact_budget_bytes"] != 10_000_000_000:
        raise ValueError("explicit 10 GB own-artifact authorization required")
    window = NIGHT / "WINDOW_AUTHORIZATION.json"
    if hashlib.sha256(window.read_bytes()).hexdigest() != p["original_window_sha256"]:
        raise ValueError("historical window bytes changed")
    protocol = NIGHT / "PROTOCOL_COST_CONTEXT.json"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != p["scientific_protocol_sha256"]:
        raise ValueError("scientific protocol changed")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != p["operational_source_sha256"]:
        raise ValueError("operational source differs from the admitted continuation")
    return p


def apply_policy(p: dict[str, Any]) -> None:
    """Overlay resource/deadline reads in this process; leave pinned bytes intact."""
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from operational.simplex_t_h16_replication import common
    from operational.simplex_t_io_recovery import run as io

    original_read = common.record
    original_window = original_read(NIGHT / "WINDOW_AUTHORIZATION.json")
    effective_window = dict(
        original_window,
        accepted_at_utc=p["accepted_at_utc"],
        deadline_utc=p["deadline_utc"],
        accepted_at_europe_madrid=p["accepted_at_europe_madrid"],
        deadline_europe_madrid=p["deadline_europe_madrid"],
        own_artifact_budget_bytes=p["own_artifact_budget_bytes"],
        no_push=False,
        continuation_authorization=str(AUTH),
        original_window_sha256=p["original_window_sha256"],
    )

    def operational_read(path: Path) -> dict:
        if Path(path).resolve() == (NIGHT / "WINDOW_AUTHORIZATION.json").resolve():
            return dict(effective_window)
        return original_read(path)

    common.record = operational_read
    sys.modules["common"] = common
    from operational.simplex_t_cost_context import engine

    engine.ARTIFACT_CAP = p["own_artifact_budget_bytes"]
    os.replace = io.replace_with_retry


def run_module(name: str, args: list[str]) -> None:
    """Execute one authorized phase with the unchanged numeric implementation."""
    if name not in MODULES:
        raise ValueError("phase outside the finite continuation")
    p = policy()
    apply_policy(p)
    module: Any = importlib.import_module(name)
    sys.argv = [name, *args]
    if name.endswith(".engine"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--family", choices=["N3"], required=True)
        raise SystemExit(module.main(parser.parse_args(args).family))
    if name.endswith(".deliver"):
        select = module.selected_files

        def current_delivery_files() -> dict[str, Path]:
            files = {
                key: path
                for key, path in select().items()
                if key not in {"RECOVERED_DRIVER.log", "IO_REPLACE_RETRIES.jsonl"}
                and not key.startswith("compact_delivery/")
                and not key.endswith("CONTINUATION_DRIVER_1.log")
            }
            files["provenance/operational/continue_campaign.py"] = Path(__file__).resolve()
            return files

        module.selected_files = current_delivery_files
    module.main()


def driver(p: dict[str, Any]) -> None:
    """Reuse completed stages and run exactly the remaining deterministic queue."""
    from operational.simplex_t_closure.runtime import atomic_json
    from operational.simplex_t_h16_replication.common import memory
    from operational.simplex_t_io_recovery.resume_after_commit_recovery import check_startup

    if datetime.now(UTC) >= datetime.fromisoformat(p["deadline_utc"]):
        raise InterruptedError("new continuation window expired")
    metrics = memory()
    if p.get("startup_margin_waiver_by_user"):
        if metrics["available"] < 3 * 1024**3 or metrics["commit_headroom"] < 1024**3:
            raise InterruptedError(
                "User waived additional startup slack; RAM3GiB/commit1GiB still required: "
                + json.dumps(metrics)
            )
    else:
        check_startup(metrics)
    apply_policy(p)
    module: Any = importlib.import_module("operational.simplex_t_cost_context.driver")
    original_run = module.run

    def command(script: str, *args: str) -> list[str]:
        target = "operational." + script.removesuffix(".py").replace("/", ".")
        return [
            sys.executable,
            "-B",
            "-m",
            "operational.simplex_t_io_recovery.continue_campaign",
            "--module",
            target,
            *args,
        ]

    def reuse(phase: str, argv: list[str], *, training: bool = False) -> int:
        if phase in {
            "N1_ANALYSIS",
            "N1_DELIVERY",
            "N2_QA",
            "N2_REGISTER",
            "N2_TECHNICAL",
            "N2_ANALYSIS",
        }:
            module.persist(phase, "REUSED_VERIFIED_COMPLETED_STAGE_ZERO_UPDATES")
            return 0
        return original_run(phase, argv, training=training)

    module.command = command
    module.run = reuse
    atomic_json(
        NIGHT / "CONTINUATION_PREFLIGHT.json",
        dict(
            resources=metrics,
            startup_margin_waiver_by_user=p.get("startup_margin_waiver_by_user", False),
            authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),
            scientific_protocol_unchanged=True,
            completed_stages_reused=True,
            next_id="COST_CONTEXT_20261003/SET_AGE_C0/fold1/seed7",
            resume_from=495,
            optimizer_updates_in_preflight=0,
        ),
    )
    sys.argv = ["continued_driver", "--campaign", "SIMPLEX_T_NOCTURNAL_20261003", "--adopt-h16"]
    raise SystemExit(module.main())


def main() -> None:
    """Dispatch a verified amendment or perform its zero-update admission check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=sorted(MODULES))
    parser.add_argument("--check-policy", action="store_true")
    args, remaining = parser.parse_known_args()
    p = policy()
    if args.check_policy:
        apply_policy(p)
        from operational.simplex_t_cost_context import engine

        assert (
            engine.record(NIGHT / "WINDOW_AUTHORIZATION.json")["deadline_utc"] == p["deadline_utc"]
        )
        assert engine.ARTIFACT_CAP == 10_000_000_000
        assert "torch" not in sys.modules
        print(
            json.dumps(
                dict(
                    status="OPERATIONAL_OVERLAY_VERIFIED",
                    optimizer_updates=0,
                    historical_bytes_unchanged=True,
                    artifact_cap=engine.ARTIFACT_CAP,
                )
            )
        )
    elif args.module:
        run_module(args.module, remaining)
    else:
        driver(p)


if __name__ == "__main__":
    main()
