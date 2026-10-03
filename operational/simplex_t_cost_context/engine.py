"""Isolated new-study execution using the unchanged historical numeric engine.

Factory/loss bindings are process-local and scoped. Never import this runner into
an active historical training process; N1 runs in its own unchanged interpreter.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from operational.simplex_t_closure.runtime import (
    atomic_bytes,
    atomic_json,
    digest,
    publish_json,
)
from operational.simplex_t_h16_replication.common import (
    ROOT,
    historical_spec,
    memory,
    record,
    sources,
    validate_pins,
)

if TYPE_CHECKING:
    from torch import Tensor, nn

    from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources
    from e_jepa_ttc.simplex_t.training import FitJournal, QuerySource

OUT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
EXEC = OUT / "execution"
N1 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
CAMPAIGN = "COST_CONTEXT_20261003"
ARM_ORDER = (
    "FULL_C0",
    "A5_ONLY_C0",
    "C2F_ONLY_C0",
    "A5_PAIR_C0",
    "SET_AGE_C0",
    "SET_NOTIME_C0",
)


def ids() -> list[dict[str, Any]]:
    """Fixed N2 then N3 queue, each seed7 arm's folds0/1/2."""
    return [
        dict(
            key=f"{CAMPAIGN}/{arm}/fold{fold}/seed7",
            arm=arm,
            family="N2" if arm in ARM_ORDER[:4] else "N3",
            fold=fold,
            seed=7,
            updates=2500,
        )
        for arm in ARM_ORDER
        for fold in range(3)
    ]


def protocol() -> tuple[dict, str]:
    """Reject changes to the independent new-study protocol."""
    path = OUT / "PROTOCOL_COST_CONTEXT.json"
    p = record(path)
    pin = (OUT / "PROTOCOL_COST_CONTEXT.sha256").read_text().split()[0]
    if digest(path) != pin or p["fits"] != ids():
        raise ValueError("new-study protocol/fit graph changed")
    if p["scientific_updates"] != 45000 or p["lambda_cost"] != 0:
        raise ValueError("new-study authorization or objective changed")
    return p, pin


def _owner(path: Path) -> psutil.Process | None:
    if not path.exists():
        return None
    prior = record(path)
    try:
        process = psutil.Process(prior["pid"])
        return process if process.create_time() == prior["create_time"] else None
    except psutil.NoSuchProcess:
        return None


class Lease:
    """One writer for the new root; no stopping or rewriting another owner."""

    def __enter__(self) -> Lease:
        EXEC.mkdir(parents=True, exist_ok=True)
        self.path = EXEC / "WRITER.lock"
        live = _owner(self.path)
        if live is not None:
            raise RuntimeError(f"live new-study writer pid={live.pid}")
        if self.path.exists():
            publish_json(
                EXEC / ("STALE_OWNER_" + digest(self.path)[:12] + ".json"), record(self.path)
            )
            self.path.unlink()
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *exc: object) -> None:
        if record(self.path)["pid"] != os.getpid():
            raise RuntimeError("new-study owner changed")
        self.path.unlink()


class Resources:
    """Check shared campaign limits while writing only new-study measurements."""

    def __init__(self, *, training: bool = False) -> None:
        self.training = training
        self.last = 0.0
        self.peak = 0
        self.minimum_available = 2**63 - 1
        self.minimum_commit = 2**63 - 1
        self._journals: dict[Path, tuple[int, dict]] = {}

    def _read(self, path: Path) -> dict:
        if not path.exists():
            return {}
        stamp = path.stat().st_mtime_ns
        if path not in self._journals or self._journals[path][0] != stamp:
            self._journals[path] = (stamp, record(path))
        return self._journals[path][1]

    def __call__(self) -> bool:
        m = memory()
        # Include another explicitly authorized campaign owner if present. This
        # reads its RSS only; no signal, scheduling or resource mutation occurs.
        seen = {os.getpid(), *(v.pid for v in psutil.Process().children(recursive=True))}
        for lock in (N1 / "WRITER.lock", EXEC / "WRITER.lock", OUT / "DRIVER.lock"):
            owner = _owner(lock)
            if owner is not None:
                for process in [owner, *owner.children(recursive=True)]:
                    if process.pid not in seen:
                        try:
                            m["tree_rss"] += process.memory_info().rss
                            seen.add(process.pid)
                        except psutil.NoSuchProcess:
                            pass
        self.peak = max(self.peak, m["tree_rss"])
        self.minimum_available = min(self.minimum_available, m["available"])
        self.minimum_commit = min(self.minimum_commit, m["commit_headroom"])
        reasons = []
        if m["available"] < 2 * 1024**3:
            reasons.append("AVAILABLE_RAM_BELOW_2_GIB")
        if m["tree_rss"] > 4 * 1024**3:
            reasons.append("AUTHORIZED_PROCESS_RSS_EXCEEDS_4_GIB")
        if m["free_disk"] - 1024**3 < 10_000_000_000:
            reasons.append("DISK_AFTER_1_GIB_RESERVATION_BELOW_10_GB")
        if m["commit_headroom"] < 1024**3:
            reasons.append("WINDOWS_COMMIT_HEADROOM_BELOW_1_GIB")
        if self.training:
            window = self._read(OUT / "WINDOW_AUTHORIZATION.json")
            if not window or datetime.now(UTC) >= datetime.fromisoformat(window["deadline_utc"]):
                reasons.append("NOCTURNAL_TRAINING_DEADLINE_REACHED")
            accounting = [
                self._read(path).get("accounting", {})
                for path in (N1 / "PHYSICAL_WORK.json", EXEC / "PHYSICAL_WORK.json")
            ]
            saved = sum(a.get("scientific_saved_updates", 0) for a in accounting)
            lost = sum(a.get("scientific_uncertain_lost_upper", 0) for a in accounting)
            technical = self._read(OUT / "TECHNICAL_WORK.json").get("reserved_updates", 0)
            if saved > 60000 or lost > 6000 or technical > 200:
                reasons.append("UNION_LOGICAL_OR_REPLAY_CAP_EXCEEDED")
            if 60000 + lost + technical > 66200:
                reasons.append("UNION_PHYSICAL_UPPER_CAP_EXCEEDED")
        if reasons or time.monotonic() - self.last >= 5:
            atomic_json(
                EXEC / "RESOURCES.json",
                dict(
                    snapshot=m,
                    reasons=reasons,
                    peak_tree_rss=self.peak,
                    minimum_available=self.minimum_available,
                    minimum_commit_headroom=self.minimum_commit,
                    reservation_bytes=1024**3,
                ),
            )
            self.last = time.monotonic()
        return not reasons

    def check(self) -> None:
        """Fail recoverably without changing the scientific recipe."""
        if not self():
            raise InterruptedError("PAUSED_RESOURCE: execution/RESOURCES.json")


def bound_source(s: CampaignSources, p: dict, row: dict, role: str) -> QuerySource:
    """Bind only verified historical D1/H8 parents and fixed new information mask."""
    from .model import MaskedSource

    if role not in {"inner_oof", "outer_dev"}:
        raise ValueError("only TRAIN and OLD_DEV roles are authorized")
    parent = s.source(historical_spec(s, row["fold"], 8), role)
    key = "train" if role == "inner_oof" else "dev"
    expected = p["sources"][str(row["fold"])]
    if parent.identity_sha256 != expected[f"parent_{key}_sha256"]:
        raise ValueError("historical H8 parent source changed")
    wrapped = MaskedSource(parent, row["arm"])
    if wrapped.identity_sha256 != expected["wrapped"][row["arm"]][f"{key}_sha256"]:
        raise ValueError("prospective masked source changed")
    return wrapped


@contextmanager
def numeric_binding(arm: str) -> Iterator[None]:
    """Scope two factory globals in this separate worker, preserving engine bytes."""
    from e_jepa_ttc.simplex_t import training
    from e_jepa_ttc.simplex_t.model import TemporalConfig

    from .model import build_model, training_loss

    factory, loss = training.TemporalRefiner, training.training_loss

    def model_factory(config: TemporalConfig) -> nn.Module:
        if config != TemporalConfig(feature_count=17, hidden=160):
            raise ValueError("new study requires canonical GRU160 config envelope")
        return build_model(arm)

    def objective(
        output: dict[str, Tensor],
        truth: Tensor,
        experts: Tensor,
        mass: Tensor,
        population: int,
        *,
        selector_only: bool = False,
    ) -> Tensor:
        if selector_only:
            raise ValueError("no selector objective in prospective study")
        return training_loss(output, truth, experts, mass, population)

    # These are deliberately replaceable engine globals, not scientific source
    # edits. The process owns no other active fit; restore even on an exception.
    training.TemporalRefiner = model_factory  # type: ignore[assignment]
    training.training_loss = objective
    try:
        yield
    finally:
        training.TemporalRefiner, training.training_loss = factory, loss


def fit_arm(
    source: QuerySource,
    arm: str,
    output: Path,
    *,
    seed: int,
    freeze_sha256: str,
    resource_ok: Callable[[], bool],
    resume: bool = False,
    stop_after: int = 2500,
    journal: FitJournal | None = None,
) -> dict:
    """Run the historical batch128 FP32/scheduler/checkpoint loop with new head/loss."""
    from e_jepa_ttc.simplex_t.model import TemporalConfig
    from e_jepa_ttc.simplex_t.training import fit

    if arm not in ARM_ORDER or seed != 7:
        raise ValueError("only the fixed seed7 prospective matrix is authorized")
    with numeric_binding(arm):
        return fit(
            source,
            TemporalConfig(feature_count=17, hidden=160),
            output,
            seed=seed,
            freeze_sha256=freeze_sha256,
            resource_ok=resource_ok,
            resume=resume,
            stop_after=stop_after,
            journal=journal,
            device="cpu",
        )


def load_new_endpoint(row: dict, p: dict, pin: str) -> tuple[nn.Module, dict]:
    """Admit a sealed endpoint without opening any OLD_DEV inputs or scores."""
    from e_jepa_ttc.simplex_t.training import load_checkpoint

    from .model import build_model

    if {k: row[k] for k in ("key", "arm", "family", "fold", "seed", "updates")} not in ids():
        raise ValueError("endpoint outside fixed new-study graph")
    path = EXEC / row["checkpoint"]
    if (
        not path.resolve().is_relative_to(EXEC.resolve())
        or digest(path) != row["checkpoint_sha256"]
    ):
        raise ValueError("new endpoint path/hash differs")
    state = load_checkpoint(path)
    expected_source = p["sources"][str(row["fold"])]["wrapped"][row["arm"]]["train_sha256"]
    identity = state["identity"]
    if (
        state["status"] != "COMPLETED"
        or state["completed_updates"] != 2500
        or state["identity_sha256"] != row["identity_sha256"]
        or row["train_source_sha256"] != expected_source
        or identity["source"] != expected_source
        or identity["freeze"] != pin
        or identity["seed"] != 7
        or identity["device"] != "cpu"
        or identity["batch"] != 128
        or identity["endpoint"] != 2500
        or identity["config"]
        != dict(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    ):
        raise ValueError("new endpoint training identity differs")
    model = build_model(row["arm"]).float()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state


def admit_n1(p: dict) -> None:
    """Keep N1 priority, or preserve a classified independent branch blockage.

    The caller verifies every shared input pin before this admission. Unknown
    non-resource errors and any shared-integrity failure never admit descendants.
    This records recovery state without changing N1 files or authorizing updates.
    """
    from e_jepa_ttc.simplex_t.training import load_checkpoint

    if _owner(N1 / "WRITER.lock") is not None:
        raise RuntimeError("N1 has a live owner; never start a second heavy trainer")
    seal = N1 / "ENDPOINTS.json"
    if seal.exists() and len(record(seal)["fits"]) == 6:
        return
    block_path = OUT / "BLOCK_N1_TRAIN.json"
    block = record(block_path)
    if (
        block.get("kind") not in {"NON_SHARED_FAMILY_SPECIFIC", "RESOURCE_REJECTION"}
        or block.get("shared_source_integrity_failed") is True
    ):
        raise ValueError("N1 incomplete without an explicitly classified independent blockage")
    parent = record(N1 / "PROTOCOL.json")
    if digest(N1 / "PROTOCOL.json") != p["parent_h16_protocol_sha256"]:
        raise ValueError("independent N1 parent identity changed")
    preserved = []
    for row in parent["fits"]:
        path = N1 / "fits" / row["key"] / "checkpoint_last.pt"
        if path.exists():
            state = load_checkpoint(path)
            if state["identity"]["freeze"] != p["parent_h16_protocol_sha256"]:
                raise ValueError("blocked N1 checkpoint protocol differs")
            preserved.append(
                dict(
                    key=row["key"],
                    path=str(path),
                    sha256=digest(path),
                    completed_updates=state["completed_updates"],
                    status=state["status"],
                    identity_sha256=state["identity_sha256"],
                )
            )
        else:
            preserved.append(
                dict(key=row["key"], checkpoint=None, completed_updates=0, status="NOT_STARTED")
            )
    status_path, journal_path = N1 / "STATUS.json", N1 / "PHYSICAL_WORK.json"
    publish_json(
        EXEC / f"N1_BLOCK_ADMISSION_{digest(block_path)[:12]}.json",
        dict(
            status="N1_INDEPENDENT_BLOCK_PRESERVED_H8_SHARED_PINS_VERIFIED",
            block=block,
            block_sha256=digest(block_path),
            checkpoints=preserved,
            n1_status=record(status_path) if status_path.exists() else dict(status="NOT_STARTED"),
            n1_status_sha256=digest(status_path) if status_path.exists() else None,
            n1_work_journal=record(journal_path) if journal_path.exists() else None,
            n1_work_journal_sha256=digest(journal_path) if journal_path.exists() else None,
            shared_input_pins_verified=True,
            n1_files_modified=False,
            optimizer_updates_for_admission=0,
        ),
    )


def main(family: str) -> int:
    """Execute one fixed family; N3 requires the prior durable N2 seal."""
    from e_jepa_ttc.simplex_t.physical_accounting import audit_physical_work
    from e_jepa_ttc.simplex_t.training import learning_rate, load_checkpoint
    from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget

    p, pin = protocol()
    if family not in {"N2", "N3"}:
        raise ValueError("explicit N2 or N3 family required")
    resources = Resources(training=True)
    resources.check()
    validate_pins(p, full=True)
    admit_n1(p)
    proof = record(OUT / "verification/ENGINE_RESUME.json")
    if (
        proof["status"] != "EXACT_SYNTHETIC_RESUME_EQUIVALENCE"
        or proof["optimizer_updates"] != 80
        or proof["protocol_sha256"] != pin
    ):
        raise ValueError("prospective synthetic engine/resume proof required")
    graph = {row["key"]: 2500 for row in ids()}
    with Lease():
        budget = WorkBudget(EXEC / "PHYSICAL_WORK.json", graph, technical_reserved=0)
        s = sources(p)
        sealed: list[dict] = []
        prior_seal = EXEC / "N2_ENDPOINTS.json"
        if family == "N3":
            n2 = record(prior_seal)
            if n2["protocol_sha256"] != pin or len(n2["fits"]) != 12:
                raise ValueError("complete matching N2 seal required before N3")
            for endpoint in n2["fits"]:
                model, state = load_new_endpoint(endpoint, p, pin)
                del model, state
            sealed.extend(n2["fits"])
        for row in (v for v in ids() if v["family"] == family):
            resources.check()
            validate_pins(p, full=False)
            source = bound_source(s, p, row, "inner_oof")
            folder = EXEC / "fits" / row["key"]
            path = folder / "checkpoint_last.pt"
            prior = load_checkpoint(path) if path.exists() else None

            class ProgressJournal(EngineWorkJournal):
                def before_update(self, completed: int) -> None:
                    super().before_update(completed)
                    if completed % 25 == 0:
                        atomic_json(
                            EXEC / "UPDATE_PROGRESS.json",
                            dict(
                                current_fit=self.key,
                                next_id=self.key,
                                confirmed_completed_updates=completed,
                                durable_checkpoint_updates=self.saved,
                                endpoint_count=len(sealed),
                                protocol_sha256=pin,
                            ),
                        )

            if prior is None or prior["status"] != "COMPLETED":
                result = fit_arm(
                    source,
                    row["arm"],
                    folder,
                    seed=7,
                    freeze_sha256=pin,
                    resource_ok=resources,
                    resume=prior is not None,
                    journal=ProgressJournal(budget, row["key"]),
                )
                if result["status"] != "COMPLETED":
                    atomic_json(
                        EXEC / "STATUS.json",
                        dict(**result, next_id=row["key"], endpoint_count=len(sealed)),
                    )
                    return 3
            state = load_checkpoint(path)
            endpoint = dict(
                **row,
                checkpoint=str(path.relative_to(EXEC)),
                checkpoint_sha256=digest(path),
                identity_sha256=state["identity_sha256"],
                train_source_sha256=source.identity_sha256,
                model=state["identity"]["config"],
                completed_updates=2500,
            )
            model, verified = load_new_endpoint(endpoint, p, pin)
            EngineWorkJournal(budget, row["key"]).start(2500, path)
            curves = "update,loss,learning_rate,sampler_sha256\n" + "".join(
                f"{i + 1},{loss:.17g},{learning_rate(i + 1):.17g},{state['sampler_hashes'][i]}\n"
                for i, loss in enumerate(state["losses"])
            )
            atomic_bytes(folder / "TRAINING_CURVE.csv", curves.encode())
            sealed.append(endpoint)
            atomic_json(
                EXEC / "ENDPOINT_PROGRESS.json",
                dict(
                    protocol_sha256=pin,
                    fits=sealed,
                    endpoint_count=len(sealed),
                    scientific_saved_updates=record(budget.path)["accounting"][
                        "scientific_saved_updates"
                    ],
                    all_outputs_not_evaluated=True,
                ),
            )
            count = 12 if row["family"] == "N2" else 6
            family_rows = [v for v in sealed if v["family"] == row["family"]]
            if len(family_rows) == count:
                publish_json(
                    EXEC / f"{row['family']}_ENDPOINTS.json",
                    dict(
                        schema="prospective_cost_context_endpoints_v1",
                        family=row["family"],
                        protocol_sha256=pin,
                        fits=family_rows,
                        all_family_frozen_before_evaluation=True,
                        old_dev_scores_read=False,
                        scientific_saved_updates=count * 2500,
                    ),
                )
            print(
                json.dumps(
                    dict(
                        status="ENDPOINT2500_VERIFIED",
                        key=row["key"],
                        endpoint_count=len(sealed),
                        saved_updates=len(sealed) * 2500,
                    )
                ),
                flush=True,
            )
            del source, state, model, verified, prior
            s.release()
            gc.collect()
        validate_pins(p, full=True)
        accounting = audit_physical_work(
            record(budget.path), expected_graph=graph, technical_reserved=0, resource_ok=resources
        )
        atomic_json(EXEC / "ACCOUNTING.json", accounting)
        atomic_json(
            EXEC / "STATUS.json",
            dict(
                status=f"{family}_ALL_ENDPOINTS_SEALED",
                endpoint_count=len(sealed),
                scientific_saved_updates=accounting["saved_updates"],
                technical_optimizer_updates=80,
            ),
        )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("N2", "N3"), required=True)
    raise SystemExit(main(parser.parse_args().family))
