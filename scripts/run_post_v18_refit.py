"""Preflight or explicitly authorized CPU refit of one frozen-parent V16 head.

The default command only reads caches. No authorization record is created here.
Training requires a separate, human-authorized receipt bound to the plan hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
V16 = WORKSPACE / "e-jepa-ttc-v16-multimodal-validation"
DEFAULT_PLAN = ROOT / "docs/post_v18_20261017/REFIT_PLAN.json"
DESTINATION = ROOT / "artifacts/post_v18_refit/PHASE_ONLY_HV_S7"


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash bytes without loading a checkpoint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path: Path, expected: str) -> Path:
    """Refuse altered or absent dependencies."""
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Dependency mismatch: {path}")
    return path


def local_path(value: str) -> Path:
    """Relocate historical absolute bindings by their worktree name."""
    parts = value.replace("\\", "/").split("/")
    starts = [i for i, part in enumerate(parts) if part.startswith("e-jepa-ttc")]
    if not starts:
        raise ValueError(f"No worktree in binding: {value}")
    path = WORKSPACE.joinpath(*parts[starts[-1] :]).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("Dependency escapes workspace")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a durable JSON record within the isolated run."""
    temporary = path.with_suffix(f".{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def group_mass(groups: Any) -> Any:
    """Return equal group mass, preserving equal query mass within each group."""
    import numpy as np

    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (1.0 / (len(unique) * counts[inverse])).astype(np.float32)


def learning_rate(update: int, plan: dict[str, Any]) -> float:
    """Use one-based committed updates for the registered warmup and cosine."""
    total = plan["sampling"]["updates"]
    schedule = plan["scheduler"]
    warmup = schedule["warmup_updates"]
    if not 1 <= update <= total:
        raise ValueError("Update outside fixed endpoint")
    if update <= warmup:
        return schedule["lr_max"] * update / warmup
    fraction = (update - warmup) / (total - warmup)
    return (
        schedule["lr_min"]
        + (schedule["lr_max"] - schedule["lr_min"]) * (1 + math.cos(math.pi * fraction)) / 2
    )


def preflight(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify genealogy and both actual caches without importing/running a model."""
    import numpy as np

    plan = read(plan_path)
    if (plan["family"], plan["seed"], plan["model"]["mode"]) != ("PHASE_ONLY_V16", 7, "EVENT"):
        raise ValueError("This adapter implements PHASE_ONLY EVENT seed7 only")
    if plan["execution_enabled"] or plan["authorized_optimizer_updates"] != 0:
        raise ValueError("Keep the proposal unmodified; supply a separate authorization receipt")
    if plan["stacking"]["OOF"] or set(plan["training_inputs"]) != {"H", "V"}:
        raise ValueError("Expected declared in-sample H+V stacking")
    if (ROOT / plan["export"]["output_root"]).resolve() != DESTINATION.resolve():
        raise ValueError("Historical or alternate destinations are forbidden")
    if not DESTINATION.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Output link escapes this worktree")
    checked(Path(__file__), plan["training_adapter"]["source"]["sha256"])
    freeze = read(
        checked(local_path(plan["parent_freeze"]["path"]), plan["parent_freeze"]["sha256"])
    )
    dependencies: dict[str, str] = {}

    def bind(path: Path, expected: str) -> None:
        checked(path, expected)
        dependencies[path.relative_to(WORKSPACE).as_posix()] = expected

    for name, expected in freeze["source_sha256"].items():
        bind(V16 / name, expected)
    for name, expected in freeze["parent_sha256"].items():
        bind(local_path(name), expected)
    for entry in (
        plan["model"]["source"],
        plan["objective"]["source"],
        plan["objective"]["mass_source"],
    ):
        bind(local_path(entry["path"]), entry["sha256"])
    for name, expected in plan["training_adapter"]["runtime_sources"].items():
        bind(local_path(name), expected)
    arrays: dict[str, list[Any]] = {}
    role_summary = {}
    for role, item in plan["training_inputs"].items():
        path = local_path(item["path"])
        bind(path, item["expected_sha256"])
        seal_path = local_path(item["seal"]["path"])
        bind(seal_path, item["seal"]["sha256"])
        seal = read(seal_path)
        binding_path = path.parent / "BINDING.json"
        bind(binding_path, seal["binding_sha256"])
        binding = read(binding_path)
        if seal["states_sha256"] != item["expected_sha256"] or seal["population"] != item["rows"]:
            raise ValueError("Cache/seal mismatch")
        for source in binding["inputs"].values():
            bind(local_path(source["path"]), source["sha256"])
        for name, expected in binding["sources"].items():
            bind(V16 / Path(name.replace("\\", "/")), expected)
        if not set(binding["parents"].values()) <= set(freeze["parent_sha256"].values()):
            raise ValueError("Cache parents differ from candidate genealogy")
        with np.load(path, allow_pickle=False) as archive:
            values = {
                name: archive[name].copy()
                for name in (
                    "features",
                    "rgb_width_logits",
                    "target_phase",
                    "mass",
                    "sample_token",
                    "group_id",
                    "role",
                )
            }
        rows = item["rows"]
        shapes = {"features": (rows, 340), "rgb_width_logits": (rows, 2)}
        for name, value in values.items():
            if value.shape != shapes.get(name, (rows,)):
                raise ValueError(f"Bad {role}/{name} shape")
            if value.dtype.kind == "f" and (
                value.dtype != np.float32 or not np.isfinite(value).all()
            ):
                raise ValueError(f"Nonfinite or non-FP32 {role}/{name}")
        if set(values["role"].tolist()) != {role}:
            raise ValueError("Mixed cache roles")
        if not np.allclose(values["mass"], group_mass(values["group_id"]), rtol=1e-6, atol=0):
            raise ValueError("Original group mass differs")
        role_summary[role] = {"rows": rows, "groups": len(np.unique(values["group_id"]))}
        for name, value in values.items():
            arrays.setdefault(name, []).append(value)
    if set(arrays["group_id"][0]) & set(arrays["group_id"][1]):
        raise ValueError("H/V group leakage")
    data = {name: np.concatenate(parts) for name, parts in arrays.items()}
    population = len(data["sample_token"])
    if len(np.unique(data["sample_token"])) != population:
        raise ValueError("Duplicate query IDs")
    if population != plan["sampling"]["population"]:
        raise ValueError("Wrong union population")
    data["mass"] = group_mass(data["group_id"])
    if plan["sampling"]["updates"] != math.ceil(2500 * population / 18422):
        raise ValueError("Exposure differs from registered scaling")
    budget = plan["budget_proposed_only"]
    if (
        budget["scientific_updates"],
        budget["physical_update_cap"],
        budget["GPU_seconds"],
        budget["technical_updates"],
        budget["recovery_update_cap"],
    ) != (4730, 4980, 0, 0, 250):
        raise ValueError("Unexpected budget")
    return (
        plan,
        data,
        {
            "status": "PASS_ACTUAL_CACHES_AND_GENEALOGY",
            "plan_sha256": sha(plan_path),
            "dependencies": dependencies,
            "roles": role_summary,
            "population": population,
            "groups": len(np.unique(data["group_id"])),
            "mass_sum": float(data["mass"].sum(dtype=np.float64)),
            "optimizer_updates": 0,
            "model_inferences": 0,
            "training_started": False,
        },
    )


def authorize(plan_path: Path, receipt_path: Path | None) -> dict[str, Any]:
    """Require an explicit separate authorization; never generate it implicitly."""
    if receipt_path is None:
        raise PermissionError(
            "Zero updates authorized. A separate explicit authorization receipt is required."
        )
    receipt = read(receipt_path)
    plan = read(plan_path)
    if (
        receipt.get("schema") != "post_v18_refit_authorization_v1"
        or receipt.get("plan_sha256") != sha(plan_path)
        or receipt.get("authorized") is not True
        or not str(receipt.get("user_instruction_reference", "")).strip()
        or receipt.get("budget") != plan["budget_proposed_only"]
    ):
        raise PermissionError("Authorization must bind explicit instruction, exact plan and budget")
    return receipt


def load_runtime() -> tuple[Any, Any, Any]:
    """Import the frozen V16 implementation, refusing mixed-worktree modules."""
    sys.path[:0] = [str(V16), str(V16 / "src")]
    names = (
        "operational.multimodal_research.phase_only_anchor",
        "e_jepa_ttc.rgb_port.fusion",
        "operational.rgb_port.train_producers",
    )
    modules = [importlib.import_module(name) for name in names]
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if name.startswith(("operational.", "e_jepa_ttc.")) and isinstance(source, str):
            if not Path(source).resolve().is_relative_to(V16.resolve()):
                raise RuntimeError(
                    f"Mixed worktree import: {name}; run this adapter in a fresh process"
                )
    return modules[0].PhaseOnlyAnchor, modules[1].fusion_loss, modules[2].ProducerCheckpoint


def enforce_updates(state: Any, plan: dict[str, Any], *, next_update: bool) -> None:
    """Apply this plan's smaller recovery cap over the reused checkpoint journal."""
    budget = plan["budget_proposed_only"]
    if state.recovery_upper > budget["recovery_update_cap"]:
        raise RuntimeError("Refit recovery cap exhausted")
    if state.completed + state.recovery_upper + int(next_update) > budget["physical_update_cap"]:
        raise RuntimeError("Refit physical update cap exhausted")


def export_head(
    model: Any, data: dict[str, Any], directory: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    """Export only the head and five fixed TRAIN examples, checking saved parity."""
    import numpy as np
    import torch

    class PointHead(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.head = head

        def forward(self, features: Any, widths: Any) -> Any:
            return self.head(features, widths)["ttc"]

    ids = np.argsort(data["sample_token"], kind="stable")[:5]
    features = torch.from_numpy(data["features"][ids])
    widths = torch.from_numpy(data["rgb_width_logits"][ids])
    wrapper = PointHead(model.eval())
    with torch.no_grad():
        expected = wrapper(features, widths)
        traced = torch.jit.trace(wrapper, (features, widths))
        temporary = directory / "head.pending.pt"
        with temporary.open("wb") as stream:
            torch.jit.save(traced, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            loaded = torch.jit.load(stream, map_location="cpu")
        actual = loaded(features, widths)
    error = float((actual - expected).abs().max())
    if not torch.isfinite(actual).all() or error > 1e-5:
        raise ValueError("Saved TorchScript head fails TTC parity")
    os.replace(temporary, directory / "head.pt")
    np.savez(
        directory / "TRAIN_EXAMPLES.npz",
        features=features.numpy(),
        rgb_width_logits=widths.numpy(),
        ttc=expected.numpy(),
        sample_token=data["sample_token"][ids],
    )
    result = {
        "identity": identity,
        "scope": "head_only; frozen upstream pipeline required",
        "input_shapes": [[5, 340], [5, 2]],
        "output": "signed TTC seconds [5]",
        "examples_role": "TRAIN_HV",
        "max_abs_ttc_error_seconds": error,
        "head_sha256": sha(directory / "head.pt"),
        "examples_sha256": sha(directory / "TRAIN_EXAMPLES.npz"),
        "test12_inference": False,
        "trace_contract": (
            "Validated fixed batch5 FP32 finite inputs only. Original Python shape and "
            "finiteness guards are not preserved by tracing; callers must validate inputs."
        ),
    }
    atomic_json(directory / "EXPORT.json", result)
    return result


def train(plan_path: Path, receipt_path: Path | None, *, resume: bool) -> None:
    """Run the fixed endpoint only after separately authorized, verified preflight."""
    receipt = authorize(plan_path, receipt_path)  # Before imports, cache reads or writes.
    plan, data, evidence = preflight(plan_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This process only.
    import numpy as np
    import psutil
    import torch

    anchor, loss_fn, checkpoint = load_runtime()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    random.seed(plan["seed"])
    runtime_sources = {}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if isinstance(source, str):
            path = Path(source).resolve()
            if path.is_relative_to(V16.resolve()) and path.suffix == ".py":
                runtime_sources[path.relative_to(V16).as_posix()] = sha(path)
    identity = {
        "fit_id": "PHASE_ONLY_HV_S7",
        "plan_sha256": sha(plan_path),
        "adapter_sha256": sha(Path(__file__)),
        "dependencies": evidence["dependencies"],
        "runtime_sources": runtime_sources,
        "authorization": receipt,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "host": platform.node(),
            "device": "cpu",
        },
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if resume:
        if not (DESTINATION / "CHECKPOINT_POINTER.json").is_file():
            raise ValueError("No durable checkpoint to resume")
        if read(DESTINATION / "IDENTITY.json") != identity:
            raise ValueError("Resume identity changed")
    else:
        DESTINATION.mkdir(parents=True, exist_ok=False)
        atomic_json(DESTINATION / "IDENTITY.json", identity)
        atomic_json(DESTINATION / "INPUTS.json", evidence)
    # A crashed run leaves the lock as evidence; removing a stale lock is a separate manual action.
    lock = DESTINATION / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        model = anchor("EVENT").cpu()
        if sum(p.numel() for p in model.parameters()) != plan["model"]["parameters"]:
            raise ValueError("Head capacity changed")
        config = plan["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=tuple(config["betas"]),
            eps=config["eps"],
            foreach=config["foreach"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        generator = torch.Generator(device="cpu").manual_seed(plan["seed"])
        state = checkpoint(DESTINATION, identity, plan["sampling"]["updates"])
        cursor: dict[str, Any] = {"losses": []}
        if resume:
            cursor = state.restore(model, optimizer, scheduler, generator)
        else:
            state.save(model, optimizer, scheduler, generator, cursor, status="READY")
        enforce_updates(state, plan, next_update=False)
        values = {
            name: torch.from_numpy(data[name])
            for name in ("features", "rgb_width_logits", "target_phase", "mass")
        }
        ledger_path = DESTINATION / "CPU_BUDGET.json"
        ledger = read(ledger_path) if ledger_path.exists() else {"reserved_wall_seconds": 0}
        deadline = 0.0

        def resource_guard() -> None:
            nonlocal deadline
            if psutil.Process().memory_info().rss > plan["budget_proposed_only"]["RAM_bytes"]:
                raise MemoryError("Refit RAM guard exceeded")
            if time.monotonic() >= deadline:
                if (
                    ledger["reserved_wall_seconds"] + 60
                    > plan["budget_proposed_only"]["CPU_wall_seconds"]
                ):
                    raise RuntimeError("Refit CPU wall budget exhausted")
                ledger["reserved_wall_seconds"] += 60
                atomic_json(ledger_path, ledger)
                deadline = time.monotonic() + 60

        try:
            while state.completed < state.updates:
                resource_guard()
                enforce_updates(state, plan, next_update=True)
                ids = torch.randint(
                    len(data["mass"]), (plan["sampling"]["batch_size"],), generator=generator
                )
                optimizer.param_groups[0]["lr"] = learning_rate(state.completed + 1, plan)
                optimizer.zero_grad(set_to_none=True)
                output = model(values["features"][ids], values["rgb_width_logits"][ids])
                loss = loss_fn(
                    output, values["target_phase"][ids], values["mass"][ids], len(data["mass"])
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite refit loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["clip_norm"], error_if_nonfinite=True
                )
                state.begin()
                optimizer.step()
                scheduler.step()
                state.commit()
                cursor["losses"].append(float(loss.detach()))
                if state.completed % plan["checkpoint"]["every_updates"] == 0:
                    state.save(model, optimizer, scheduler, generator, cursor, status="RUNNING")
                    print(
                        json.dumps(
                            {"completed_updates": state.completed, "loss": cursor["losses"][-1]}
                        ),
                        flush=True,
                    )
            state.save(model, optimizer, scheduler, generator, cursor, status="COMPLETE")
            resource_guard()
            export_head(model, data, DESTINATION, identity)
        except BaseException:
            # Retain the last durable sampler/RNG state even if the failed batch drew IDs.
            # The journal accounts for replay of committed and possibly pending updates.
            raise
    finally:
        lock.unlink()


def main() -> None:
    """Expose read-only preflight and separately authorized execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "train"), nargs="?", default="dry-run")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--authorization", type=Path, help="Separate receipt; never produced by this script"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Require an exact existing run identity"
    )
    args = parser.parse_args()
    if args.action == "dry-run":
        if args.authorization or args.resume:
            parser.error("dry-run accepts neither authorization nor resume")
        _, _, evidence = preflight(args.plan)
        print(json.dumps(evidence, indent=2))
    else:
        train(args.plan, args.authorization, resume=args.resume)


if __name__ == "__main__":
    main()
