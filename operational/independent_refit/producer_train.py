"""Real, fail-closed E_A5 fitter for one independent TRAIN24 rotation.

This is a supervised producer fit: TTC, risk and training-box geometry losses
are active, with frozen DINO relational targets as an auxiliary objective.  It
must not be described as SSL.  No historical model checkpoint or learned
normalizer is accepted by this API.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import psutil

from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_eap import (
    _autocast,
    _foreground_only_loss_config,
    _loss,
)
from operational.train40_system.checkpoint import payload_digest

from .fitter import state_tensor_sha256
from .producer_contract import (
    canonical_sha256,
    file_sha256,
    verify_producer_plan,
)
from .producer_inputs import (
    CROSSWALK_PARQUET_SHA256,
    CROSSWALK_RECEIPT_SHA256,
    INPUT_MANIFEST_SHA256,
    PREPARE_FREEZE_SHA256,
    open_event_producer_source,
    verify_h5_admission_for_fit,
)
from .teacher_cache import load_teacher_role

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

torch: Any = importlib.import_module("torch")


class GeneratorLike(Protocol):
    """Minimal saved CPU generator surface used by the fitter."""

    def get_state(self) -> object: ...

    def set_state(self, state: object) -> None: ...

RUN_SCHEMA = "independent_phase_e_a5_run_contract_v1"
CHECKPOINT_SCHEMA = "independent_phase_e_a5_checkpoint_v1"
JOURNAL_SCHEMA = "independent_phase_e_a5_update_journal_v1"
SUMMARY_SCHEMA = "independent_phase_e_a5_summary_v1"
CHECKPOINT_INTERVAL = 100
RECOVERY_UPPER_LIMIT = 100
HOST_AVAILABLE_GATE_BYTES = 8 * 1024**3
GPU_FREE_GATE_BYTES = 4 * 1024**3
RUN_PARENT_KEYS = {
    "h5_admission",
    "input_manifest",
    "prepare_freeze",
    "crosswalk_receipt",
    "crosswalk",
    "role_mask",
    "teacher_manifest",
    "teacher_role_receipt",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _idempotent_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _json(path) != dict(value):
            raise ValueError(f"Existing durable artifact differs: {path}")
        return
    _atomic_json(path, value)


def _safe_numpy_state() -> dict[str, Any]:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "name": str(name),
        "keys": np.asarray(keys, dtype=np.uint32).astype(np.uint64).tolist(),
        "position": int(position),
        "has_gaussian": int(gaussian),
        "cached_gaussian": float(cached),
    }


def _restore_numpy_state(value: Mapping[str, Any]) -> None:
    np.random.set_state(
        (
            str(value["name"]),
            np.asarray(value["keys"], dtype=np.uint32),
            int(value["position"]),
            int(value["has_gaussian"]),
            float(value["cached_gaussian"]),
        )
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "operational/independent_refit/producer_train.py",
        "operational/independent_refit/producer_contract.py",
        "operational/independent_refit/producer_inputs.py",
        "src/e_jepa_ttc/models/causal_scale_ttc.py",
        "src/e_jepa_ttc/data/object_event_v4.py",
        "src/e_jepa_ttc/training/causal_scale_eap.py",
        "src/e_jepa_ttc/losses/causal_scale_ttc.py",
    )
    return {name: file_sha256(root / name) for name in names}


def build_run_contract(
    *,
    plan: Mapping[str, Any],
    h5_admission_path: Path,
    input_manifest_path: Path,
    prepare_freeze_path: Path,
    crosswalk_receipt_path: Path,
    crosswalk_path: Path,
    teacher_manifest_path: Path,
    teacher_role_receipt_path: Path,
) -> dict[str, Any]:
    """Bind every admitted source without accepting a learned parent."""

    verify_producer_plan(plan)
    if (
        plan.get("component") != "E_A5"
        or plan.get("role") != "producer"
        or plan.get("drop_last") is not False
        or plan.get("producer_queries") != 35_048
        or plan.get("updates_per_epoch") != 1_096
        or plan.get("fixed_updates") != 19_728
        or cast(Mapping[str, Any], plan["recipe"]).get("learned_parent_checkpoints")
    ):
        raise ValueError("Independent E_A5 fixed rotation-0 endpoint differs")
    h5 = _json(h5_admission_path)
    if (
        h5.get("status") != "PHYSICAL_H5_HASHES_AND_BOUNDS_VERIFIED"
        or h5.get("training_admitted") is not True
        or h5.get("full_hashes_computed") is not True
    ):
        raise RuntimeError("Independent E_A5 requires completed physical H5 admission")
    paths = {
        "h5_admission": h5_admission_path,
        "input_manifest": input_manifest_path,
        "prepare_freeze": prepare_freeze_path,
        "crosswalk_receipt": crosswalk_receipt_path,
        "crosswalk": crosswalk_path,
        "role_mask": Path(str(cast(Mapping[str, Any], plan["history_role_mask"])["path"])),
        "teacher_manifest": teacher_manifest_path,
        "teacher_role_receipt": teacher_role_receipt_path,
    }
    unsigned: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "FROZEN_NOT_STARTED",
        "scientific_mode": "SUPERVISED_TTC_BBOX_PLUS_FROZEN_DINO_AUXILIARY",
        "ssl_claim_allowed": False,
        "rotation": 0,
        "seed": 7,
        "epochs": 18,
        "batch_size": 32,
        "drop_last": False,
        "producer_queries": 35_048,
        "final_batch_size": 8,
        "updates_per_epoch": 1_096,
        "updates_limit": 19_728,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "recovery_upper_limit": RECOVERY_UPPER_LIMIT,
        "checkpoint_selection": "FIXED_LAST_EPOCH_NO_METRIC_SELECTION",
        "initialization": "FRESH_SEED7_NO_LEARNED_PARENT",
        "historical_weights_allowed": False,
        "historical_normalizers_allowed": False,
        "plan_sha256": str(plan["plan_sha256"]),
        "plan": dict(plan),
        "parents": {
            name: {"path": str(path.resolve(strict=True)), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "executed_source_sha256": _source_hashes(),
    }
    unsigned["contract_sha256"] = canonical_sha256(unsigned)
    verify_run_contract(unsigned)
    return unsigned


def verify_run_contract(value: Mapping[str, Any]) -> None:
    """Verify the complete immutable run identity."""

    unsigned = dict(value)
    claimed = unsigned.pop("contract_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("Independent E_A5 run identity changed")
    if (
        value.get("schema") != RUN_SCHEMA
        or value.get("status") != "FROZEN_NOT_STARTED"
        or value.get("scientific_mode")
        != "SUPERVISED_TTC_BBOX_PLUS_FROZEN_DINO_AUXILIARY"
        or value.get("ssl_claim_allowed") is not False
        or value.get("rotation") != 0
        or value.get("seed") != 7
        or value.get("epochs") != 18
        or value.get("batch_size") != 32
        or value.get("drop_last") is not False
        or value.get("producer_queries") != 35_048
        or value.get("final_batch_size") != 8
        or value.get("updates_per_epoch") != 1_096
        or value.get("updates_limit") != 19_728
        or value.get("checkpoint_selection") != "FIXED_LAST_EPOCH_NO_METRIC_SELECTION"
        or value.get("historical_weights_allowed") is not False
        or value.get("historical_normalizers_allowed") is not False
    ):
        raise ValueError("Independent E_A5 run contract differs")
    plan = cast(Mapping[str, Any], value["plan"])
    verify_producer_plan(plan)
    if plan.get("plan_sha256") != value.get("plan_sha256"):
        raise ValueError("Independent E_A5 plan binding differs")
    parents = cast(Mapping[str, Mapping[str, Any]], value["parents"])
    if set(parents) != RUN_PARENT_KEYS:
        raise ValueError("Independent E_A5 run parent set differs")
    for reference in parents.values():
        if file_sha256(Path(str(reference["path"]))) != reference.get("sha256"):
            raise ValueError("Independent E_A5 parent changed")
    expected_hashes = {
        "input_manifest": INPUT_MANIFEST_SHA256,
        "prepare_freeze": PREPARE_FREEZE_SHA256,
        "crosswalk_receipt": CROSSWALK_RECEIPT_SHA256,
        "crosswalk": CROSSWALK_PARQUET_SHA256,
        "role_mask": cast(Mapping[str, Any], plan["history_role_mask"])["sha256"],
        "teacher_manifest": cast(Mapping[str, str], plan["parent_sha256"])[
            "teacher_manifest"
        ],
        "teacher_role_receipt": cast(Mapping[str, str], plan["parent_sha256"])[
            "teacher_role"
        ],
    }
    if any(parents[name].get("sha256") != digest for name, digest in expected_hashes.items()):
        raise ValueError("Independent E_A5 run parent identity differs")
    if (
        Path(str(parents["role_mask"]["path"])).resolve(strict=True)
        != Path(str(cast(Mapping[str, Any], plan["history_role_mask"])["path"])).resolve(
            strict=True
        )
        or Path(str(parents["teacher_manifest"]["path"])).resolve(strict=True)
        != Path(str(cast(Mapping[str, str], plan["parent_paths"])["teacher_manifest"])).resolve(
            strict=True
        )
        or Path(str(parents["teacher_role_receipt"]["path"])).resolve(strict=True)
        != Path(str(cast(Mapping[str, str], plan["parent_paths"])["teacher_role"])).resolve(
            strict=True
        )
    ):
        raise ValueError("Independent E_A5 run parent path differs")
    verify_h5_admission_for_fit(Path(str(parents["h5_admission"]["path"])))
    if _source_hashes() != value.get("executed_source_sha256"):
        raise ValueError("Independent E_A5 executed source changed")


def epoch_order(
    groups: Sequence[Sequence[int]], generator: GeneratorLike, *, population_size: int
) -> list[int]:
    """Reproduce the admitted group/row shuffle with one saved RNG stream."""

    flattened = [int(index) for group in groups for index in group]
    if sorted(flattened) != list(range(population_size)):
        raise ValueError("Producer sampling groups must partition the valid role")
    order: list[int] = []
    for group_index in torch.randperm(len(groups), generator=generator).tolist():
        group = tuple(int(index) for index in groups[group_index])
        for row_index in torch.randperm(len(group), generator=generator).tolist():
            order.append(group[row_index])
    return order


@dataclass
class IndependentDurableState:
    """Atomic full-state checkpoint and conservative physical-update journal."""

    directory: Path
    contract: Mapping[str, Any]
    committed: int = 0
    durable: int = 0
    recovery_upper: int = 0
    pending: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "checkpoint_last.pt"

    @property
    def journal_path(self) -> Path:
        return self.directory / "UPDATE_JOURNAL.json"

    def journal(self) -> None:
        _atomic_json(
            self.journal_path,
            {
                "schema": JOURNAL_SCHEMA,
                "contract_sha256": self.contract["contract_sha256"],
                "committed_updates": self.committed,
                "durable_updates": self.durable,
                "recovery_upper": self.recovery_upper,
                "pending_update_upper": int(self.pending),
            },
        )

    def begin_update(self) -> None:
        if self.pending or self.committed >= int(self.contract["updates_limit"]):
            raise RuntimeError("Independent optimizer boundary is unavailable")
        self.pending = True
        self.journal()

    def commit_update(self) -> None:
        if not self.pending:
            raise RuntimeError("Independent optimizer update was not reserved")
        self.committed += 1
        self.pending = False
        self.journal()

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        sampler: GeneratorLike,
        cursor: Mapping[str, Any],
        initial_state_sha256: str,
        status: str,
    ) -> None:
        if self.pending:
            raise RuntimeError("Cannot checkpoint a pending optimizer update")
        payload: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA,
            "status": status,
            "contract": dict(self.contract),
            "contract_sha256": self.contract["contract_sha256"],
            "initial_state_sha256": initial_state_sha256,
            "model_state_dict": model.state_dict(),
            "model_state_sha256": state_tensor_sha256(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "sampler_generator_state": sampler.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_random_state": _safe_numpy_state(),
            "python_random_state": random.getstate(),
            "cursor": dict(cursor),
            "committed_updates": self.committed,
            "recovery_upper": self.recovery_upper,
        }
        payload["full_state_sha256"] = payload_digest(payload)
        temporary = self.checkpoint_path.with_suffix(".pending.pt")
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, self.checkpoint_path)
        self.durable = self.committed
        self.journal()
        _atomic_json(
            self.directory / "CHECKPOINT_RECEIPT.json",
            {
                "schema": "independent_phase_e_a5_checkpoint_receipt_v1",
                "status": status,
                "contract_sha256": self.contract["contract_sha256"],
                "checkpoint_sha256": file_sha256(self.checkpoint_path),
                "model_state_sha256": payload["model_state_sha256"],
                "full_state_sha256": payload["full_state_sha256"],
                "committed_updates": self.committed,
                "recovery_upper": self.recovery_upper,
                "full_optimizer_scheduler_sampler_and_all_rng": True,
            },
        )

    def restore(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        sampler: GeneratorLike,
        initial_state_sha256: str,
    ) -> dict[str, Any]:
        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise ValueError("Independent checkpoint payload differs")
        expected = saved.pop("full_state_sha256", None)
        if expected != payload_digest(saved):
            raise ValueError("Independent checkpoint full state changed")
        if (
            saved.get("schema") != CHECKPOINT_SCHEMA
            or saved.get("contract") != dict(self.contract)
            or saved.get("contract_sha256") != self.contract["contract_sha256"]
            or saved.get("initial_state_sha256") != initial_state_sha256
            or saved.get("model_state_sha256")
            != state_tensor_sha256(cast(Mapping[str, Any], saved["model_state_dict"]))
        ):
            raise ValueError("Independent checkpoint identity differs")
        journal = _json(self.journal_path)
        if (
            journal.get("schema") != JOURNAL_SCHEMA
            or journal.get("contract_sha256") != self.contract["contract_sha256"]
        ):
            raise ValueError("Independent update journal differs")
        self.committed = int(saved["committed_updates"])
        self.durable = self.committed
        lost = max(0, int(journal["committed_updates"]) - self.committed)
        lost += int(journal["pending_update_upper"])
        self.recovery_upper = max(
            int(saved["recovery_upper"]), int(journal["recovery_upper"])
        ) + lost
        if self.recovery_upper > int(self.contract["recovery_upper_limit"]):
            raise RuntimeError("Independent recovery budget exceeded")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        sampler.set_state(saved["sampler_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if saved["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        _restore_numpy_state(cast(Mapping[str, Any], saved["numpy_random_state"]))
        random.setstate(saved["python_random_state"])
        self.pending = False
        self.journal()
        return cast(dict[str, Any], saved["cursor"])


def _resource_preflight() -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    if available < HOST_AVAILABLE_GATE_BYTES or gpu_free < GPU_FREE_GATE_BYTES:
        raise RuntimeError("Independent E_A5 resource gate failed before source/model allocation")
    return {
        "host_available_bytes": available,
        "gpu_free_bytes": int(gpu_free),
        "gpu_total_bytes": int(gpu_total),
        "host_gate_bytes": HOST_AVAILABLE_GATE_BYTES,
        "gpu_gate_bytes": GPU_FREE_GATE_BYTES,
    }


def _seed_all(seed: int) -> GeneratorLike:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    return torch.Generator(device="cpu").manual_seed(seed)


def _new_model(plan: Mapping[str, Any]) -> tuple[CausalScaleTTC, str]:
    recipe = cast(Mapping[str, Any], plan["recipe"])
    config = CausalScaleTTCConfig(**cast(dict[str, Any], recipe["model_config"]))
    model = CausalScaleTTC(config)
    return model, state_tensor_sha256(model.state_dict())


def _curve_segment(
    run_root: Path, contract_sha256: str, curve: list[dict[str, Any]], update: int
) -> None:
    if not curve:
        return
    _idempotent_json(
        run_root / "curves" / f"curve_{update:06d}.json",
        {
            "schema": "independent_phase_e_a5_curve_v1",
            "contract_sha256": contract_sha256,
            "first_update": curve[0]["update"],
            "last_update": curve[-1]["update"],
            "updates": curve,
        },
    )


def _durable_curve_transaction(
    run_root: Path,
    contract_sha256: str,
    cursor: dict[str, Any],
    checkpoint: Callable[[], None],
) -> None:
    """Bind curve publication to checkpoint state across either crash boundary."""

    pending = cast(list[dict[str, Any]], cursor.get("pending_curve", []))
    pending_update = int(cursor.get("pending_curve_update", 0))
    if pending:
        _curve_segment(run_root, contract_sha256, pending, pending_update)
        cursor["pending_curve"] = []
        cursor["pending_curve_update"] = 0
        checkpoint()
    curve = cast(list[dict[str, Any]], cursor.get("curve", []))
    if not curve:
        return
    cursor["pending_curve"] = list(curve)
    cursor["pending_curve_update"] = int(curve[-1]["update"])
    cursor["curve"] = []
    checkpoint()
    _curve_segment(
        run_root,
        contract_sha256,
        cast(list[dict[str, Any]], cursor["pending_curve"]),
        int(cursor["pending_curve_update"]),
    )
    cursor["pending_curve"] = []
    cursor["pending_curve_update"] = 0
    checkpoint()


def run_e_a5(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    source_kwargs: Mapping[str, Any],
    teacher_manifest_path: Path,
    fold_contract_path: Path,
    teacher_role_receipt_path: Path,
) -> dict[str, Any]:
    """Execute or exactly resume the fixed E_A5 endpoint on CUDA."""

    verify_run_contract(run_contract)
    if not torch.cuda.is_available():
        raise RuntimeError("Independent E_A5 requires an explicitly allocated CUDA device")
    resources = _resource_preflight()
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "RUN_CONTRACT.json"
    if contract_path.exists():
        if _json(contract_path) != dict(run_contract):
            raise ValueError("Existing independent run contract differs")
    else:
        if any(root.iterdir()):
            raise FileExistsError("Independent run output must be fresh")
        _atomic_json(contract_path, run_contract)
    lock = root / "ACTIVE.lock"
    runtime_teacher_role = root / f".TEACHER_ROLE.{os.getpid()}.json"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    os.close(descriptor)
    try:
        generator = _seed_all(7)
        source = open_event_producer_source(**dict(source_kwargs))
        if len(source) != 35_048:
            raise ValueError("Independent E_A5 source population differs")
        teacher = load_teacher_role(
            teacher_manifest_path,
            expected_manifest_sha256=file_sha256(teacher_manifest_path),
            fold_contract_path=fold_contract_path,
            rotation=0,
            role="producer",
            role_receipt_path=runtime_teacher_role,
        )
        if teacher.role_receipt_sha256 != file_sha256(teacher_role_receipt_path):
            raise ValueError("Runtime teacher role differs from the frozen plan parent")
        valid_source_tokens = frozenset(source.index["sample_token"].astype(str))
        if (
            not valid_source_tokens < teacher.allowed_tokens
            or teacher.allowed_tokens - valid_source_tokens
            != source.history_invalid_tokens
            or len(source.history_invalid_tokens) != 68
        ):
            raise ValueError("Teacher role does not equal valid source plus 68 masked tokens")
        model, initial_state = _new_model(plan)
        device = torch.device("cuda")
        model.to(device).train()
        recipe = cast(Mapping[str, Any], plan["recipe"])
        training = cast(Mapping[str, Any], recipe["training_config"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=18, eta_min=float(training["minimum_learning_rate"])
        )
        loss_config = CausalScaleTTCLossConfig(**cast(dict[str, Any], recipe["loss_config"]))
        warmup = _foreground_only_loss_config(loss_config)
        state = IndependentDurableState(root, run_contract)
        cursor: dict[str, Any] = {
            "epoch": 1,
            "batch_position": 0,
            "order": epoch_order(
                source.sampling_groups, generator, population_size=len(source)
            ),
            "curve": [],
            "pending_curve": [],
            "pending_curve_update": 0,
            "loss_sums": {},
            "examples": 0,
        }
        if state.checkpoint_path.exists():
            cursor = state.restore(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=generator,
                initial_state_sha256=initial_state,
            )
            model.to(device).train()
        else:
            state.journal()
            state.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=generator,
                cursor=cursor,
                initial_state_sha256=initial_state,
                status="READY",
            )
        if cursor.get("pending_curve"):
            _durable_curve_transaction(
                root,
                str(run_contract["contract_sha256"]),
                cursor,
                lambda: state.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=generator,
                    cursor=cursor,
                    initial_state_sha256=initial_state,
                    status="RUNNING",
                ),
            )
        while int(cursor["epoch"]) <= 18:
            epoch = int(cursor["epoch"])
            position = int(cursor["batch_position"])
            order = cast(list[int], cursor["order"])
            if len(order) != 35_048 or sorted(order) != list(range(35_048)):
                raise ValueError("Independent no-drop epoch order differs")
            while position < 1_096:
                indices = order[position * 32 : min((position + 1) * 32, len(order))]
                batch = source.batch(indices, teacher).to(device)
                optimizer.zero_grad(set_to_none=True)
                selected_loss = warmup if epoch <= 3 else loss_config
                with _autocast(device, "bf16"):
                    total, components, _ = _loss(
                        model,
                        batch,
                        selected_loss,
                        mask_t0_as_proxy=True,
                        foreground_supervision="bbox_geometry",
                        representation_supervision="dinov3_local_relational",
                        representation_distillation_weight=8.0,
                    )
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("Independent E_A5 loss is non-finite")
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                state.begin_update()
                optimizer.step()
                state.commit_update()
                losses = {
                    name: float(value.detach().float().cpu())
                    for name, value in components.items()
                }
                losses["total"] = float(total.detach().float().cpu())
                position += 1
                cursor["batch_position"] = position
                batch_size = len(indices)
                cursor["examples"] = int(cursor["examples"]) + batch_size
                sums = cast(dict[str, float], cursor["loss_sums"])
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.0) + value * batch_size
                cast(list[dict[str, Any]], cursor["curve"]).append(
                    {
                        "update": state.committed,
                        "epoch": epoch,
                        "batch_position": position,
                        "losses": losses,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                if state.committed % CHECKPOINT_INTERVAL == 0:
                    _durable_curve_transaction(
                        root,
                        str(run_contract["contract_sha256"]),
                        cursor,
                        lambda: state.save(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            sampler=generator,
                            cursor=cursor,
                            initial_state_sha256=initial_state,
                            status="RUNNING",
                        ),
                    )
            scheduler.step()
            cursor["epoch"] = epoch + 1
            cursor["batch_position"] = 0
            cursor["order"] = (
                epoch_order(source.sampling_groups, generator, population_size=len(source))
                if epoch < 18
                else []
            )
            completion_status = "COMPLETE" if epoch == 18 else "RUNNING"
            _durable_curve_transaction(
                root,
                str(run_contract["contract_sha256"]),
                cursor,
                lambda status=completion_status: state.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=generator,
                    cursor=cursor,
                    initial_state_sha256=initial_state,
                    status=status,
                ),
            )
        if state.committed != 19_728 or int(cursor["examples"]) != 630_864:
            raise RuntimeError("Independent E_A5 fixed endpoint was not reached")
        summary: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "status": "COMPLETE_FIXED_LAST_EPOCH",
            "scientific_mode": "SUPERVISED_NOT_SSL",
            "contract_sha256": run_contract["contract_sha256"],
            "logical_updates": state.committed,
            "recovery_upper": state.recovery_upper,
            "examples": cursor["examples"],
            "epochs": 18,
            "checkpoint_path": str(state.checkpoint_path),
            "checkpoint_sha256": file_sha256(state.checkpoint_path),
            "model_state_sha256": state_tensor_sha256(model.state_dict()),
            "initial_state_sha256": initial_state,
            "checkpoint_selection": "FIXED_LAST_EPOCH_NO_METRIC_SELECTION",
            "resource_preflight": resources,
            "mean_training_losses": {
                name: value / int(cursor["examples"])
                for name, value in cast(dict[str, float], cursor["loss_sums"]).items()
            },
        }
        summary["summary_sha256"] = canonical_sha256(summary)
        _idempotent_json(root / "SUMMARY.json", summary)
        return summary
    except BaseException as error:
        _atomic_json(
            root / "FAILURE.json",
            {
                "schema": "independent_phase_e_a5_failure_v1",
                "error": f"{type(error).__name__}: {error}",
                "scientific_negative": False,
                "output_preserved": True,
            },
        )
        raise
    finally:
        runtime_teacher_role.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


__all__ = [
    "IndependentDurableState",
    "build_run_contract",
    "epoch_order",
    "run_e_a5",
    "verify_run_contract",
]
