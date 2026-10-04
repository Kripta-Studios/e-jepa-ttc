"""Complete-state fixed50-epoch native Garl producers with bounded lossless cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import random
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import cast

from .common import ROOT, Campaign, Lease, atomic_bytes, atomic_json, digest, read
from .native_garl import model_class, native_config


def records(c: Campaign) -> dict:
    """Join only exact authorized D1 tokens; never materialize excluded payloads."""
    import numpy as np
    import pyarrow.parquet as pq

    admission = read(c.out / "garl/ADMISSION.json")
    tokens = set()
    for fit in admission["fits"]:
        with np.load(
            c.out / f"garl/admission/{fit['key']}_native_tokens.npz", allow_pickle=False
        ) as z:
            tokens.update(map(str, z["tokens"]))
    filters = [("sample_token", "in", sorted(tokens))]
    dataset = Path(c.local["garl_annotations_candidate"])
    original = read(Path(c.launch["source_configuration"]))["expansion"]["0"]
    label_path = dataset.parents[1] / "annotations/train.parquet"
    if (
        digest(dataset) != original["metadata_sha256"]
        or digest(label_path) != original["labels_sha256"]
    ):
        raise ValueError("native TRAIN metadata or labels differ from original frozen SHA256")
    inputs = pq.read_table(
        dataset,
        columns=[
            "sample_token",
            "sequence_id",
            "event_windows_us",
            "boxes_xyxy",
            "frame_timestamps_us",
        ],
        filters=filters,
        use_threads=False,
    ).to_pylist()
    labels = pq.read_table(
        label_path,
        columns=["sample_token", "frame_ttc", "box3d_h", "box3d_Fcam"],
        filters=filters,
        use_threads=False,
    ).to_pylist()
    output = {row["sample_token"]: row for row in inputs}
    for label in labels:
        output[label["sample_token"]].update(label)
    if set(output) != tokens:
        raise ValueError("native exact-token input/label coverage incomplete")
    return output


class InputCache:
    """Bounded FP32 memory LRU and lossless disk cache under the owned10GB quota."""

    def __init__(self, c: Campaign, rows: dict) -> None:
        from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

        self.c, self.rows = c, rows
        self.memory_limit = 6 * 1024**3 if c.policy["max_tree_rss_gib"] > 4 else 256 * 1024**2
        self.pool = ReaderPool()
        self.values: OrderedDict = OrderedDict()
        self.root = c.out / "garl/native_cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = digest(c.out / "garl/PROTOCOL.json")
        self.bytes = 0
        self.reads = 0
        self.hits = 0
        self.files = OrderedDict(
            (p.name, p.stat().st_size) for p in sorted(self.root.glob("*.npz"))
        )
        self.disk = sum(self.files.values())
        self.limit = 0
        self.refresh_capacity()

    def refresh_capacity(self) -> None:
        """Reserve1GB for atomic checkpoints; existing scientific outputs are never evicted."""
        owned_noncache = sum(
            p.stat().st_size
            for p in self.c.out.rglob("*")
            if p.is_file() and self.root not in p.parents
        )
        self.limit = max(0, min(8_000_000_000, 9_000_000_000 - owned_noncache))
        self.evict(0)
        atomic_json(
            self.c.out / "garl/CACHE_BUDGET.json",
            {
                "owned_noncache_bytes": owned_noncache,
                "cache_limit_bytes": self.limit,
                "cached_bytes": self.disk,
                "atomic_checkpoint_reservation_bytes": 1_000_000_000,
                "owned_quota_bytes": 10_000_000_000,
                "lossless_FP32": True,
            },
        )

    def evict(self, incoming_bytes: int) -> None:
        """Evict only verified paths inside this process's disposable numeric cache."""
        while self.disk + incoming_bytes > self.limit and self.files:
            name, size = self.files.popitem(last=False)
            candidate = (self.root / name).resolve()
            if candidate.parent != self.root.resolve():
                raise ValueError("cache eviction escaped owned root")
            candidate.unlink()
            self.disk -= size

    def get(self, token: str) -> tuple:
        """Cache by bound exact token; FP32 tensors, no uint16/FP16 approximation."""
        import io

        import numpy as np
        import torch

        from e_jepa_ttc.efficient_context.garl_input import encode_record

        if token in self.values:
            self.values.move_to_end(token)
            self.hits += 1
            return self.values[token]
        row = self.rows[token]
        raw_stat = (self.c.raw / row["sequence_id"] / "events.h5").stat()
        key = (
            hashlib.sha256(
                f"{self.binding}:{token}:{raw_stat.st_size}:{raw_stat.st_mtime_ns}".encode()
            ).hexdigest()
            + ".npz"
        )
        path = self.root / key
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                value = (
                    torch.from_numpy(z["events"].copy()),
                    torch.from_numpy(z["visible"].copy()),
                    float(z["target"]),
                )
            self.hits += 1
            self.files.move_to_end(key)
        else:
            self.c.require_resources()
            value = encode_record(self.rows[token], self.pool, self.c.raw)
            self.reads += 1
            stream = io.BytesIO()
            np.savez_compressed(
                stream, events=value[0].numpy(), visible=value[1].numpy(), target=value[2]
            )
            payload = stream.getvalue()
            self.evict(len(payload))
            if len(payload) <= self.limit:
                atomic_bytes(path, payload)
                self.files[key] = len(payload)
                self.disk += len(payload)
        size = value[0].numel() * 4 + value[1].numel() * 4
        while self.bytes + size > self.memory_limit and self.values:
            _, previous = self.values.popitem(last=False)
            self.bytes -= previous[0].numel() * 4 + previous[1].numel() * 4
        self.values[token] = value
        self.bytes += size
        return value

    def close(self) -> None:
        """Release resident inputs and HDF5 handles owned by this producer process."""
        self.pool.close()
        self.values.clear()


def execute(c: Campaign) -> None:
    """Freeze native endpoints and run only missing registered producer fits."""
    import torch

    c.require_resources()
    c.freeze()
    admission = read(c.out / "garl/ADMISSION.json")
    profile = read(c.out / "garl/MICROBATCH_PROFILE.json")
    if admission["reasons"] or profile["status"] != "ADMITTED":
        raise ValueError("native producer dependencies not admitted")
    if read(c.out / "garl/INPUT_QA.json")["status"] != "PASSED":
        raise ValueError("native TRAIN sensor/input QA must pass before the first update")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    code = Path(c.local["garl_code_candidate"])
    inventory = [
        Path(__file__),
        Path(__file__).with_name("native_garl.py"),
        ROOT / "src/e_jepa_ttc/efficient_context/garl_input.py",
        Path(__file__).with_name("garl_qa.py"),
        Path(__file__).with_name("budget.py"),
        code / "garl_ttc/models/ttc_network.py",
        code / "garl_ttc/models/resnet.py",
    ]
    protocol = {
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "admission_sha256": digest(c.out / "garl/ADMISSION.json"),
        "microbatch_profile_sha256": digest(c.out / "garl/MICROBATCH_PROFILE.json"),
        "input_QA_sha256": digest(c.out / "garl/INPUT_QA.json"),
        "raw_restoration_plan_sha256": digest(c.out / "data_recovery/DOWNLOAD_PLAN.json"),
        "resource_authorization_sha256": digest(c.out / "RESOURCE_AUTHORIZATION.json")
        if (c.out / "RESOURCE_AUTHORIZATION.json").exists()
        else None,
        "active_tree_RSS_limit_bytes": int(c.policy["max_tree_rss_gib"] * 1024**3),
        "fits": admission["fits"],
        "updates": admission["updates_exact_50_epochs"],
        "epochs": 50,
        "seed": 7,
        "effective_batch": 128,
        "microbatch": profile["selected_microbatch"],
        "batchnorm_adaptation": profile["hardware_adaptation"],
        "native_config": native_config(c),
        "execution_order": [
            fit["key"]
            for fit in sorted(
                admission["fits"], key=lambda fit: (-fit["updates_50_epochs"], fit["key"])
            )
        ],
        "cache": {
            "FP32_lossless": True,
            "memory_bytes": 6 * 1024**3 if c.policy["max_tree_rss_gib"] > 4 else 256 * 1024**2,
            "disk_max_bytes": 8_000_000_000,
            "atomic_checkpoint_reservation_bytes": 1_000_000_000,
            "within_owned_total_bytes": 10_000_000_000,
        },
        "science_files": [{"path": str(p), "sha256": digest(p)} for p in inventory],
    }
    freeze = c.out / "garl/PROTOCOL.json"
    if freeze.exists():
        if read(freeze) != protocol:
            raise ValueError("native producer protocol changed")
    else:
        atomic_json(freeze, protocol)
    pin = digest(freeze)
    ctor = model_class(c)
    rows = records(c)
    cache = InputCache(c, rows)
    endpoints = []
    ledger_path = c.out / "garl/PHYSICAL_WORK.json"
    ledger = (
        read(ledger_path)
        if ledger_path.exists()
        else {"fits": {}, "uncertain_lost_upper": 0, "saved_updates": 0}
    )
    with Lease(c.out):
        try:
            fits = {fit["key"]: fit for fit in protocol["fits"]}
            failures_path = c.out / "garl/FIT_FAILURES.json"
            failures = read(failures_path) if failures_path.exists() else {}
            for key in protocol["execution_order"]:
                fit = fits[key]
                cache.refresh_capacity()
                if key in failures:
                    # Preserve an integrity failure for inspection; no automatic scientific retry.
                    continue
                try:
                    endpoint = run_fit(c, fit, protocol, pin, ledger, cache, ctor)
                except ArithmeticError as error:
                    failures[key] = {
                        "status": "FAILED_INTEGRITY",
                        "error": repr(error),
                        "scientific_negative": False,
                        "automatic_retry": False,
                    }
                    atomic_json(failures_path, failures)
                    endpoint = None
                if endpoint is None:
                    if not c.check():
                        return
                else:
                    endpoints.append(endpoint)
                atomic_json(
                    c.out / "garl/INDEPENDENT_PROGRESS.json",
                    {
                        "completed_fits": endpoints,
                        "integrity_failures": failures,
                        "all_twelve_required_before_evaluation": True,
                        "scientific_negative_for_missing_fit": False,
                    },
                )
                gc.collect()
                torch.cuda.empty_cache()
            if len(endpoints) == 12:
                atomic_json(
                    c.out / "garl/ENDPOINTS.json",
                    {
                        "fits": endpoints,
                        "all_twelve_frozen_before_evaluation": True,
                        "protocol_sha256": pin,
                        "updates": ledger["saved_updates"],
                    },
                )
        finally:
            cache.close()


def run_fit(
    c: Campaign,
    fit: dict,
    protocol: dict,
    pin: str,
    ledger: dict,
    cache: InputCache,
    ctor: Callable,
) -> dict | None:
    """Run one fixed fit; checkpoint closure cannot escape to another fit."""
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.training import atomic_checkpoint, state_digest

    profile = {"selected_microbatch": protocol["microbatch"]}
    ledger_path = c.out / "garl/PHYSICAL_WORK.json"
    c.require_resources()
    key = fit["key"]
    folder = c.out / "garl/fits" / key
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint_last.pt"
    with np.load(c.out / f"garl/admission/{key}_native_tokens.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(str)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = ctor(native_config(c), is_train=True).float().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 30, 40], gamma=0.5
    )
    generator = torch.Generator().manual_seed(7)
    epoch, cursor, completed = 1, 0, 0
    order = torch.randperm(len(tokens), generator=generator)
    losses = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seal = state.pop("state_sha256")
        if state_digest(state) != seal or state["protocol_sha256"] != pin:
            raise ValueError("native complete-state checkpoint mismatch")
        if state["status"].startswith("FAILED_INTEGRITY"):
            raise ValueError(
                "native integrity failure requires inspection; automatic retry prohibited"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        nr = state["numpy_rng"]
        np.random.set_state(
            (
                nr["algorithm"],
                nr["keys"].numpy(),
                nr["position"],
                nr["has_gauss"],
                nr["cached_gaussian"],
            )
        )
        epoch, cursor, completed = (
            state["epoch"],
            state["cursor"],
            state["completed_updates"],
        )
        order = state["order"]
        losses = state["losses"]
        del state
    work = ledger["fits"].setdefault(key, {"saved": 0, "pending": None})
    if work["pending"]:
        ledger["uncertain_lost_upper"] += work["pending"][1] - completed
        work.update(saved=completed, pending=None)
    elif work["saved"] != completed:
        raise ValueError("native ledger and checkpoint progress differ")
    session_start = time.perf_counter()
    session_start_updates = completed

    def performance() -> dict:
        elapsed = time.perf_counter() - session_start
        updates = completed - session_start_updates
        return {
            "session_elapsed_seconds": elapsed,
            "session_completed_updates": updates,
            "session_updates_per_second": updates / elapsed if updates else None,
            "includes_raw_cache_warmup_and_checkpoint_work": True,
            "source_cache_reads": cache.reads,
            "source_cache_hits": cache.hits,
            "memory_cache_bytes": cache.bytes,
            "memory_cache_limit_bytes": cache.memory_limit,
            "disk_cache_bytes": cache.disk,
            "disk_cache_limit_bytes": cache.limit,
        }

    def save(status: str) -> None:
        nr = cast(tuple, np.random.get_state(legacy=True))
        state = {
            "protocol_sha256": pin,
            "key": key,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "sampler_rng": generator.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": {
                "algorithm": nr[0],
                "keys": torch.from_numpy(nr[1].copy()),
                "position": nr[2],
                "has_gauss": nr[3],
                "cached_gaussian": nr[4],
            },
            "order": order,
            "cursor": cursor,
            "epoch": epoch,
            "completed_updates": completed,
            "losses": losses,
            "status": status,
        }
        cache.refresh_capacity()
        atomic_checkpoint(checkpoint, state)
        work.update(saved=completed, pending=None, checkpoint_sha256=digest(checkpoint))
        ledger["saved_updates"] = sum(v["saved"] for v in ledger["fits"].values())
        atomic_json(ledger_path, ledger)
        atomic_json(
            c.out / "garl/PROGRESS.json",
            {
                "fit": key,
                "epoch": epoch,
                "cursor": cursor,
                "updates": completed,
                "status": status,
                "cache_reads": cache.reads,
                "cache_hits": cache.hits,
                "performance": performance(),
            },
        )

    if not checkpoint.exists():
        save("RUNNING")
    while epoch <= 50:
        if not c.check():
            save("PAUSED_RESOURCE")
            return
        from .budget import require

        require(c)
        if work["pending"] is None:
            work["pending"] = [
                completed,
                min((completed // 100 + 1) * 100, fit["updates_50_epochs"]),
            ]
            atomic_json(ledger_path, ledger)
        take = order[cursor : cursor + 128].numpy()
        # Prepare the entire effective batch before mutating BatchNorm or RNG.
        # A transient raw error can then save the last complete optimizer boundary.
        try:
            prepared = [cache.get(str(tokens[int(i)])) for i in take]
        except (InterruptedError, OSError) as error:
            save("PAUSED_RESOURCE" if isinstance(error, InterruptedError) else "BLOCKED_DEPENDENCY")
            atomic_json(c.out / "garl/DEPENDENCY.json", {"error": repr(error), "fit": key})
            return None
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        micro = profile["selected_microbatch"]
        for start in range(0, len(take), micro):
            ids = take[start : start + micro]
            batch = prepared[start : start + micro]
            x = torch.stack([v[0] for v in batch]).cuda()
            visible = torch.stack([v[1] for v in batch]).cuda()
            target = torch.tensor([v[2] for v in batch], device="cuda")
            _, _, parts, _ = model.forward_train(
                x, target, visible_height_target=visible, epoch_idx=epoch
            )
            loss = torch.stack(list(parts.values())).sum() * len(ids) / len(take)
            if not torch.isfinite(loss):
                save("FAILED_INTEGRITY_NONFINITE_NATIVE_LOSS")
                raise ArithmeticError(
                    f"native source-style loss nonfinite at {key}:{epoch}:{cursor}"
                )
            loss.backward()
            total_loss += float(loss.detach())
            del x, visible, target, batch, parts, loss
        optimizer.step()
        del prepared
        cursor += len(take)
        completed += 1
        losses.append(total_loss)
        if cursor == len(tokens):
            scheduler.step()
            epoch += 1
            cursor = 0
            order = torch.randperm(len(tokens), generator=generator) if epoch <= 50 else order
        if completed % 100 == 0:
            save("RUNNING")
            print(f"GARL_{key}_epoch{epoch}_update{completed}", flush=True)
        if completed % 10 == 0:
            atomic_json(
                c.out / "garl/UPDATE_PROGRESS.json",
                {
                    "fit": key,
                    "epoch": epoch,
                    "confirmed_updates": completed,
                    "durable_updates": work["saved"],
                    "performance": performance(),
                    "scientific_endpoint_complete": completed == fit["updates_50_epochs"],
                },
            )
    if completed != fit["updates_50_epochs"]:
        raise ValueError("native50-epoch endpoint count differs from preregistered budget")
    save("COMPLETE")
    endpoint = {
        "key": key,
        "checkpoint": str(checkpoint),
        "sha256": digest(checkpoint),
        "updates": completed,
        "epochs": 50,
        "train_sequences": fit["train_sequences"],
        "excluded_outer": fit["excluded_outer"],
        "excluded_inner": fit["excluded_inner"],
    }
    atomic_bytes(
        folder / "TRAINING_CURVE.csv",
        ("update,loss\n" + "".join(f"{i + 1},{v:.17g}\n" for i, v in enumerate(losses))).encode(),
    )
    return endpoint


def main() -> int:
    """Execute native scientific producers only after data/hardware admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    campaign = Campaign(args.protocol)
    execute(campaign)
    return 0 if (campaign.out / "garl/ENDPOINTS.json").exists() else 3


if __name__ == "__main__":
    raise SystemExit(main())
