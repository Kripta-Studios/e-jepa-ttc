"""Complete-state fixed50-epoch native Garl producers with bounded lossless cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import random
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
        dataset.parents[1] / "annotations/train.parquet",
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
    """One256MiB in-memory LRU;2GB lossless numeric cache, never raw copies."""

    def __init__(self, c: Campaign, rows: dict) -> None:
        from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

        self.c, self.rows = c, rows
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
            while self.disk + len(payload) > 2_000_000_000 and self.files:
                name, size = self.files.popitem(last=False)
                candidate = (self.root / name).resolve()
                if candidate.parent != self.root.resolve():
                    raise ValueError("cache eviction escaped owned root")
                candidate.unlink()
                self.disk -= size
            atomic_bytes(path, payload)
            self.files[key] = len(payload)
            self.disk += len(payload)
        size = value[0].numel() * 4 + value[1].numel() * 4
        while self.bytes + size > 256 * 1024**2 and self.values:
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
        code / "garl_ttc/models/ttc_network.py",
        code / "garl_ttc/models/resnet.py",
    ]
    protocol = {
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "admission_sha256": digest(c.out / "garl/ADMISSION.json"),
        "microbatch_profile_sha256": digest(c.out / "garl/MICROBATCH_PROFILE.json"),
        "fits": admission["fits"],
        "updates": admission["updates_exact_50_epochs"],
        "epochs": 50,
        "seed": 7,
        "effective_batch": 128,
        "microbatch": profile["selected_microbatch"],
        "batchnorm_adaptation": profile["hardware_adaptation"],
        "native_config": native_config(c),
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
            for fit in protocol["fits"]:
                endpoint = run_fit(c, fit, protocol, pin, ledger, cache, ctor)
                if endpoint is None:
                    return
                endpoints.append(endpoint)
                gc.collect()
                torch.cuda.empty_cache()
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
