"""Read-bound frozen producer runtime; every receipt belongs to the new campaign."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

from .common import Campaign, atomic_json, digest, read


class FrozenRuntime:
    """One pinned family and three historical heads, plus the sealed WIDE head."""

    def __init__(self, c: Campaign, protocol: dict) -> None:
        import numpy as np
        import torch

        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner

        self.c, self.protocol = c, protocol
        self.models, self.heads = {}, {}
        self.family = None
        root = c.historical / "artifacts/simplex_t/shared_gpu_route_20261004"
        normalizer = root / "NORMALIZER.npz"
        if digest(normalizer) != protocol["normalizer_sha256"]:
            raise ValueError("frozen profile normalizer changed")
        with np.load(normalizer, allow_pickle=False) as z:
            self.mean, self.scale = z["mean"], z["scale"]
        night = c.historical / "artifacts/simplex_t/nocturnal_20261003"
        manifest = read(night / "PROFILE_INPUT_EXPORT.json")
        if digest(night / "PROFILE_INPUT_EXPORT.json") != protocol["head_index_sha256"]:
            raise ValueError("frozen profile head inventory changed")
        self.cached = {}
        for label in ("H1_SEED7", "H8_SEED7", "H16_SEED7"):
            row = manifest["models"][label]
            head = TemporalRefiner(TemporalConfig(**row["constructor"]["config"]))
            for kind in ("weights", "inputs"):
                if digest(night / row[kind + "_path"]) != row[kind + "_sha256"]:
                    raise ValueError("frozen head weights or inputs changed")
            with np.load(night / row["weights_path"], allow_pickle=False) as z:
                head.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
            self.heads[label[: label.index("_")]] = head.float().eval()
            with np.load(night / row["inputs_path"], allow_pickle=False) as z:
                self.cached[label[: label.index("_")]] = {k: z[k] for k in z.files}
        if not (c.out / "ENDPOINTS_seed7.json").exists():
            return
        seal = read(c.out / "ENDPOINTS_seed7.json")
        if len(seal["fits"]) != 3:
            raise ValueError("WIDE three-fold seal required for profiling")
        row = next(v for v in seal["fits"] if v["fold"] == 0)
        if digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("WIDE endpoint changed")
        from e_jepa_ttc.simplex_t.endpoint import load_endpoint

        self.heads["WIDE"] = load_endpoint(
            Path(row["checkpoint"]),
            TemporalConfig(**row["model"]),
            seed=7,
            freeze_sha256=digest(c.out / "PROTOCOL.json"),
            train_source_sha256=c.freeze()["sources"]["0"]["wide_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )

    def load(self, query: dict) -> None:
        """Bind the INNER-OOF producer family and account initialization separately."""
        import torch

        from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
        from e_jepa_ttc.training.stage61_pair_head import load_pair_head

        if self.family == query["family_id"]:
            return
        self.close()
        self.c.require_resources()
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise InterruptedError("frozen family requires2GiB free GPU before loading")
        paths = {r["registered_sha256"]: Path(r["path"]) for r in self.protocol["checkpoints"]}
        begin = time.perf_counter()
        for name in ("A5", "C2F", "PAIR"):
            pin = query["experts"][name]
            path = paths[pin]
            if digest(path) != pin:
                raise ValueError("frozen INNER-OOF producer changed")
            loader = load_pair_head if name == "PAIR" else load_causal_scale_replay_checkpoint
            self.models[name] = loader(path, device=torch.device("cuda"))
        torch.cuda.synchronize()
        self.family = query["family_id"]
        atomic_json(
            self.c.out / f"profile/model_load/family{self.family}.json",
            {"milliseconds": (time.perf_counter() - begin) * 1000, "optimizer_updates": 0},
        )

    def prepared(
        self,
        query: dict,
        index: dict,
        label: str,
        tensor: Tensor,
        valid: np.ndarray,
        compact: bool,
    ) -> dict:
        """Prepared FP32 inputs through producers, normalization and the fixed head."""
        import numpy as np
        import torch

        from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS
        from e_jepa_ttc.simplex_t.phase import phase_to_ttc
        from operational.simplex_t_shared_route.adapter import extract, inputs

        i = query["index_row"]
        adapter = "H16_SEED7" if label == "WIDE" else label + "_SEED7"
        begin = time.perf_counter()
        slots = np.flatnonzero(valid) if compact else np.arange(16)
        delta = torch.tensor(
            np.diff(index["base_windows_us"][i][:, 1]) / 1e6, dtype=torch.float32
        ).repeat(len(slots), 1)
        events_gpu = tensor[slots].to("cuda") if compact else tensor.to("cuda")
        delta_gpu = delta.to("cuda")
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        raw = np.zeros((16, 17), np.float32)
        with torch.inference_mode():
            raw[slots] = extract(adapter, self.models, events_gpu, delta_gpu)
        del events_gpu, delta_gpu
        torch.cuda.synchronize()
        produced = time.perf_counter()
        # Build the H16 parent before WIDE selection; retain parent padding invariants.
        parent_valid = index["valid"][i] if label == "WIDE" else valid
        xs = inputs(
            adapter,
            raw,
            parent_valid,
            index["lag_us"],
            int(index["anchor_us"][i]),
            int(index["roi_available_us"][i]),
            self.mean,
            self.scale,
        )
        if label == "WIDE":
            x, timing, mask, experts = xs
            x, timing, mask = x[:, WIDE_SLOTS], timing[:, WIDE_SLOTS].clone(), mask[:, WIDE_SLOTS]
            timing[:, :, 2] = 0
            gap = np.diff(-index["lag_us"][list(WIDE_SLOTS)]) / 1e6
            timing[:, 1:, 2] = torch.from_numpy(gap.astype(np.float32)) * mask[:, :-1]
            timing[~mask] = 0
            xs = x, timing, mask, experts
        normalized = time.perf_counter()
        with torch.inference_mode():
            output = self.heads[label](*xs)
            ttc = phase_to_ttc(output["point_phase"].to(torch.float64))
        end = time.perf_counter()
        if not bool(torch.isfinite(ttc).all()):
            raise ValueError("nonfinite frozen-route TTC")
        return {
            "xs": tuple(v.numpy() for v in xs),
            "phase": output["point_phase"].numpy(),
            "ttc": ttc.numpy(),
            "prepared_ms": (end - begin) * 1000,
            "transfer_ms": (transferred - begin) * 1000,
            "producer_ms": (produced - transferred) * 1000,
            "normalization_ms": (normalized - produced) * 1000,
            "head_ms": (end - normalized) * 1000,
            "producer_observations": len(slots),
        }

    def close(self) -> None:
        """Release this process's family, preserving all historical checkpoints."""
        import torch

        self.models.clear()
        self.family = None
        gc.collect()
        torch.cuda.empty_cache()
