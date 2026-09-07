"""Historical expert/ROI adapter for the separately leased expanded replay queue."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.training.stage61_pair_head import load_pair_head

from .cached_event_reader import ReaderPool
from .context_raw_union import encode_context_union
from .expert_features import extract_family


@contextmanager
def expanded_inference_family(
    family_id: int,
    *,
    families: list[dict],
    checkpoint_paths: dict[str, Path],
    index: dict[str, np.ndarray],
    history: np.ndarray,
    raw_train_root: Path,
    allowed_sequences: set[str],
    preprocessing: dict,
    validate_prerequisites: Callable[[], None],
    producer_scope: Literal["expanded", "original_qa"] = "expanded",
) -> Iterator[Callable[[int], dict[str, np.ndarray]]]:
    """Load a family once and emit exactly the original H16 FP32 query layout.

    Caller holds CURRENT_REPLAY.lock and admits resources before entering. Its
    mandatory validator checks the acknowledged time scope, producer ancestry,
    index/preprocessing pins and numerical identity. The explicit original_qa
    scope also permits outer-dev families, only for the pinned OLD8192 QA path.
    Expanded replay retains the default inner-only restriction. This adapter
    never grants authority and never refits a producer or changes its teacher.
    """
    validate_prerequisites()
    if producer_scope not in {"expanded", "original_qa"}:
        raise ValueError("unknown historical inference scope")
    if producer_scope == "original_qa" and len(index["tokens"]) != 8192:
        raise ValueError("original QA requires the acknowledged OLD8192 index")
    permitted = set(range(12)) if producer_scope == "original_qa" else {0, 1, 2, 4, 5, 6, 8, 9, 10}
    if family_id not in permitted:
        raise ValueError("expanded inference requires an inner producer family")
    family = families[family_id]
    outer = family_id // 4
    role = "outer_dev" if family_id % 4 == 3 else f"inner{family_id % 4}"
    if family["outer_fold"] != outer or family["role"] != role:
        raise ValueError("expanded producer descriptor differs")
    if (
        torch.get_num_threads() != 4
        or torch.get_num_interop_threads() != 2
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
    ):
        raise ValueError("frozen FP32 runtime with 4/2 threads and TF32 off required")
    raw_root = raw_train_root.resolve(strict=True)
    paths = {}
    for expert in ("A5", "C2F", "PAIR"):
        digest = family["experts"][expert]
        path = checkpoint_paths[digest]
        if sha256(path) != digest:
            raise ValueError("expanded historical producer bytes changed")
        paths[expert] = path
    device = torch.device("cuda")
    models = []
    readers = ReaderPool()
    try:
        models.append(load_causal_scale_replay_checkpoint(paths["A5"], device=device))
        models.append(load_causal_scale_replay_checkpoint(paths["C2F"], device=device))
        models.append(load_pair_head(paths["PAIR"], device=device))

        def infer(qi: int) -> dict[str, np.ndarray]:
            validate_prerequisites()
            if (
                not 0 <= qi < len(index["tokens"])
                or index["producer_family"][outer, qi] != family_id
            ):
                raise ValueError("query inactive or bound to another producer")
            sequence = str(index["sequences"][qi])
            if sequence not in allowed_sequences:
                raise ValueError("query sequence not in independently authorized TRAIN roles")
            raw_path = (raw_root / sequence / "events.h5").resolve(strict=True)
            if not raw_path.is_relative_to(raw_root):
                raise ValueError("raw event path escapes permitted TRAIN root")
            mask, windows = index["valid"][qi], index["base_windows_us"][qi]
            if (
                mask.shape != (16,)
                or mask.dtype != bool
                or not mask[-1]
                or history.shape != index["valid"].shape
                or (history[qi, mask] < 0).any()
            ):
                raise ValueError("invalid active H16 history")
            tensor = encode_context_union(
                readers.get(raw_path),
                windows,
                index["lag_us"],
                mask,
                tuple(index["square_xyxy"][qi]),
                sequence_id=sequence,
                roi_size=preprocessing["roi_size"],
                event_pixel_diff=preprocessing["event_pixel_diff"],
            )
            delta = torch.tensor(np.diff(windows[:, 1]) / 1e6, dtype=torch.float32)
            arrays = extract_family(
                models[0], models[1], models[2], tensor.to(device), delta.repeat(16, 1).to(device)
            )
            return {
                **{key: value[mask] for key, value in arrays.items()},
                "observation_ids": history[qi, mask],
                "anchor_us": index["anchor_us"][qi] - index["lag_us"][mask],
                "available_us": np.full(int(mask.sum()), index["roi_available_us"][qi]),
            }

        yield infer
    finally:
        readers.close()
        models.clear()
        gc.collect()
