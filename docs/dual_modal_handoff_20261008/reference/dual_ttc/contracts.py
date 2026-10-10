"""Small independent checks. Presence of a hash does not establish training ancestry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import hashlib
import numpy as np

@dataclass(frozen=True)
class CheckpointOrigin:
    identity: str
    training_groups: frozenset[str]
    provenance: str

def require_heldout(origin: CheckpointOrigin, dev_groups: set[str]) -> None:
    if origin.provenance not in {'generic_pretraining', 'audited_task_training'}:
        raise ValueError('Unknown checkpoint genealogy cannot certify model-held-out evaluation')
    overlap = origin.training_groups & dev_groups
    if overlap:
        raise ValueError(f'Checkpoint trained on held-out groups: {sorted(overlap)}')

def split_groups(groups: Sequence[str], dev_fraction: float=.2, salt: str='dual-ttc-v13-dev-20261008') -> tuple[list[str], list[str]]:
    unique = sorted(set(groups))
    if len(unique) < 2 or not 0 < dev_fraction < 1:
        raise ValueError('Insufficient independent group identifiers or invalid fraction')
    order = sorted(unique, key=lambda g: hashlib.sha256((salt+'\0'+g).encode()).hexdigest())
    n = max(1, min(len(order)-1, round(len(order)*dev_fraction)))
    return sorted(order[n:]), sorted(order[:n])

def seconds_since(anchor_us: int, observation_us: int) -> float:
    # Int conversion before subtraction avoids subtraction after float32 timestamp rounding.
    return (int(anchor_us)-int(observation_us))*1e-6

def nearest_rgb(timestamps_us: np.ndarray, endpoint_us: int, tolerance_us: int, allowed_cutoff_us: int) -> int | None:
    t = np.asarray(timestamps_us)
    if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer) or np.any(t[1:] < t[:-1]):
        raise ValueError('RGB timestamps must be a sorted integer vector')
    if tolerance_us < 0:
        raise ValueError('Negative tolerance')
    if len(t)==0: return None
    i = int(np.searchsorted(t, np.int64(endpoint_us), side='left'))
    candidates = sorted({max(0,i-1), min(len(t)-1,i)})
    # Nearest frame is selected first, as in the frozen reference; no silent reselection
    # to an earlier less-near frame to hide an unavailable nearest acquisition.
    idx = min(candidates, key=lambda j: (abs(int(t[j])-int(endpoint_us)), j))
    if abs(int(t[idx])-int(endpoint_us)) > tolerance_us or int(t[idx]) > int(allowed_cutoff_us):
        return None
    return idx

def validate_modality_input(mode: str, rgb: object | None) -> None:
    if mode not in {'event', 'rgb_event'}: raise ValueError('Invalid mode')
    if mode == 'event' and rgb is not None:
        raise ValueError('Strict event inference must not accept RGB pixels')
