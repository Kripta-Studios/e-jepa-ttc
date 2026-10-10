"""Reference contracts for reusing CausalScaleTTC with real RGB observations.

No model is trained here. Use the repository's CausalScaleTTC implementation;
this module only demonstrates input, time, schema and split boundaries.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

RGB_PHASE17_NAMES = (
    "rgb_luminance_mean", "rgb_spatial_gradient_log1p",
    "rgb_a5_flow", "rgb_a5_margin", "rgb_a5_log_variance",
    "rgb_c2f_flow", "rgb_c2f_margin", "rgb_c2f_log_variance",
    "rgb_a5_phase", "rgb_c2f_phase", "rgb_pair_phase",
    "rgb_pair_minus_a5", "rgb_pair_minus_c2f", "rgb_c2f_minus_a5",
    "rgb_abs_pair_minus_a5", "rgb_abs_pair_minus_c2f", "rgb_abs_c2f_minus_a5",
)
RGB_SCHEMA_HASH = hashlib.sha256(json.dumps(RGB_PHASE17_NAMES).encode()).hexdigest()


@dataclass(frozen=True)
class RGBFrame:
    frame_id: str
    timestamp_us: int
    available_us: int

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame identity required")
        if not isinstance(self.timestamp_us, int) or not isinstance(self.available_us, int):
            raise TypeError("timestamps must be Python integer microseconds")
        if self.available_us < self.timestamp_us:
            raise ValueError("availability cannot precede this timestamp convention")


def select_frames(
    frames: Sequence[RGBFrame], *, cutoff_us: int, lookback_us: int = 650_000
) -> list[RGBFrame]:
    """Select a label-independent list; no future interpolation or fake frames."""
    if lookback_us <= 0:
        raise ValueError("lookback must be positive")
    identifiers = [f.frame_id for f in frames]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate frame IDs in timeline")
    selected = [f for f in frames if cutoff_us-lookback_us <= f.timestamp_us <= cutoff_us
                and f.available_us <= cutoff_us]
    selected.sort(key=lambda f: (f.timestamp_us, f.frame_id))
    if any(a.timestamp_us >= b.timestamp_us for a,b in zip(selected, selected[1:])):
        raise ValueError("simultaneous frames require an explicit camera/tie policy")
    return selected


def triplet_context(
    frames: Sequence[RGBFrame], *, cutoff_us: int, lookback_us: int = 650_000,
    max_observations: int = 8,
) -> list[tuple[RGBFrame, RGBFrame, RGBFrame]]:
    if not 1 <= max_observations <= 16:
        raise ValueError("invalid history capacity")
    ordered = select_frames(frames, cutoff_us=cutoff_us, lookback_us=lookback_us)
    observations = [tuple(ordered[i-2:i+1]) for i in range(2,len(ordered))]
    return observations[-max_observations:]


def deltas_seconds(timestamps_us: Sequence[int]) -> np.ndarray:
    """Subtract integer times BEFORE converting intervals to float32 seconds."""
    if len(timestamps_us) < 2 or not all(isinstance(t, (int,np.integer)) for t in timestamps_us):
        raise TypeError("at least two integer timestamps required")
    diffs = [int(b)-int(a) for a,b in zip(timestamps_us, timestamps_us[1:])]
    if any(d <= 0 for d in diffs):
        raise ValueError("distinct chronological endpoints required")
    return np.asarray(diffs, dtype=np.float64).astype(np.float32)/np.float32(1_000_000)


def rgb_unit_interval(values: np.ndarray) -> np.ndarray:
    """Prepare [..,3,H,W] RGB for existing modality='rgb' support semantics."""
    x = np.asarray(values)
    if x.ndim < 3 or x.shape[-3] != 3 or min(x.shape[-2:]) < 2:
        raise ValueError("RGB must end in [3,H>=2,W>=2]")
    if x.dtype == np.uint8:
        x = x.astype(np.float32)/np.float32(255.0)
    elif x.dtype in (np.dtype('float32'),np.dtype('float64')):
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("float RGB must be raw unit-interval, not ImageNet-normalized")
        x = x.astype(np.float32,copy=False)
    else:
        raise TypeError("declare uint8 or raw float32/64 RGB; no implicit uint16 scaling")
    return np.ascontiguousarray(x)


def rgb_statistics(values: np.ndarray) -> np.ndarray:
    """Two proposed, label-free RGB coordinates; NOT synthetic event counts."""
    x = rgb_unit_interval(values)
    luma = (np.float32(.2126)*x[...,0,:,:] + np.float32(.7152)*x[...,1,:,:]
            + np.float32(.0722)*x[...,2,:,:])
    mean = luma.mean(axis=(-2,-1),dtype=np.float32)
    dx=np.abs(np.diff(luma,axis=-1)).mean(axis=(-2,-1),dtype=np.float32)
    dy=np.abs(np.diff(luma,axis=-2)).mean(axis=(-2,-1),dtype=np.float32)
    return np.stack([mean,np.log1p((dx+dy)*np.float32(.5))],axis=-1).astype(np.float32)


def legacy_rgb_support(values: np.ndarray) -> np.ndarray:
    """Mirror current RGB heuristic for a contract test, not calibrated confidence.

    It includes channel variance. In particular, a flat coloured patch need not
    have zero support. Do not interpret this heuristic as a proof of texture.
    """
    x=rgb_unit_interval(values)
    std=x.std(axis=(-3,-2,-1));mean=x.mean(axis=(-3,-2,-1))
    return np.clip(std/.1,0,1)*np.sqrt(4*mean*(1-mean))


def rgb_config(event_config: Mapping[str, object]) -> dict[str, object]:
    """Reuse geometry/transport config; this DOES NOT adapt learned weights."""
    c=copy.deepcopy(dict(event_config))
    if c.get('modality','event') != 'event':
        raise ValueError("expected an explicitly identified event configuration")
    if c.get('temporal_channel_gate_enabled',False):
        raise ValueError("event-bin gate needs a separately registered RGB decision")
    c['modality']='rgb';c['in_channels']=3
    return c


def phase17(stats: np.ndarray, a5_diag: np.ndarray, c2f_diag: np.ndarray,
            expert_phases: np.ndarray) -> np.ndarray:
    arrays=[np.asarray(a,dtype=np.float32) for a in (stats,a5_diag,c2f_diag,expert_phases)]
    if any(a.ndim < 1 for a in arrays):
        raise ValueError("feature blocks require a trailing channel dimension")
    prefix=arrays[0].shape[:-1]
    if [a.shape[-1] for a in arrays] != [2,3,3,3] or any(a.shape[:-1]!=prefix for a in arrays):
        raise ValueError("RGB PHASE17 block shape mismatch")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("unavailable observations use masks, never nonfinite features")
    p=arrays[3]
    d=np.stack([p[...,2]-p[...,0],p[...,2]-p[...,1],p[...,1]-p[...,0]],axis=-1)
    return np.concatenate([*arrays,d,np.abs(d)],axis=-1)


def validate_group_roles(producer: set[str], head: set[str], development: set[str]) -> None:
    if not all((producer,head,development)):
        raise ValueError("all three partitions must be nonempty")
    if producer&head or producer&development or head&development:
        raise ValueError("producer/head/development groups overlap")


def assert_checkpoint_excludes(trained_groups: set[str], eval_groups: set[str]) -> None:
    if trained_groups & eval_groups:
        raise ValueError("checkpoint/teacher/normalizer has seen evaluation groups")


def current_ttc_from_heights(h_previous: float, h_current: float, dt: float) -> float:
    if h_previous <= 0 or h_current <= 0 or dt <= 0:
        raise ValueError("positive heights and elapsed time required")
    return dt/np.expm1(np.log(h_current/h_previous))


def previous_ttc_from_heights(h_previous: float, h_current: float, dt: float) -> float:
    if h_previous <= 0 or h_current <= 0 or dt <= 0:
        raise ValueError("positive heights and elapsed time required")
    return dt/(1-h_previous/h_current)


def phase_from_ttc(ttc: np.ndarray) -> np.ndarray:
    """Canonical target transform; no attempt to silently repair illegal targets."""
    t=np.asarray(ttc,dtype=np.float64)
    if not np.isfinite(t).all() or not ((t<0)|(t>.1)).all():
        raise ValueError("invalid signed TTC domain")
    return -np.log1p(-.1/t)
