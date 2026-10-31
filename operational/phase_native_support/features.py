"""Exact native clock/count coordinates, with explicit unexecuted-context masks."""

import numpy as np

INDICES = tuple(range(165, 170)) + tuple(range(335, 340))
HEAD_INPUT = "PHASE_ONLY_EVENT_S7_TRAIN40.input_0"


def validate_support(quality: np.ndarray, known: np.ndarray) -> None:
    """Validate clocks in seconds and count/8; unknown zeros are placeholders only."""
    if quality.ndim != 2 or quality.shape[1] != 10 or quality.dtype != np.float32:
        raise ValueError("Expected FP32 native support [N,10]")
    if known.shape != (len(quality), 2) or known.dtype != bool:
        raise ValueError("Expected boolean executed-context mask [N,2]")
    if not np.isfinite(quality).all():
        raise ValueError("Native support must be finite, with explicit unknown placeholders")
    for modality in range(2):
        block = quality[:, modality * 5 : modality * 5 + 5]
        if np.any(block[~known[:, modality]] != 0):
            raise ValueError("Unexecuted contexts require zero placeholders")
        observed = block[known[:, modality]]
        clocks, counts = observed[:, :4], observed[:, 4] * 8
        if (
            np.any(counts != np.round(counts))
            or np.any((counts < 1) | (counts > 8))
            or np.any(clocks < 0)
            or np.any(clocks > 0.650001)
            or not np.allclose(clocks[:, 0], clocks[:, 1] + clocks[:, 3], rtol=0, atol=1e-7)
            or np.any(clocks[counts == 1, 2] != 0)
            or np.any(clocks[counts > 1, 2] <= 0)
        ):
            raise ValueError("Native clock identities, interval or valid count differ")


def from_phase340(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy the ten original coordinates from real fused states, preserving their order."""
    if features.ndim != 2 or features.shape[1] != 340 or features.dtype != np.float32:
        raise ValueError("Expected original FP32 [N,340] fused states")
    quality = features[:, INDICES].copy()
    known = np.ones((len(features), 2), bool)
    validate_support(quality, known)
    return quality, known


def from_context_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Recover executed contexts, including explicit event-only/unavailable routes."""
    quality = np.zeros((1, 10), np.float32)
    known = np.zeros((1, 2), bool)
    for index, name in enumerate(("R_CTX", "E_CTX_MATCHED")):
        timing_key, valid_key = f"{name}.input_1", f"{name}.input_2"
        if timing_key not in arrays and valid_key not in arrays:
            continue
        timing, valid = arrays[timing_key], arrays[valid_key]
        if (
            timing.shape != (1, 8, 4)
            or timing.dtype != np.float32
            or valid.shape != (1, 8)
            or valid.dtype != bool
            or not valid[0, -1]
            or np.any(valid[:, :-1] & ~valid[:, 1:])
        ):
            raise ValueError("Executed context timing or validity suffix differs")
        known[0, index] = True
        quality[0, index * 5 : index * 5 + 4] = timing[0, -1]
        quality[0, index * 5 + 4] = valid.sum() / 8
    if HEAD_INPUT in arrays:
        fused_quality, fused_known = from_phase340(arrays[HEAD_INPUT])
        if not np.array_equal(quality, fused_quality) or not np.array_equal(known, fused_known):
            raise ValueError("Head coordinates differ from executed context inputs")
    validate_support(quality, known)
    return quality, known
