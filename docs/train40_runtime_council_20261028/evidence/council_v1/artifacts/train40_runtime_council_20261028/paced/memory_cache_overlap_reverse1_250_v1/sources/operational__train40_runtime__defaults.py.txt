"""Conservative default for the admitted one/two-object TRAIN40 runtime.

The profile preserves frozen weights and FP32 arithmetic. It is not an admission
of 50 Hz: compiled transport still fails TTC parity, and uint8 preprocessing did
not show a consistent end-to-end latency advantage in the paired pilot.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeDefaults:
    """Explicit defaults shared by the persistent service and measurement CLIs."""

    name: str = "train40_single_dual_parity_v1"
    max_objects: int = 2
    replay_objects: int = 1
    backend: str = "prefix"
    event_query_graphs: bool = True
    event_normalization: str = "topk"
    reuse_event_encodings: bool = False
    rgb_object_chunk_size: int = 1
    parallel_branches: bool = True
    rgb_workers: int = 8
    shared_rgb_images: bool = False
    rgb_uint8: bool = False


DEFAULT_PROFILE = RuntimeDefaults()
