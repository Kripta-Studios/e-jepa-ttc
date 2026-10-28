"""Measured PIL/exact-transport profile for bounded TRAIN40 CPU-arrival replay.

The profile preserves frozen weights, normalizers and native producer batches.
The best verified 250-update one-object replay uses this configuration without
crop caching. It does not establish the sustained 50 Hz / all-updates <20 ms goal
or physical sensor latency. Experimental crop/V2/5 ms paths remain disabled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeDefaults:
    """Explicit defaults shared by the persistent service and measurement CLIs."""

    name: str = "train40_cpu_arrivals_pil_exact_transport_v2"
    max_objects: int = 2
    replay_objects: int = 1
    backend: str = "prefix"
    event_query_graphs: bool = True
    event_normalization: str = "topk"
    reuse_event_encodings: bool = False
    rgb_object_chunk_size: int = 1
    parallel_branches: bool = True
    overlap_event_with_rgb_preparation: bool = True
    transport_reuse: str = "reverse-summaries"
    rgb_workers: int = 8
    shared_rgb_images: bool = False
    rgb_uint8: bool = False
    paced_raw_source: str = "memory"
    paced_prime_native_histories: bool = True
    paced_rgb_feature_cache: bool = True


DEFAULT_PROFILE = RuntimeDefaults()
