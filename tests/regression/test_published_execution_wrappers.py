"""Preserve exact published bytes of hash-bound RunPod runtime source."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

PUBLISHED_COMMIT = "2ad7a05797799caf1bcdc9202c307d244f9fe31d"
REPOSITORY = Path(__file__).resolve().parents[2]

PUBLISHED_WRAPPERS = [
    (
        "scripts/runpod_deadline_extension_0100_20261027.py",
        "86432e9cd11deb2c3b5bb982ef3d0b864b1515af1c00df12911c8623b5a65460",
    ),
    (
        "scripts/runpod_deadline_extension_20261027.py",
        "3c5e1923157ce475842396cb556665c79d2cca64d3c46691cc9c519f482c0ee0",
    ),
    (
        "scripts/runpod_event_extended_0100_20261027.py",
        "481e24b88496443b606b977892295469662e8059cabb83564643905dcc3004ab",
    ),
    (
        "scripts/runpod_event_extended_parallel_20261027.py",
        "b016595fab78ad19a016ffc69bb7d5aa8fc147c626312897a7db5aa3c1c29c42",
    ),
    (
        "scripts/runpod_event_relocated_parallel_20261027.py",
        "dba2fd4d238fdf6b6526a7776e7ba6689c7feb340274fffebe023f2d39eb630f",
    ),
    (
        "scripts/runpod_event_resource_watchdog_v14.py",
        "cf0776e4737d61524d8e7508e3da3f5514a9c1d61fe0c76c0b53b61145077e77",
    ),
]

RUNTIME_BOUND_COMMIT = "2a047bf1afc5c584e0cf6e3546dfe7bbb5883541"
RUNTIME_BOUND_SOURCES = [
    (
        "operational/runpod_migration/event_queue.py",
        "28157f567dd4f2ac30b227d6b9dd7e846913f235d3bfba1366b30137f5d662b2",
    ),
    (
        "operational/runpod_migration/pod2_rgb_handoff.py",
        "f5d5fcb2075f01830e3a00068375e827fa3433c36433a201d00f3f382324974b",
    ),
    (
        "scripts/runpod_event_queue_v1.py",
        "885438c69937cce791eff59b02f7557f0ee0208e82d5044fbec37fe17a41aa00",
    ),
    (
        "scripts/runpod_event_queue_waiter_v2.py",
        "2ecd5f7f4b41ae569ca9670b555dd3617d0a0f2849a73c44a8761dfc2a140292",
    ),
    (
        "scripts/runpod_event_resource_watchdog_v1.py",
        "8eb97e5683c5061b918da0be17ee996f1d5abe0939c01af68aca671b77ce7d5c",
    ),
    (
        "scripts/runpod_event_resource_watchdog_v12.py",
        "06a453b3facd3db2dc55a0356b8705e91daaf2da49a7ce8c737932bf04bd6944",
    ),
    (
        "scripts/runpod_event_resource_watchdog_v7.py",
        "aeb8a855a98d58e62f267f017817b8b16d9010d2bff048cdce35eba435093592",
    ),
    (
        "scripts/runpod_measure_window.py",
        "8bb610974fdd72e6a3f84f45a7abdb986518eab5cdae63c6cb3948f1a5078af4",
    ),
    (
        "scripts/runpod_rgb_context_v5.py",
        "cb349a6d514245414f508800b878f9eb3cae44f167ce6c5ec8076696d299f357",
    ),
    (
        "scripts/runpod_rgb_context_v6.py",
        "88bf04a2ad2b2938b93b1a1a2a8099de07365992ea0a1ff75ef6bd81684d19ca",
    ),
    (
        "scripts/runpod_rgb_context_v7.py",
        "ae66d985c5a2bfafac2050d5e75a32e85de9f207509fadc34d978301a43c916b",
    ),
    (
        "scripts/runpod_rgb_context_v8.py",
        "53d170fb9c880c983cf22a64659d5b986abb4d058d98cbc1acff3823d3458efe",
    ),
    (
        "scripts/runpod_rgb_input_cache_reclaim_v1.py",
        "78648b254eab5b075751fb81c2d9b7d2474b87a3f9348ba78e8cbaa9d9bc6b20",
    ),
    (
        "scripts/runpod_rgb_path_compat_v1.py",
        "4bbb2d8e5a8ec8bfead7359c9e842e7c1cb9af9526c6beb027e99e0f54f097ef",
    ),
    (
        "scripts/upload_rgb_atomic_status_v1.py",
        "c630e17f040852a29935ea4896cc753ec36cd03ac1fb4fc2c64024b9ba9b6bce",
    ),
    (
        "scripts/upload_rgb_sparse_frames_v2.py",
        "f0d0ad61f2da1cb8a9ef049f4e53f419a529d5d410388437323b612475db4677",
    ),
    (
        "scripts/upload_rgb_sparse_frames_v4.py",
        "5756d64ffc502568053eb9f5f2a18e141ed85d8e3ccb405623b7916c900c274d",
    ),
    (
        "scripts/upload_rgb_sparse_frames_v5.py",
        "27d330086abff89aff4d4a33b8f6536c737f4cafb498894fc9d26814dd8157b4",
    ),
    (
        "scripts/upload_rgb_sparse_frames_v6.py",
        "753c8899a72a7f2dcbc692e79fe18451f70fe13c6d75002771fe6a9f36a0c937",
    ),
]


@pytest.mark.parametrize(
    ("relative_path", "published_sha256"),
    PUBLISHED_WRAPPERS,
    ids=[Path(path).stem for path, _ in PUBLISHED_WRAPPERS],
)
def test_expired_execution_wrapper_matches_published_commit(
    relative_path: str, published_sha256: str
) -> None:
    """Fail if a wrapper diverges from its byte identity at the published commit."""
    path = REPOSITORY / relative_path
    assert path.is_file(), f"Missing wrapper published at {PUBLISHED_COMMIT}:{relative_path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == published_sha256, (
        f"Changed wrapper published at {PUBLISHED_COMMIT}:{relative_path}"
    )


@pytest.mark.parametrize(
    ("relative_path", "published_sha256"),
    RUNTIME_BOUND_SOURCES,
    ids=[Path(path).stem for path, _ in RUNTIME_BOUND_SOURCES],
)
def test_runtime_bound_source_matches_integration_commit(
    relative_path: str, published_sha256: str
) -> None:
    """Fail if runtime-bound source diverges from its integration-commit bytes."""
    path = REPOSITORY / relative_path
    assert path.is_file(), f"Missing runtime source at {RUNTIME_BOUND_COMMIT}:{relative_path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == published_sha256, (
        f"Changed runtime source at {RUNTIME_BOUND_COMMIT}:{relative_path}"
    )
