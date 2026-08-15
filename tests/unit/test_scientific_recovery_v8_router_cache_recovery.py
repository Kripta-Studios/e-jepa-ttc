from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import sign_artifact


def _load_module():
    path = Path('scripts/train_scientific_recovery_v8_router_expert.py')
    spec = importlib.util.spec_from_file_location('v8_router_expert_cache_recovery', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def test_router_expert_uses_only_signed_storage_equivalent_recovery_when_historical_shards_missing(tmp_path: Path) -> None:
    module = _load_module()
    module.ROOT = tmp_path

    historical_dir = tmp_path / 'artifacts/cache/historical'
    historical_manifest_path = historical_dir / 'manifest.json'
    historical_manifest = {
        'artifact_sha256': 'h' * 64,
        'split_sha256': 's' * 64,
        'split_counts': {'train': 8192},
        'shards': [{'split': 'train', 'path': 'train/shard-00000.pt', 'count': 8192}],
    }
    _write_json(historical_manifest_path, historical_manifest)
    historical_sha = module._sha(historical_manifest_path)

    recovery_dir = tmp_path / 'artifacts/scientific_recovery_v8/cache/autopsy_v4_recovered_v1'
    recovered_shard = recovery_dir / 'train/shard-00000.pt'
    recovered_shard.parent.mkdir(parents=True, exist_ok=True)
    recovered_shard.write_bytes(b'not-empty')
    recovered_manifest = {
        'artifact_sha256': 'r' * 64,
        'split_sha256': 's' * 64,
        'split_counts': {'train': 8192},
        'shards': [{'split': 'train', 'path': 'train/shard-00000.pt', 'count': 8192}],
    }
    recovered_manifest_path = recovery_dir / 'manifest.json'
    _write_json(recovered_manifest_path, recovered_manifest)

    recovery = {
        'artifact_type': 'scientific_recovery_v8_autopsy_v4_cache_recovery_v1',
        'status': 'completed',
        'historical_manifest': {
            'file_sha256': historical_sha,
            'artifact_sha256': historical_manifest['artifact_sha256'],
        },
        'recovered_manifest': {
            'file_sha256': module._sha(recovered_manifest_path),
            'artifact_sha256': recovered_manifest['artifact_sha256'],
        },
        'historical_split_sha256': 's' * 64,
        'recovered_split_sha256': 's' * 64,
        'semantic_preprocessing_inherited_from_historical_manifest': True,
        'expected_frozen_rows': 8192,
        'sealed_splits_opened': False,
    }
    sign_artifact(recovery)
    _write_json(recovery_dir / 'RECOVERY.json', recovery)

    base = {
        'data': {
            'cache_manifest': 'artifacts/cache/historical/manifest.json',
            'cache_manifest_sha256': historical_sha,
            'cache_artifact_sha256': historical_manifest['artifact_sha256'],
            'expected_source_train_rows': 8192,
        }
    }
    selected, payload, provenance = module._resolve_training_cache(base)

    assert selected == recovered_manifest_path.resolve()
    assert payload['artifact_sha256'] == recovered_manifest['artifact_sha256']
    assert provenance['used'] is True
    assert provenance['storage_only_recovery'] is True
    assert provenance['semantic_preprocessing_unchanged'] is True


def test_router_expert_rejects_recovery_with_changed_split_identity(tmp_path: Path) -> None:
    module = _load_module()
    module.ROOT = tmp_path

    historical_dir = tmp_path / 'artifacts/cache/historical'
    historical_manifest_path = historical_dir / 'manifest.json'
    historical_manifest = {
        'artifact_sha256': 'h' * 64,
        'split_sha256': 'a' * 64,
        'split_counts': {'train': 8192},
        'shards': [{'split': 'train', 'path': 'train/missing.pt', 'count': 8192}],
    }
    _write_json(historical_manifest_path, historical_manifest)
    historical_sha = module._sha(historical_manifest_path)

    recovery_dir = tmp_path / 'artifacts/scientific_recovery_v8/cache/autopsy_v4_recovered_v1'
    shard = recovery_dir / 'train/shard.pt'
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b'x')
    recovered_manifest = {
        'artifact_sha256': 'r' * 64,
        'split_sha256': 'b' * 64,
        'split_counts': {'train': 8192},
        'shards': [{'split': 'train', 'path': 'train/shard.pt', 'count': 8192}],
    }
    recovered_manifest_path = recovery_dir / 'manifest.json'
    _write_json(recovered_manifest_path, recovered_manifest)
    recovery = {
        'artifact_type': 'scientific_recovery_v8_autopsy_v4_cache_recovery_v1',
        'status': 'completed',
        'historical_manifest': {'file_sha256': historical_sha, 'artifact_sha256': 'h' * 64},
        'recovered_manifest': {'file_sha256': module._sha(recovered_manifest_path), 'artifact_sha256': 'r' * 64},
        'historical_split_sha256': 'a' * 64,
        'recovered_split_sha256': 'b' * 64,
        'semantic_preprocessing_inherited_from_historical_manifest': True,
        'expected_frozen_rows': 8192,
        'sealed_splits_opened': False,
    }
    sign_artifact(recovery)
    _write_json(recovery_dir / 'RECOVERY.json', recovery)

    base = {'data': {'cache_manifest': 'artifacts/cache/historical/manifest.json', 'cache_manifest_sha256': historical_sha, 'cache_artifact_sha256': 'h' * 64, 'expected_source_train_rows': 8192}}

    import pytest
    with pytest.raises(ValueError, match='split SHA-256'):
        module._resolve_training_cache(base)
