from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.data.object_event_v4_31 import (
    ADAPT_SEQUENCES,
    OWNERSHIP_MARKER,
    PROJECTED_COLUMNS,
    SOURCE_SHA256,
    SPLIT_PATH,
    allocate_quotas,
    sha256_file,
    strict_json,
)
from scripts.analyze_object_event_v4_31_operator_audit import _stage2_metrics
from scripts.preflight_object_event_v4_31 import run
from scripts.sanitize_object_event_v4_30_stage2 import sanitize


def test_allowlist_removes_mixed_targets(tmp_path: Path) -> None:
    sources = {}
    for seed in (7, 13, 23):
        source = tmp_path / f"mixed_{seed}.npz"
        np.savez(
            source,
            oof_row_index=np.arange(2048),
            log_eta=np.full(2048, float(seed)),
            endpoint_swap_log_eta=-np.full(2048, float(seed)),
            unknown=np.zeros(2048),
            row_identity=np.asarray([f"row-{index}" for index in range(2048)]),
            ttc_s=np.ones(2048),
        )
        sources[seed] = source
    out = tmp_path / "sanitized"
    manifest = sanitize(sources, out)
    assert manifest["source_contains_forbidden_fields"] is True
    assert set(np.load(out / "seed_7.npz").files) == {
        "row_index",
        "log_eta",
        "endpoint_swap_log_eta",
        "unknown",
        "row_identity",
    }
    assert manifest["count_per_seed"] == 2048
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    assert all(
        isinstance(outputs.get(str(seed)), dict) and "sha256" in outputs[str(seed)]
        for seed in (7, 13, 23)
    )


def test_stage2_unknown_mask_is_independent_and_fail_closed(tmp_path: Path) -> None:
    sources = {}
    for seed in (7, 13, 23):
        path = tmp_path / f"source_{seed}.npz"
        np.savez(
            path,
            oof_row_index=np.arange(2048),
            log_eta=np.linspace(-1.0, 1.0, 2048) * seed,
            endpoint_swap_log_eta=np.linspace(1.0, -1.0, 2048) * seed,
            unknown=np.ones(2048, dtype=bool),
            row_identity=np.asarray([f"row-{index}" for index in range(2048)]),
        )
        sources[seed] = path
    output = tmp_path / "stage2"
    sanitize(sources, output)
    _, evidence_complete, gates_pass = _stage2_metrics(output)
    assert not evidence_complete and not gates_pass


def test_stage2_rejects_source_output_overlap_before_mutation(tmp_path: Path) -> None:
    sources: dict[int, Path] = {}
    for seed in (7, 13, 23):
        path = tmp_path / f"overlap_{seed}.npz"
        np.savez(
            path,
            oof_row_index=np.arange(2048),
            log_eta=np.linspace(-1.0, 1.0, 2048) * seed,
            endpoint_swap_log_eta=np.linspace(1.0, -1.0, 2048) * seed,
            unknown=np.zeros(2048, dtype=bool),
            row_identity=np.asarray([f"row-{index}" for index in range(2048)]),
        )
        sources[seed] = path
    with pytest.raises(ValueError, match="must not overlap"):
        sanitize(sources, sources[7])


def test_sanitizer_cli_executes_cpu_fixture_with_output_dir(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    arguments = [sys.executable, str(root / "scripts/sanitize_object_event_v4_30_stage2.py")]
    for seed in (7, 13, 23):
        source = tmp_path / f"cli_{seed}.npz"
        np.savez(
            source,
            oof_row_index=np.arange(2048),
            log_eta=np.linspace(-1.0, 1.0, 2048) * seed,
            endpoint_swap_log_eta=np.linspace(1.0, -1.0, 2048) * seed,
            unknown=np.zeros(2048, dtype=bool),
            row_identity=np.asarray([f"row-{index}" for index in range(2048)]),
        )
        arguments.extend(["--source", f"{seed}={source}"])
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [*arguments, "--output-dir", str(output)],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "manifest.json").is_file()


def test_builder_and_analyzer_cli_bind_output_dir_before_cpu_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "audit_version: object_event_v4_31_operator_audit_v1",
                "source:",
                "  train_parquet: E:/GarlTTC_dataset/data/train.parquet",
                "  sha256: absent",
            ]
        ),
        encoding="utf-8",
    )
    for script in (
        "build_object_event_v4_31_sanitized_cache.py",
        "analyze_object_event_v4_31_operator_audit.py",
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / script),
                "--config",
                str(config),
                "--cache",
                str(tmp_path / "absent-cache"),
                "--output-dir",
                str(tmp_path / f"{script}-output"),
            ]
            if script.startswith("analyze")
            else [
                sys.executable,
                str(root / "scripts" / script),
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / f"{script}-output"),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
        assert result.returncode == 2
        assert "AttributeError" not in result.stderr


class _Array:
    def __init__(
        self, shape: tuple[int, ...], dtype: np.dtype, values: np.ndarray | None = None
    ) -> None:
        self.shape, self.dtype = shape, dtype
        self._values = values

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: object = None) -> np.ndarray:
        del dtype
        return (
            self._values if self._values is not None else np.ones(self.shape[0], dtype=self.dtype)
        )


def _preflight_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    event_root = tmp_path / "event-root"
    split_sha = sha256_file(SPLIT_PATH)
    source_sha = SOURCE_SHA256
    cache = tmp_path / "cache"
    cache.mkdir()
    events = cache / "events.npy"
    delta = cache / "delta_t_s.npy"
    rows = cache / "rows.jsonl"
    events.write_bytes(b"events")
    delta.write_bytes(b"delta")
    quota = allocate_quotas(full=False)
    entries: list[dict[str, object]] = []
    index = 0
    for sequence, count in quota.items():
        pool = "adaptation" if sequence in ADAPT_SEQUENCES else "audit"
        for _ in range(count):
            entries.append(
                {
                    "row_index": index,
                    "row_sha256": f"{index:064x}",
                    "sequence_id": sequence,
                    "pool": pool,
                    "delta_t_s": 0.1,
                }
            )
            index += 1
    rows.write_text("\n".join(json.dumps(x) for x in entries), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "audit_version: object_event_v4_31_operator_audit_v1",
                "source:",
                "  train_parquet: E:/GarlTTC_dataset/data/train.parquet",
                f"  sha256: {source_sha}",
                f'event_root: "{event_root.as_posix()}"',
                f'split: "{SPLIT_PATH.as_posix()}"',
                f"split_sha256: {split_sha}",
            ]
        ),
        encoding="utf-8",
    )
    raw_config = {
        "audit_version": "object_event_v4_31_operator_audit_v1",
        "source": {
            "train_parquet": "E:/GarlTTC_dataset/data/train.parquet",
            "sha256": source_sha,
        },
        "event_root": str(event_root.resolve()),
        "split": str(SPLIT_PATH.resolve()),
        "split_sha256": split_sha,
    }
    raw_config["source"]["train_parquet"] = str(
        Path("E:/GarlTTC_dataset/data/train.parquet").resolve()
    )
    config_identity = hashlib.sha256(strict_json(raw_config).encode("utf-8")).hexdigest()
    source_identity = f"{Path('E:/GarlTTC_dataset/data/train.parquet').resolve()}:{source_sha}"
    (cache / OWNERSHIP_MARKER).write_text(
        json.dumps(
            {
                "artifact": "object_event_v4_31",
                "owner": "e_jepa_ttc",
                "config_identity": config_identity,
                "source_identity": source_identity,
            }
        )
    )
    manifest = {
        "artifact_type": "object_event_v4_31_sanitized_cache_v1",
        "schema_version": "1.0",
        "evidence_type": "sanitized_event_roi_cache",
        "code_commit": "unavailable",
        "protocol_version": "object_event_v4_31_train_only_v1",
        "protocol_sha256": split_sha,
        "created_at": "2026-08-09T00:00:00+00:00",
        "artifact_sha256": "a" * 64,
        "mode": "diagnostic",
        "count": 512,
        "events": {
            "path": "events.npy",
            "dtype": "float16",
            "shape": [512, 3, 12, 128, 128],
            "sha256": sha256_file(events),
        },
        "delta_t_s": {
            "path": "delta_t_s.npy",
            "dtype": "float32",
            "sha256": sha256_file(delta),
        },
        "rows_path": "rows.jsonl",
        "rows_sha256": sha256_file(rows),
        "source": {
            "path": str(Path("E:/GarlTTC_dataset/data/train.parquet").resolve()),
            "sha256": source_sha,
            "projection": list(PROJECTED_COLUMNS),
        },
        "representation": {
            "id": "v4_30_common_roi",
            "interval": "[start,end)",
            "shape": [3, 12, 128, 128],
        },
        "provenance": {"boxes_transient_only": True, "targets_opened": False},
        "opened_paths": [str(Path("E:/GarlTTC_dataset/data/train.parquet").resolve())],
        "split": {
            "path": str(SPLIT_PATH.resolve()),
            "sha256": split_sha,
            "version": "object_event_v4_31_train_only_v1",
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))

    def fake_load(path: Path, **_: object) -> object:
        if Path(path).name == "events.npy":
            return _Array((512, 3, 12, 128, 128), np.dtype("float16"))
        return np.full(512, 0.1, dtype=np.float32)

    monkeypatch.setattr("scripts.preflight_object_event_v4_31.np.load", fake_load)
    return config, cache


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "full"),
        ("count", 511),
        ("events_hash", "bad"),
        ("delta_hash", "bad"),
        ("rows_hash", "bad"),
    ],
)
def test_preflight_adversarial_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config, cache = _preflight_fixture(tmp_path, monkeypatch)
    assert run(config, cache, full=False)["status"] == "passed"
    manifest = json.loads((cache / "manifest.json").read_text())
    if field == "events_hash":
        manifest["events"]["sha256"] = value
    elif field == "delta_hash":
        manifest["delta_t_s"]["sha256"] = value
    elif field == "rows_hash":
        manifest["rows_sha256"] = value
    else:
        manifest[field] = value
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(config, cache, full=False)


@pytest.mark.parametrize("mutation", ["marker", "index", "pool", "sequence", "duplicate"])
def test_preflight_rejects_adversarial_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config, cache = _preflight_fixture(tmp_path, monkeypatch)
    assert run(config, cache, full=False)["status"] == "passed"
    if mutation == "marker":
        (cache / OWNERSHIP_MARKER).write_text(
            json.dumps({"artifact": "wrong", "owner": "e_jepa_ttc"})
        )
    else:
        rows = [
            json.loads(line)
            for line in (cache / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if mutation == "index":
            rows[0]["row_index"] = 11
        elif mutation == "pool":
            rows[0]["pool"] = "audit"
        elif mutation == "sequence":
            rows[0]["sequence_id"] = "not-in-locked-split"
        else:
            rows[1]["row_sha256"] = rows[0]["row_sha256"]
        (cache / "rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        manifest["rows_sha256"] = sha256_file(cache / "rows.jsonl")
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        run(config, cache, full=False)


@pytest.mark.parametrize("bad_delta", [0.0, float("nan")])
def test_preflight_rejects_nonpositive_or_nonfinite_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_delta: float
) -> None:
    config, cache = _preflight_fixture(tmp_path, monkeypatch)

    def fake_bad_delta(path: Path, **_: object) -> object:
        if Path(path).name == "events.npy":
            return _Array((512, 3, 12, 128, 128), np.dtype("float16"))
        return np.full(512, bad_delta, dtype=np.float32)

    monkeypatch.setattr("scripts.preflight_object_event_v4_31.np.load", fake_bad_delta)
    with pytest.raises(ValueError):
        run(config, cache, full=False)
