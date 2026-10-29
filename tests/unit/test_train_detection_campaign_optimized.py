from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts/train_detection_campaign_optimized.py"
    spec = importlib.util.spec_from_file_location("optimized_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_injects_one_store_and_closes_it(monkeypatch: Any, tmp_path: Path) -> None:
    module = _module()
    performance = module.PerformanceConfig.from_dict(
        {
            "schema": "eap_detection_performance_config_v1",
            "reader": "shared_indexed_tar_v1",
            "max_open_handles": 2,
            "shared_store": True,
            "prefetch_batches": 0,
            "instrument_phases": [
                "tar_index",
                "payload_read",
                "png_decode",
                "prepare",
                "h2d",
                "compute",
                "evaluation",
            ],
            "expected_indexed_reader_sha256": "a" * 64,
            "expected_indexed_store_sha256": "b" * 64,
        }
    )
    config = SimpleNamespace(
        sha256="config", data=SimpleNamespace(root=tmp_path, label_cache_sequences=24)
    )

    class Store:
        root = tmp_path.resolve()
        closed = False
        views: list[tuple[str, ...]] = []

        def metrics(self) -> dict[str, int | float | bool]:
            return {
                "index_build_seconds": 1.0,
                "read_count": 0,
                "read_bytes": 0,
                "read_seconds": 0.0,
                "open_handle_count": 0,
                "closed": self.closed,
            }

        def dataset_view(self, tokens: Any, **_: Any) -> object:
            self.views.append(tuple(tokens))
            return object()

        def close(self) -> None:
            self.closed = True

    store = Store()
    monkeypatch.setattr(module.PerformanceConfig, "load", lambda _: (performance, {}))
    monkeypatch.setattr(
        module, "sha256_file", lambda path: "b" * 64 if "store" in path.name else "a" * 64
    )
    monkeypatch.setattr(module, "load_campaign_config", lambda _: config)
    monkeypatch.setattr(module, "_build_store", lambda *_: (store, {"union_count": 28054}))

    def fake_run(*_: Any, **kwargs: Any) -> dict[str, object]:
        view = kwargs["dataset_factory"](tmp_path, ["a", "b"], config)
        assert view is not None
        assert kwargs["runtime_extension"]["union_count"] == 28054
        owned = {path.name for path in kwargs["extra_owned_paths"]}
        assert "optimized_campaign.py" in owned
        assert "indexed_rgb_data.py" in owned
        assert "indexed_rgb_store.py" in owned
        kwargs["performance_observer"]("prepare", 0.25)
        metrics = kwargs["performance_metrics"]()
        assert metrics["prefetch_batches"] == 0
        return {"status": "COMPLETED"}

    monkeypatch.setattr(module.ordinary, "run", fake_run)
    result = module.run(Path("config.json"), Path("performance.json"), tmp_path / "run")
    assert result["status"] == "COMPLETED"
    assert store.views == [("a", "b")]
    assert store.closed


def test_pending_stop_is_excluded_from_periodic_checkpoint() -> None:
    module = _module()
    assert not module.ordinary.should_publish_periodic_checkpoint(
        8, checkpoint_interval=4, evaluation_updates=(), stop_after_update=8
    )
    assert module.ordinary.should_publish_periodic_checkpoint(
        8, checkpoint_interval=4, evaluation_updates=(), stop_after_update=None
    )


def test_full_evaluation_failure_publishes_no_checkpoint() -> None:
    module = _module()
    published: list[str] = []

    def fail() -> None:
        raise RuntimeError("full evaluation failed")

    with pytest.raises(RuntimeError, match="full evaluation failed"):
        module.ordinary.commit_evaluation_artifacts(
            run_full_evaluation=fail,
            publish_selected=lambda: published.append("selected"),
            publish_last=lambda: published.append("last"),
        )
    assert published == []


def test_scheduled_and_full_transactions_publish_only_after_evaluation() -> None:
    module = _module()
    events: list[str] = []
    module.ordinary.commit_evaluation_artifacts(
        run_full_evaluation=lambda: events.append("full"),
        publish_selected=lambda: events.append("selected"),
        publish_last=lambda: events.append("last"),
    )
    assert events == ["full", "selected", "last"]
