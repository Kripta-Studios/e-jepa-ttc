"""Contract tests for evaluation-only CPU overlap."""

import ast
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.detection.bounded_prefetch import PrefetchWorkerError
from e_jepa_ttc.detection.evaluation_prefetch import prefetch_evaluation_rows


class Cache:
    label_free_model_input = True
    dataset = range(7)

    def __init__(self, *, device: str = "cpu", labels: int = 0) -> None:
        self.device = device
        self.labels = labels
        self.visited: list[int] = []
        self.threads: set[int] = set()

    def get(self, index: int) -> tuple[SimpleNamespace, int]:
        self.threads.add(threading.get_ident())
        self.visited.append(index)
        return SimpleNamespace(
            image=SimpleNamespace(device=SimpleNamespace(type=self.device)),
            native_ground_truth=[],
            target={"labels": SimpleNamespace(numel=lambda: self.labels)},
        ), index


def test_order_values_and_exclusive_worker_ownership() -> None:
    cache = Cache()
    with prefetch_evaluation_rows(cache, lambda raw: raw * 2, lambda raw: (raw, None)) as rows:
        values = list(rows)
        assert rows.committed_cursor == 7
    assert [row[1:4] for row in values] == [(i, i * 2, (i, None)) for i in range(7)]
    assert cache.visited == list(range(7))
    assert len(cache.threads) == 1
    assert threading.get_ident() not in cache.threads


@pytest.mark.parametrize(("device", "labels"), [("cuda", 0), ("cpu", 1)])
def test_reject_gpu_or_supervised_input(device: str, labels: int) -> None:
    cache = Cache(device=device, labels=labels)
    with prefetch_evaluation_rows(cache, lambda raw: raw, lambda raw: raw) as rows:
        with pytest.raises(PrefetchWorkerError):
            next(rows)


def test_worker_failure_propagates_without_skipping_a_frame() -> None:
    def fail_on_second(raw: int) -> int:
        if raw == 1:
            raise ValueError("broken geometry")
        return raw

    with prefetch_evaluation_rows(Cache(), fail_on_second, lambda raw: raw) as rows:
        assert next(rows)[1] == 0
        with pytest.raises(PrefetchWorkerError):
            next(rows)
        assert rows.committed_cursor == 1


def test_materializer_preserves_model_decoder_scorer_and_geometry_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "prefetch_materializer", root / "scripts/materialize_runpod_evaluation_prefetch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = (root / "scripts/train_detection_campaign.py").read_text(encoding="utf-8")
    updated = module.rewrite_evaluation(original)
    names = {
        "model",
        "decode_eap",
        "evaluate_detections_v2",
        "evaluate_detections",
        "build_evaluation_ground_truth",
        "legacy_ground_truth",
        "model_inputs_to_cuda",
        "require_finite",
    }

    def calls(source: str) -> list[str]:
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
        )
        return sorted(
            ast.dump(node)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        )

    assert calls(original) == calls(updated)
    compile(updated, "candidate", "exec")
