import json
from pathlib import Path

from e_jepa_ttc.evaluation.aggregate import aggregate_metric_files


def _write_metrics(path: Path, *, seed: int, mae: float, relative: float) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "checkpoint_epoch": seed + 1,
                "splits": {
                    "test": {
                        "count": 4,
                        "metrics": {
                            "mae_s": mae,
                            "mean_abs_relative_error_pct": relative,
                            "rmse_s": mae * 2.0,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_metric_files(tmp_path: Path) -> None:
    first = tmp_path / "seed7.json"
    second = tmp_path / "seed13.json"
    _write_metrics(first, seed=7, mae=0.2, relative=5.0)
    _write_metrics(second, seed=13, mae=0.4, relative=7.0)

    payload = aggregate_metric_files(
        [first, second],
        split="test",
        metric_names=("mae_s", "mean_abs_relative_error_pct"),
    )

    assert payload["count"] == 2
    assert payload["rows"][0]["seed"] == 7
    assert payload["summary"]["mae_s"]["mean"] == 0.30000000000000004
    assert payload["summary"]["mae_s"]["std"] == 0.1
    assert payload["summary"]["mean_abs_relative_error_pct"]["mean"] == 6.0
