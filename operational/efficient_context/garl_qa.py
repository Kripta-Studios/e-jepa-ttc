"""Source-native parity and input admission, using only authorized TRAIN records."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest


def run(c: Campaign) -> dict:
    """Check all index intervals cheaply, and hash-select32 exact native encodings."""
    import numpy as np
    import torch

    from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
    from e_jepa_ttc.data.garl_official_preprocessing import (
        official_resize_feature,
        official_square_box,
    )
    from e_jepa_ttc.efficient_context.garl_input import native_feature, read_native_window
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    from .garl_train import records

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    c.require_resources()
    rows = records(c)
    sys.dont_write_bytecode = True
    sys.path.insert(0, c.local["garl_code_candidate"])
    get_timevolume_roi_np = importlib.import_module(
        "garl_ttc.datasets.event_representation"
    ).get_timevolume_roi_np
    extract_from_h5_by_timewindow = importlib.import_module(
        "garl_ttc.utils.events"
    ).extract_from_h5_by_timewindow

    pool = ReaderPool()
    failures = []
    cases = []
    try:
        for rank, (token, row) in enumerate(
            sorted(rows.items(), key=lambda v: (v[1]["sequence_id"], v[0]))
        ):
            if rank % 128 == 0:
                c.require_resources()
            reader = pool.get(c.raw / row["sequence_id"] / "events.h5")
            index = reader.datasets["ms_to_idx"]
            for start, end in row["event_windows_us"]:
                start, end = int(start), int(end)
                if (
                    start < 0
                    or end // 1000 >= len(index)
                    or end > int(reader.datasets["events/t"][-1])
                    or int(index[end // 1000]) <= int(index[start // 1000])
                ):
                    failures.append(
                        {"token": token, "reason": "EMPTY_OR_UNSUPPORTED_NATIVE_WINDOW"}
                    )
            target = float(row["frame_ttc"][-1])
            if target == 0 or not np.isfinite(1 - 0.1 / target) or 1 - 0.1 / target <= 0:
                failures.append({"token": token, "reason": "SOURCE_LHR_TARGET_LOG_DOMAIN_INVALID"})
        chosen = sorted(rows, key=lambda t: hashlib.sha256(t.encode()).hexdigest())[:32]
        for token in chosen:
            c.require_resources()
            row = rows[token]
            path = c.raw / row["sequence_id"] / "events.h5"
            windows = row["event_windows_us"]
            expected = extract_from_h5_by_timewindow(
                path,
                [int(v[0]) for v in windows],
                [int(v[1]) for v in windows],
                5,
                list(EAP_IMAGE_SIZE[::-1]),
            )
            for i, (start, end) in enumerate(windows):
                actual = read_native_window(pool, path, int(start), int(end))
                exact_sensor = all(np.array_equal(actual[k], expected[i][k]) for k in actual)
                square = official_square_box([tuple(v) for v in row["boxes_xyxy"]], i)
                source, _ = get_timevolume_roi_np(
                    np.asarray(square, np.int16),
                    expected[i]["x"],
                    expected[i]["y"],
                    expected[i]["t"],
                )
                if source is None:
                    raise ValueError("source native encoder returned no tensor")
                a = native_feature(actual, square)
                b = official_resize_feature(source, (128, 128))
                exact_tensor = torch.equal(a, b)
                cases.append(
                    {
                        "token": token,
                        "endpoint": i,
                        "sensor_exact": exact_sensor,
                        "tensor_exact": exact_tensor,
                        "max_abs": float((a - b).abs().max()),
                        "dtype": str(a.dtype),
                        "events": len(actual["t"]),
                    }
                )
                if not exact_sensor or not exact_tensor:
                    failures.append({"token": token, "reason": "NATIVE_SOURCE_PARITY_FAILED"})
            atomic_json(c.out / "garl/INPUT_QA_PROGRESS.json", {"checked": len(cases)})
        result = {
            "status": "PASSED" if not failures else "BLOCKED_DEPENDENCY",
            "optimizer_updates": 0,
            "training_queries": len(rows),
            "selected_by": "sha256(token), no GT selection",
            "cases": cases,
            "failures": failures[:64],
            "failure_count": len(failures),
            "implementation_sha256": digest(Path(__file__)),
        }
        atomic_json(c.out / "garl/INPUT_QA.json", result)
        return result
    finally:
        pool.close()


def main() -> int:
    """Native-input admission CLI, with actionable failures and no optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    result = run(Campaign(args.protocol))
    print(result["status"], result["failure_count"], flush=True)
    return 0 if result["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
