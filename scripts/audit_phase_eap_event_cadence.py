"""Audit TRAIN40 history clock inputs without loading targets or running producers."""

import argparse
import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    """Bind the history index to the prior current-query audit and report identity scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--current-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior evidence; choose a new output")
    freeze_path = args.current_audit / "INPUT_FREEZE.json"
    complete_path = args.current_audit / "INPUTS_COMPLETE.json"
    freeze, complete = read(freeze_path), read(complete_path)
    if complete["status"] != "COMPLETE" or complete["freeze_sha256"] != file_sha(freeze_path):
        raise ValueError("Current-query input audit is incomplete or changed")
    archive_name = complete["modalities"]["event"]["archive"]
    archive_path = args.archive_root / archive_name
    archive_sha = file_sha(archive_path)
    if archive_sha != freeze["archives"][archive_name]:
        raise ValueError("Input archive differs from sealed current-query evidence")
    current_path = args.current_audit / "EVENT_INPUTS.npz"
    if file_sha(current_path) != complete["artifacts"][current_path.name]:
        raise ValueError("Current-query features changed")
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive if m.name.endswith("/TRAIN40_INDEX.npz")]
        if len(members) != 1:
            raise ValueError("Require one unambiguous TRAIN40 index")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("TRAIN40 index is not a file")
        payload = stream.read()
    names = ("tokens", "sequences", "delta_t_s", "windows_us")
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        tokens, groups, dt, windows = (source[name] for name in names)
    with np.load(current_path, allow_pickle=False) as source:
        current_tokens, current_groups = source["token"], source["group"]
        current_dt = source["features"][:, 128]
    if (
        dt.shape != (88744,)
        or dt.dtype != np.float32
        or windows.shape != (88744, 3, 2)
        or not np.isfinite(dt).all()
        or not (dt > 0).all()
        or not np.array_equal(tokens, current_tokens)
        or not np.array_equal(groups, current_groups)
        or not np.array_equal(dt.view(np.int32), current_dt.view(np.int32))
    ):
        raise ValueError("History index and current-query population/clocks differ")
    if not (windows[:, :, 1] > windows[:, :, 0]).all():
        raise ValueError("Nonpositive physical event windows")
    intervals_us = windows[:, -1, 1] - windows[:, -2, 1]
    identity = dt == np.float32(0.1)
    source_paths = [
        Path(__file__),
        Path("operational/final_refit/event_history.py"),
        Path("operational/phase_timewarp/canonical_pair.py"),
    ]
    atomic_receipt(
        args.output,
        {
            "status": "CLOCKS_AUDITED",
            "scope": "TRAIN40 event-history clock precondition; not complete eAP preservation",
            "targets_read": False,
            "producer_calls": 0,
            "optimizer_updates": 0,
            "input_freeze_sha256": file_sha(freeze_path),
            "inputs_complete_sha256": file_sha(complete_path),
            "archive_sha256": archive_sha,
            "index_member": members[0].name,
            "index_member_sha256": hashlib.sha256(payload).hexdigest(),
            "index_arrays_read": list(names),
            "current_arrays_read": ["features", "token", "group"],
            "rows": len(dt),
            "groups": len(set(groups.tolist())),
            "identity_order_and_clock_bits_equal_current_query": True,
            "float32_100ms_rows": int(identity.sum()),
            "nonidentity_rows": int((~identity).sum()),
            "physical_last_interval_us_quantiles": np.quantile(
                intervals_us, [0, 0.01, 0.5, 0.99, 1]
            ).tolist(),
            "physical_last_interval_not_100000_us": int((intervals_us != 100000).sum()),
            "source_hashes": {str(p): file_sha(p) for p in source_paths},
            "code_trace": [
                "event_history._query_plan takes delta_seconds from index delta_t_s",
                "event_history._fresh_pair_phases repeats that delta over all 16 producer slots",
                "CanonicalPair identity branch calls original with unchanged full batch once",
                "Identity output preserves original phase bits including historical log epsilon",
            ],
            "interpretation": (
                "For finite valid FP32 history inputs using this materialization path, all indexed "
                "queries meet the identity-clock precondition; PAIR correction is inactive. "
                "This is structural evidence, not an executed history/context/head comparison."
            ),
            "remaining": [
                "Verify deployed history path and producer/checkpoint ancestry",
                "Validate historical features and reconstruct contexts/head "
                "with consistent ancestry",
                "Audit RGB histories with measured varying cadence",
                "Keep OOF caches distinct from deployment inputs",
            ],
        },
    )
    print({"rows": len(dt), "identity": int(identity.sum()), "output": str(args.output)})


if __name__ == "__main__":
    main()
