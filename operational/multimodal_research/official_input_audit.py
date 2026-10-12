"""Independently check sealed official inputs for group separation, causal support and drift."""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from operational.multimodal_validation.common import ROOT, RUN, read, sha, write

from .common import OUT


def main() -> None:
    dest = OUT / "test12_phase_only"
    freeze_path = dest / "MODEL_INPUT_FREEZE.json"
    frozen = read(freeze_path)
    execution = read(dest / "EXECUTION_ENVIRONMENT.json")
    assert execution["model_input_freeze_sha256"] == sha(freeze_path)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", execution["git_commit"], "--", "src", "operational"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    diagnostic_files = {
        "phase_only_attribution.py",
        "phase_only_cost.py",
        "phase_only_garl_contrasts.py",
        "phase_only_strata.py",
        "research_bundle.py",
        "research_metrics_replay.py",
        "status_report.py",
        "official_input_audit.py",
        "gpu_fcwd_cost.py",
    }
    assert all(
        name.startswith("operational/multimodal_research/") and Path(name).name in diagnostic_files
        for name in changed
    ), changed
    assert sha(dest / "INPUTS.json") == frozen["inputs_sha256"]
    inputs = read(dest / "INPUTS.json")
    for name, digest in frozen["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in frozen["parent_sha256"].items():
        assert sha(Path(name)) == digest, name
    for mode, digest in frozen["readout_sha256"].items():
        assert sha(OUT / f"fits/R_ANCHOR_PHASE_ONLY_{mode}_S7/checkpoint_last.pt") == digest
    roles, role_sources = {}, {}
    for role in ("P", "H", "V"):
        path = RUN / f"{role}_MANIFEST.json"
        manifest = read(path)
        roles[role] = set(manifest["group_ids"])
        assert manifest["role"] == role and len(roles[role]) == manifest["group_count"]
        role_sources[role] = dict(path=str(path), sha256=sha(path))
    test = {r["sequence_id"] for r in inputs["queries"]}
    assert len(test) == 12
    roles["test"] = test
    for a in roles:
        for b in roles:
            if a != b:
                assert not roles[a] & roles[b], (a, b)
    tokens = {r["sample_token"] for r in inputs["queries"]}
    assert len(tokens) == len(inputs["queries"]) == 6762
    assert tokens == set(inputs["physical_aliases"])
    lengths = {"event": Counter(), "rgb": Counter()}
    support_max = {"event": 0, "rgb": 0}
    by_token = {r["sample_token"]: r for r in inputs["queries"]}
    for query in inputs["queries"]:
        token, stamp = query["sample_token"], query["query_time_us"]
        eids, rids = query["event_history"], query["rgb_history"]
        assert 0 < len(eids) <= 8 and len(rids) <= 8
        assert len(eids) == len(set(eids)) and len(rids) == len(set(rids))
        assert eids[-1] == inputs["physical_aliases"][token]
        for mode, ids in (("event", eids), ("rgb", rids)):
            lengths[mode][len(ids)] += 1
            anchors = []
            for identity in ids:
                obs = inputs[mode + "_observations"][identity]
                anchor = obs["anchor_us"]
                assert anchor <= obs["available_us"] <= stamp
                anchors.append(anchor)
                if mode == "event":
                    assert (obs["sequence_id"], obs["track_id"]) == (
                        query["sequence_id"],
                        query["track_id"],
                    )
                    windows = obs["windows_us"]
                    assert all(a < b <= obs["sensor_anchor_us"] for a, b in windows)
                    support = stamp - anchor + obs["input_span_us"]
                else:
                    assert all(v.split("|")[0] == query["track_id"] for v in obs["frame_ids"])
                    times, available = obs["producer_times_us"], obs["producer_available_us"]
                    assert times == sorted(set(times)) and times[-1] == anchor
                    assert len(times) == len(available) == len(obs["frame_ids"])
                    assert all(t <= a <= stamp for t, a in zip(times, available, strict=True))
                    support = stamp - min(times)
                assert 0 <= support <= 650000, (token, mode, support)
                support_max[mode] = max(support_max[mode], support)
            assert anchors == sorted(set(anchors))
        canonical = by_token[inputs["physical_aliases"][token]]
        for key in ("sequence_id", "track_id", "query_time_us", "event_history", "rgb_history"):
            assert query[key] == canonical[key], (token, key)
    receipt = dict(
        utc=datetime.now(UTC).isoformat(),
        status="PASS",
        source_sha256=sha(__file__),
        frozen_model_input_sha256=sha(freeze_path),
        execution_commit=execution["git_commit"],
        all_runtime_sources_unchanged_since_execution_commit=True,
        post_freeze_diagnostic_files_changed=changed,
        role_manifest_sources=role_sources,
        sequence_groups={k: sorted(v) for k, v in roles.items()},
        all_role_sequence_groups_disjoint=True,
        all_frozen_source_and_parent_hashes_unchanged=True,
        all_6762_query_histories_same_object_and_causal=True,
        all_aliases_have_identical_event_and_rgb_histories=True,
        history_length_counts={k: dict(v) for k, v in lengths.items()},
        maximum_total_sensor_support_us=support_max,
        query_count=len(tokens),
        physical_observations=len(inputs["event_observations"]),
        optimizer_updates=0,
        test_targets_read=False,
        scope="Audit of declared public sequence/object IDs and native clocks. It verifies "
        "causal sensor support, not detector latency, metric calibration, or dataset-wide "
        "scene independence beyond the released sequence split.",
    )
    write(dest / "INPUT_AUDIT.json", receipt)
    print(
        {
            k: receipt[k]
            for k in (
                "status",
                "query_count",
                "history_length_counts",
                "maximum_total_sensor_support_us",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
