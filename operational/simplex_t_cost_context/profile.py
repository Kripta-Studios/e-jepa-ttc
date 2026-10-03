"""Measure only prepared-input head latency on a preregistered TRAIN selection."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

    from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources

    from .engine import Resources

from operational.simplex_t_h16_replication.common import (
    OUT as H16,
)
from operational.simplex_t_h16_replication.common import (
    ROOT,
    atomic_json,
    csv,
    digest,
    historical_spec,
    publish_json,
    record,
    sources,
)
from operational.simplex_t_h16_replication.common import (
    protocol as h16_protocol,
)

OUT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
DESIGN = dict(
    schema="prospective_prepared_head_profile_v1",
    role="TRAIN",
    fold=0,
    queries=64,
    selection="lowest SHA256 of TRAIN sample_token; no targets or errors",
    precision="FP32",
    device="CPU",
    numerical_threads=4,
    interop_threads=2,
    batch1_warmups=25,
    batch1_measurements=500,
    batch128_measurements=50,
    percentile_method="linear",
    context_lengths=[1, 8, 16],
    score_selection=False,
    sensor_to_decision_claim=False,
    canonical_ttc_emission_included=True,
)


def deadline_guard() -> None:
    deadline = datetime.fromisoformat(record(OUT / "WINDOW_AUTHORIZATION.json")["deadline_utc"])
    if datetime.now(UTC) >= deadline:
        raise InterruptedError("Nocturnal profile deadline reached; committed fragments retained")


def prepare_inputs(s: CampaignSources, p: dict, resource: Resources) -> tuple[dict, dict]:
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
    from e_jepa_ttc.simplex_t.expansion_sources import load_expansion_inputs

    resource.check()
    original = load_current_inputs(
        s.historical_root,
        0,
        "inner_oof",
        ancestry_sha256=s.ancestry_sha256,
        allowed_sequences=s.allowed_sequences,
    )["metadata"]
    extra = load_expansion_inputs(
        s.expansion_folds[0],
        outer=0,
        feature_count=17,
        allowed_sequences=s.expansion_sequences,
    )
    tokens = np.concatenate((original.sample_token.to_numpy(), extra.tokens))
    sequences = np.concatenate((original.sequence_id.to_numpy(), extra.sequences))
    indexes = sorted(
        range(len(tokens)), key=lambda i: hashlib.sha256(str(tokens[i]).encode()).hexdigest()
    )[:64]
    selected = dict(
        design=DESIGN,
        queries=[
            dict(query_index=i, sample_token=str(tokens[i]), sequence_id=str(sequences[i]))
            for i in indexes
        ],
        selection_uses_targets=False,
        h16_protocol_sha256=digest(H16 / "PROTOCOL.json"),
    )
    publish_json(OUT / "profiling/SELECTION.json", selected)
    arrays = {}
    for length in (1, 8, 16):
        resource.check()
        parent = s.train(historical_spec(s, 0, length))
        if parent.population != len(tokens):
            raise ValueError("profile IDs do not match historical TRAIN source order")
        arrays[length] = parent.gather(torch.tensor(indexes))[:4]
    return arrays, selected


def measure(
    model: nn.Module, inputs: tuple[Tensor, ...], label: str, resource: Resources, endpoint_sha: str
) -> dict:
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    path = OUT / "profiling" / (label + ".json")
    if path.exists():
        prior = record(path)
        if prior["endpoint_sha256"] != endpoint_sha or prior["selection_sha256"] != digest(
            OUT / "profiling/SELECTION.json"
        ):
            raise ValueError("profile binding differs")
        return prior
    model.eval()
    progress = path.with_suffix(".progress.json")
    prior = record(progress) if progress.exists() else {}
    timing = prior.get("raw_ms", [])
    if prior and (
        prior["endpoint_sha256"] != endpoint_sha
        or prior["selection_sha256"] != digest(OUT / "profiling/SELECTION.json")
    ):
        raise ValueError("profile fragment binding changed")
    with torch.inference_mode():
        for i in range(25):
            resource.check()
            deadline_guard()
            batch = tuple(v[i % 64 : i % 64 + 1] for v in inputs)
            phase_to_ttc(model(*batch)["point_phase"].to(torch.float64))
        for i in range(len(timing), 500):
            resource.check()
            deadline_guard()
            batch = tuple(v[i % 64 : i % 64 + 1] for v in inputs)
            start = time.perf_counter_ns()
            output = model(*batch)
            phase_to_ttc(output["point_phase"].to(torch.float64))
            end = time.perf_counter_ns()
            if not all(bool(torch.isfinite(v).all()) for v in output.values()):
                raise ValueError("nonfinite profiling output")
            timing.append((end - start) / 1e6)
            if len(timing) % 25 == 0:
                atomic_json(
                    progress,
                    dict(
                        raw_ms=timing,
                        endpoint_sha256=endpoint_sha,
                        selection_sha256=digest(OUT / "profiling/SELECTION.json"),
                        measured_samples=len(timing),
                        status="PROFILE_FRAGMENT_COMMITTED",
                    ),
                )
        throughput = []
        batch128 = tuple(v.repeat((2,) + (1,) * (v.ndim - 1)) for v in inputs)
        for _ in range(50):
            resource.check()
            deadline_guard()
            start = time.perf_counter_ns()
            phase_to_ttc(model(*batch128)["point_phase"].to(torch.float64))
            throughput.append((time.perf_counter_ns() - start) / 1e6)
    result = dict(
        model=label,
        status="MEASURED",
        scope="prepared_input_head_only",
        device="CPU_FP32",
        fold=0,
        batch=1,
        measurements=500,
        warmups=25,
        raw_ms=timing,
        p50_ms=float(np.percentile(timing, 50)),
        p95_ms=float(np.percentile(timing, 95)),
        min_ms=float(min(timing)),
        max_ms=float(max(timing)),
        batch128_measurements=50,
        batch128_p50_ms=float(np.median(throughput)),
        batch128_windows_per_second=128000 / float(np.median(throughput)),
        parameters=sum(v.numel() for v in model.parameters()),
        host=platform.node(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        processor=platform.processor(),
        endpoint_sha256=endpoint_sha,
        selection_sha256=digest(OUT / "profiling/SELECTION.json"),
        rss_snapshot=record(OUT / "execution/RESOURCES.json"),
        current_roi_context_preparation_included=False,
        expert_execution_included=False,
        sensor_to_decision_latency=False,
        canonical_ttc_emission_included=True,
    )
    publish_json(path, result)
    print(
        json.dumps({k: result[k] for k in ("model", "scope", "measurements", "p50_ms", "p95_ms")}),
        flush=True,
    )
    return result


def full_route_admission(p: dict) -> dict:
    """Inspect bounded resource interfaces, never private fit results or raw payloads."""
    import psutil

    paths = record(Path(p["launch"]["local_paths"]))
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack = record(ack_path)
    owners = []
    for name in ack["resources"].get("active_owner_files", []):
        path = Path(name)
        state = record(path) if path.exists() else {}
        pid = state.get("pid", state.get("process_id"))
        alive = bool(pid and psutil.pid_exists(pid))
        owners.append(
            dict(
                path=str(path),
                exists=path.exists(),
                pid=pid,
                alive=alive,
                sha256=digest(path) if path.exists() else None,
            )
        )
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        observation = dict(returncode=gpu.returncode, output=gpu.stdout, error=gpu.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        observation = dict(status="UNAVAILABLE", reason=str(exc))
    lease = ack["resources"].get("exclusive_gpu_and_heavy_io_slot")
    admission = dict(
        status="BLOCKED_NO_EXCLUSIVE_GPU_HEAVY_IO_LEASE"
        if not lease
        else "LEASE_REQUIRES_LIVE_VALIDATION",
        observed_at_utc=datetime.now(UTC).isoformat(),
        coordination_ack=str(ack_path),
        coordination_ack_sha256=digest(ack_path),
        acknowledged_slot=lease,
        resource_owner=ack["resources"].get("owner"),
        current_owner_metadata=owners,
        gpu_process_observation=observation,
        eap_root=str(paths["eap_root"]),
        eap_root_exists=Path(paths["eap_root"]).is_dir(),
        cached_train_available=True,
        raw_payloads_read=False,
        missing_dependency=(
            "Explicit exclusive GPU/heavy-I/O slot from the shared resource owner; "
            "no such lease is present in the acknowledged interface."
        ),
        no_owner_or_stage70_changes=True,
    )
    atomic_json(OUT / "profiling/FULL_ROUTE_ADMISSION.json", admission)
    return admission


def main() -> None:
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.model import TemporalConfig

    from .engine import EXEC, Resources, bound_source, load_new_endpoint, protocol

    resource = Resources()
    resource.check()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    publish_json(OUT / "PROFILING_DESIGN.json", DESIGN)
    p16, _ = h16_protocol()
    full = full_route_admission(p16)
    s = sources(p16)
    arrays, _ = prepare_inputs(s, p16, resource)
    rows = []
    stage = record(Path(p16["launch"]["publications"]["T2"]["endpoints"]))
    h1 = next(
        r
        for r in stage["fits"]
        if r["fit"]["name"] == "TPR-D1-H1-C160" and r["fit"]["seed"] == 7 and r["fit"]["fold"] == 0
    )
    candidates = [
        (
            "H1_SEED7",
            h1,
            Path(p16["launch"]["publications"]["T2"]["checkpoint_root"]) / h1["checkpoint"],
            1,
            stage["scientific_freeze_sha256"],
        )
    ]
    for history in (8, 16):
        control = next(
            v
            for v in p16["historical_controls"].values()
            if v["endpoint"]["fit"]["name"] == f"TPR-D1-H{history}-C160"
            and v["endpoint"]["fit"]["seed"] == 7
            and v["endpoint"]["fit"]["fold"] == 0
        )
        endpoint = control["endpoint"]
        candidates.append(
            (
                f"H{history}_SEED7",
                endpoint,
                Path(control["checkpoint"]),
                history,
                stage["scientific_freeze_sha256"],
            )
        )
    for label, row, checkpoint, length, freeze in candidates:
        resource.check()
        model = load_endpoint(
            checkpoint,
            TemporalConfig(**row["model"]),
            seed=7,
            freeze_sha256=freeze,
            train_source_sha256=row["train_source_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        rows.append(measure(model, arrays[length], label, resource, row["checkpoint_sha256"]))
        del model
        gc.collect()
    if (OUT / "PROTOCOL_COST_CONTEXT.json").exists():
        p, pin = protocol()
        selection = record(OUT / "profiling/SELECTION.json")
        for family in ("N2", "N3"):
            seal = EXEC / f"{family}_ENDPOINTS.json"
            if not seal.exists():
                continue
            for row in record(seal)["fits"]:
                if row["fold"] != 0:
                    continue
                resource.check()
                parent = bound_source(s, p, row, "inner_oof")
                inputs = parent.gather(
                    torch.tensor([v["query_index"] for v in selection["queries"]])
                )[:4]
                model, _ = load_new_endpoint(row, p, pin)
                rows.append(measure(model, inputs, row["arm"], resource, row["checkpoint_sha256"]))
                del parent, model
                s.release()
                gc.collect()
    columns = [
        "model",
        "status",
        "scope",
        "device",
        "fold",
        "batch",
        "measurements",
        "p50_ms",
        "p95_ms",
        "parameters",
        "batch128_windows_per_second",
        "endpoint_sha256",
        "selection_sha256",
    ]
    csv(OUT / "HEAD_COST.csv", pd.DataFrame([{k: r[k] for k in columns} for r in rows]))
    atomic_json(
        OUT / "N4_COST_STATUS.json",
        dict(
            prepared_head="COMPLETE",
            models=len(rows),
            full_route="NOT_MEASURED",
            missing_dependency=full["missing_dependency"],
            full_route_admission_sha256=digest(OUT / "profiling/FULL_ROUTE_ADMISSION.json"),
            route_cost_does_not_follow_from_head_latency=True,
        ),
    )


if __name__ == "__main__":
    main()
