"""Measure frozen fusion routes using the exact admitted historical producer sources."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.range_transfer.common import freeze, read, sha, write
from operational.rgb_port import train_producers
from operational.rgb_port.infer_experts import _producer
from operational.rgb_port_revision import migration

from . import fusion_cost
from .common import ROOT, RUN, owner
from .train import committed_sources

OUT = RUN / "fusion/cost/raw_e2e_native_v1"
MODEL = "src/e_jepa_ttc/models/causal_scale_ttc.py"
TRANSPORT = "src/e_jepa_ttc/models/local_transport.py"


def load_private(name: str, path: Path) -> ModuleType:
    """Load archived native definitions without replacing project import modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def admit() -> dict:
    """Validate original freezes normally; permit no unreviewed numerical source drift."""
    historical = owner().parents[1]
    with patch.object(migration, "ROOT", historical):
        manifest = migration.validate(owner())
    equivalence = {}
    for relative in manifest["files"]:
        current, original = ROOT / relative, historical / relative
        equal = current.read_bytes().replace(b"\r\n", b"\n") == original.read_bytes().replace(
            b"\r\n", b"\n"
        )
        if not equal and relative != MODEL:
            raise ValueError(f"Unreviewed historical dependency difference: {relative}")
        equivalence[relative] = {
            "current_sha256": sha(current),
            "historical_sha256": sha(original),
            "equal_after_CRLF_normalization": equal,
            "runtime": "private exact historical source" if relative == MODEL else "current copy",
        }
    source = "operational/joint_measurement/fusion_cost_native.py"
    sources = {source: sha(ROOT / source), **read(fusion_cost.OUT / "REGISTRATION.json")["sources"]}
    committed_sources(sources)
    record = {
        "schema": "v18_fusion_cost_native_sources_v1",
        "original_failed_registration_sha256": sha(fusion_cost.OUT / "REGISTRATION.json"),
        "historical_root": str(historical),
        "migration_sha256": sha(owner() / migration.MANIFEST),
        "source_freeze_sha256": sha(owner() / "SOURCE_FREEZE.json"),
        "source_equivalence": equivalence,
        "private_sources": {name: sha(historical / name) for name in (MODEL, TRANSPORT)},
        "sources": sources,
        "admission": "Original migration.validate runs against its actual historical root",
        "scope": "Historical comparators and shared RGB producers use their original definitions",
        "new_candidate": "Unchanged frozen C1 and normalized fusion heads",
        "optimizer_updates": 0,
        "original_files_modified": False,
        "prediction_tolerance_changed": False,
    }
    freeze(OUT / "NATIVE_SOURCE_ADMISSION.json", record)
    return record


@contextmanager
def native_context(record: dict) -> Iterator[None]:
    """Scope the original loader validation and constructor; restore them on exit."""
    historical = Path(record["historical_root"])
    for relative, expected in record["private_sources"].items():
        if sha(historical / relative) != expected:
            raise ValueError("Admitted historical source changed")
    transport = load_private("v18_cost_native_transport", historical / TRANSPORT)
    model = load_private("v18_cost_native_model", historical / MODEL)
    for name in (
        "TRANSPORT_FEATURE_NAMES",
        "local_correlation_match",
        "transport_physical_features",
    ):
        setattr(model, name, getattr(transport, name))
    with (
        patch.object(migration, "ROOT", historical),
        patch.object(train_producers, "CausalScaleTTC", model.CausalScaleTTC),
        patch.object(train_producers, "CausalScaleTTCConfig", model.CausalScaleTTCConfig),
    ):
        yield


def qa(record: dict) -> dict:
    """Zero-update CPU parity on four native endpoints; synthetic pixels, no TTC targets."""
    path = OUT / "CPU_QA.json"
    if path.exists():
        result = read(path)
        if result["admission_sha256"] != sha(OUT / "NATIVE_SOURCE_ADMISSION.json"):
            raise ValueError("Native cost QA admission changed")
        if result["status"] != "PASS":
            raise ValueError("Preserved native cost QA failure")
        return result
    torch.set_num_threads(1)
    original_root = migration.ROOT
    checks = {}
    with native_context(record), torch.inference_mode():
        for fit in ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"):
            native = _producer(
                owner(),
                ROOT / "configs/rgb_port/producers.json",
                owner() / "P_MANIFEST.json",
                fit,
                "cpu",
            )
            values = dict(vars(native.config))
            current = CausalScaleTTC(CausalScaleTTCConfig(**values)).eval().requires_grad_(False)
            current.load_state_dict(native.state_dict(), strict=True)
            channels = values["in_channels"]
            generator = torch.Generator().manual_seed(181016)
            pixels = torch.rand((1, 3, channels, 128, 128), generator=generator)
            delta = torch.tensor([[0.04, 0.06]])
            a = train_producers.producer_features(native, pixels, delta)
            b = train_producers.producer_features(current, pixels, delta)
            checks[fit] = {key: torch.equal(a[key], b[key]) for key in a}
    restored = (
        migration.ROOT == original_root
        and train_producers.CausalScaleTTC is CausalScaleTTC
        and train_producers.CausalScaleTTCConfig is CausalScaleTTCConfig
    )
    result = {
        "status": "PASS" if restored and all(all(v.values()) for v in checks.values()) else "FAIL",
        "admission_sha256": sha(OUT / "NATIVE_SOURCE_ADMISSION.json"),
        "checks": checks,
        "context_restored": restored,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "scope": (
            "CPU native producer outputs on synthetic pixels; raw-route parity remains required"
        ),
    }
    freeze(path, result)
    if result["status"] != "PASS":
        raise ValueError("Native cost CPU parity failed")
    return result


def run(*, qa_only: bool = False) -> dict:
    """Preserve the failed attempt; measure into a new admitted directory with unchanged gates."""
    record = admit()
    proof = qa(record)
    if qa_only:
        return proof
    with native_context(record), patch.object(fusion_cost, "OUT", OUT):
        result = fusion_cost.measure()
    receipt = {
        "native_admission_sha256": sha(OUT / "NATIVE_SOURCE_ADMISSION.json"),
        "qa_sha256": sha(OUT / "CPU_QA.json"),
        "cost_result": result,
        "cost_registration_sha256": sha(OUT / "REGISTRATION.json"),
        "optimizer_updates": 0,
    }
    if result["status"] == "COMPLETE":
        freeze(OUT / "RUNTIME_RESULT.json", receipt)
    else:
        write(OUT / "PARTIAL.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    print(run(qa_only=args.qa_only))
