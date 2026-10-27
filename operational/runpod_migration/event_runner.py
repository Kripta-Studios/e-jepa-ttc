"""Fail-closed event-Pod preparation shared by producers and downstream stages."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from scripts import runpod_resources_v3

from .event_resources import EventResourceMonitor
from .immutable_seal import verify as verify_seal
from .inventory import digest
from .paths import PathMap

RUNTIME_FILES = (
    "event_handoff.py",
    "event_resources.py",
    "event_restart_probe.py",
    "event_runner.py",
    "event_volume.py",
)


class _SupplementalMap:
    def __init__(self, primary: PathMap, roots: tuple[Path, ...]) -> None:
        self.primary, self.roots = primary, roots

    def translate(self, value: str | os.PathLike[str]) -> str:
        try:
            relative = self.primary.relative(value)
        except ValueError:
            return self.primary.translate(value)
        primary = Path(self.primary.translate(value))
        if not primary.exists():
            for root in self.roots:
                supplemental = root / relative
                if supplemental.is_file():
                    return str(supplemental)
        return str(primary)


def _verified_supplement(
    root_value: str,
    admission_value: str,
    *,
    root_expected: Path,
    admission_schema: str,
    seal_schema: str,
    files: int,
) -> Path:
    root = Path(root_value).resolve(strict=True)
    admission_path = Path(admission_value).resolve(strict=True)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    seal_path = root.parent / f"{root.name}.SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not (
        root == root_expected
        and admission.get("schema") == admission_schema
        and admission.get("status") == "VERIFIED"
        and admission.get("root") == str(root)
        and admission.get("immutable_seal_sha256") == digest(seal_path)
        and admission.get("manifest_sha256") == digest(root / "SUPPLEMENT_MANIFEST.json")
        and seal.get("schema") == seal_schema
        and seal.get("status") == "SEALED"
        and seal.get("root") == str(root)
        and seal.get("manifest_sha256") == admission.get("manifest_sha256")
        and len(seal.get("files", [])) == files
    ):
        raise ValueError("Supplemental event mapping admission differs")
    for item in seal["files"]:
        path = root / str(item["relative"])
        observed = path.stat()
        if (
            path.is_symlink()
            or not path.is_file()
            or observed.st_size != int(item["bytes"])
            or observed.st_mtime_ns != int(item["mtime_ns"])
            or observed.st_ino != int(item["inode"])
            or observed.st_dev != int(item["device"])
            or observed.st_mode != int(item["mode"])
            or observed.st_mode & 0o222
        ):
            raise ValueError(f"Supplemental event member changed: {path}")
    return root


def _mapping(manifest: dict[str, Any], bundle: Path) -> PathMap | _SupplementalMap:
    primary = PathMap(manifest["roots"], str(bundle))
    raw_root = os.environ.get("EJEPA_EVENT_SUPPLEMENT_ROOT")
    raw_admission = os.environ.get("EJEPA_EVENT_SUPPLEMENT_ADMISSION")
    if raw_root is None and raw_admission is None:
        return primary
    if raw_root is None or raw_admission is None:
        raise ValueError("Supplement root and admission must be provided together")
    roots = [
        _verified_supplement(
            raw_root,
            raw_admission,
            root_expected=Path("/opt/ejepa_event_supplement"),
            admission_schema="ejepa_event_downstream_supplemental_admission_v1",
            seal_schema="ejepa_event_supplement_immutable_seal_v1",
            files=13,
        )
    ]
    authority_root = os.environ.get("EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT")
    authority_admission = os.environ.get("EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION")
    if authority_root is None or authority_admission is None:
        raise ValueError("Authority supplement root and admission are required")
    roots.insert(
        0,
        _verified_supplement(
            authority_root,
            authority_admission,
            root_expected=Path("/opt/ejepa_event_authority_supplement_v2"),
            admission_schema="ejepa_event_authority_supplement_admission_v1",
            seal_schema="ejepa_event_authority_supplement_immutable_seal_v1",
            files=7,
        ),
    )
    return _SupplementalMap(primary, tuple(roots))


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Refusing to replace immutable event record: {path}")
        return
    atomic_write_json(path, value)


def _prime_efficient_context_without_unshipped_scanner(bundle: Path) -> None:
    """Install the base resource policy while leaving an absent optional scanner inactive."""
    name = "operational.efficient_context"
    if name in sys.modules:
        raise RuntimeError("Efficient-context package loaded before migration overlay")
    package_root = (
        bundle
        / "workspace/e-jepa-ttc-v18-joint-measurement/operational/efficient_context"
    )
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    package = ModuleType(name)
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    sys.modules[name] = package
    try:
        import_module(f"{name}.resource_policy").install()
    except BaseException:
        sys.modules.pop(name, None)
        raise


def prepare_event(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
    """Install verified local input routing; never create or rewrite fit bindings."""
    if platform.system() != "Linux":
        raise RuntimeError("Event execution preparation requires Linux")
    bundle, run = bundle.resolve(strict=True), run.resolve()
    if not bundle.is_relative_to(Path("/opt")):
        raise ValueError("Event runtime must use the immutable local /opt stage")
    if not run.is_relative_to(Path("/workspace")) or run.is_relative_to(bundle):
        raise ValueError("Durable event outputs must be separate under /workspace")
    seal_path = bundle.parent / f"{bundle.name}.SEAL.json"
    manifest = verify_seal(bundle, seal_path)
    if manifest.get("profile") != "event_producers":
        raise ValueError("Event producer bundle profile is required")
    mapping = _mapping(manifest, bundle)
    original = Path(mapping.translate(manifest["source_run"])) / "SCIENTIFIC_CONFIG.json"
    if digest(original) != manifest["scientific_config_sha256"]:
        raise ValueError("Scientific authority differs from the sealed event bundle")
    run.mkdir(parents=True, exist_ok=True)
    copied = run / "SCIENTIFIC_CONFIG.json"
    if copied.exists() and digest(copied) != digest(original):
        raise ValueError("Durable event run contains another scientific authority")
    if not copied.exists():
        shutil.copyfile(original, copied)
    migration = {
        "schema": "ejepa_event_linux_migration_binding_v1",
        "original_authority_sha256": digest(original),
        "bundle_manifest_sha256": digest(bundle / "BUNDLE_MANIFEST.json"),
        "immutable_seal_sha256": digest(seal_path),
        "bundle_root": str(bundle),
        "run_root": str(run),
        "roots": manifest["roots"],
        "pod_memory_budget_bytes": budget_bytes,
        "precision": "bf16",
        "seed": 7,
        "checkpoint_updates": 100,
        "historical_checkpoint_import": True,
        "origin_git_commit": manifest.get("origin_git_commit"),
        "scientific_files_rewritten": False,
        "network_files_rehashed_on_restart": False,
        "fit_execution_bindings_created": False,
        "runtime_sources": {
            **{name: digest(Path(__file__).with_name(name)) for name in RUNTIME_FILES},
            "scripts/runpod_resources_v3.py": digest(Path(runpod_resources_v3.__file__)),
            "event_runtime_bootstrap.py": digest(
                Path(__file__).resolve().parents[2] / "event_runtime_bootstrap.py"
            ),
            "scripts/runpod_event_scientific_v1.py": digest(
                Path(__file__).resolve().parents[2] / "scripts/runpod_event_scientific_v1.py"
            ),
        },
        "downstream_supplement_admission_sha256": digest(
            Path(os.environ["EJEPA_EVENT_SUPPLEMENT_ADMISSION"])
        ),
        "authority_supplement_admission_sha256": digest(
            Path(os.environ["EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION"])
        ),
        "optional_historical_quota_scanner": "INACTIVE_CONFIG_AND_SEAL_UNSHIPPED",
    }
    migration_path = run / "EVENT_MIGRATION_BINDING.json"
    from .runtime import install

    _prime_efficient_context_without_unshipped_scanner(bundle)
    install(mapping, budget_bytes=budget_bytes, origin_git_commit=manifest.get("origin_git_commit"))
    from operational.final_refit import shared_resources
    from operational.final_refit.scientific_authority import load_config

    shared_resources.SharedResourceMonitor = EventResourceMonitor
    config = load_config(run)
    import torch

    if platform.python_version() != "3.11.15" or str(torch.__version__) != "2.11.0+cu128":
        raise ValueError("Expected Python3.11.15 and torch2.11.0+cu128 exactly")
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise ValueError("CUDA12.8 PyTorch and a visible GPU are required")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose exactly one BF16-capable GPU")
    allowed, resources = EventResourceMonitor(budget_bytes=budget_bytes).guard(
        run, checkpoint_reservation=1024**3, completed_updates=0
    )
    if not allowed:
        raise RuntimeError(f"Event resource preflight denied: {resources}")
    _write_once(migration_path, migration)
    return {
        "manifest": manifest,
        "config": config,
        "migration": migration,
        "migration_binding_sha256": digest(migration_path),
        "resources": resources,
        "gpu": torch.cuda.get_device_name(),
    }


def preflight_event(
    setup: dict[str, Any], run: Path, rotation: int, fit: str
) -> tuple[object, object, dict[str, Any]]:
    """Decode one zero-update batch and return the fixed source and recipe."""
    if rotation not in (2, 3, 4) or fit not in ("E_A5", "E_C2F"):
        raise ValueError("Event task must be E_A5/E_C2F in rotation 2/3/4")
    from operational.final_refit import scientific_fold_producers as science
    from operational.final_refit.fold_sources import load_fold_event_source

    from .runner import batch_identity

    cfg, manifest = setup["config"], setup["manifest"]
    source = load_fold_event_source(Path(cfg["fold_contract"]), rotation, Path(cfg["event_root"]))
    recipe = science.scientific_recipe(source, fit, Path(cfg["config"]))
    observed = batch_identity(source)
    suffix = (
        "workspace/e-jepa-ttc-v18-joint-measurement/artifacts/"
        f"runpod_migration_20261026/EVENT_INPUT_REFERENCE_R{rotation}.json"
    )
    if sum(item["relative"] == suffix for item in manifest["files"]) != 1:
        source.close()
        raise ValueError("Exactly one event CPU reference is required")
    reference = json.loads(
        (Path(setup["migration"]["bundle_root"]) / suffix).read_text(encoding="utf-8")
    )
    if (
        reference.get("batch_identity") != observed
        or reference.get("authority_sha256") != manifest["scientific_config_sha256"]
    ):
        source.close()
        raise ValueError("Cloud event decoding differs from the local CPU reference")
    report = {
        "schema": "ejepa_event_cloud_preflight_v1",
        "status": "PASS",
        "rotation": rotation,
        "fit": fit,
        "endpoint_updates": int(recipe.updates),
        "population": int(source.population_size),
        "recipe": recipe.identity,
        "batch_identity": observed,
        "input_parity": "BIT_EXACT_CPU_REFERENCE",
        "migration_binding_sha256": setup["migration_binding_sha256"],
        "optimizer_updates": 0,
    }
    _write_once(run / f"PREFLIGHT_R{rotation}_{fit}.json", report)
    atomic_write_json(
        run / f"PREFLIGHT_RESOURCE_OBSERVATION_R{rotation}_{fit}.json",
        {
            "schema": "ejepa_event_preflight_resource_observation_v1",
            "rotation": rotation,
            "fit": fit,
            "optimizer_updates": 0,
            "resources": setup["resources"],
        },
    )
    return source, recipe, report
