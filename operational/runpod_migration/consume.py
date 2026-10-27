"""Consume explicitly registered complete cloud endpoints through original verifiers."""

# ruff: noqa: ANN401 -- wrappers preserve heterogeneous frozen runtime call signatures.

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from .inventory import digest, read
from .paths import mapped_path_class
from .return_endpoint import canonical_role_equal, verify_cloud_snapshot
from .return_verify import ReturnPathMap


class CloudMappings:
    """Route only registered immutable cloud roots; reject all unknown cloud paths."""

    def __init__(self, roots: list[Path]) -> None:
        self.maps = [ReturnPathMap(read(root / "MIGRATION_BINDING.json"), root) for root in roots]

    def owner(self, value: str | os.PathLike[str]) -> ReturnPathMap | None:
        raw = os.fspath(value)
        if not raw.startswith("/workspace/ejepa_technical/"):
            return None
        for mapping in self.maps:
            if PurePosixPath(raw).is_relative_to(PurePosixPath(mapping.outputs)):
                return mapping
        raise ValueError(f"Unregistered cloud output: {raw}")

    def translate(self, value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value)
        owner = self.owner(raw)
        if owner is not None:
            return owner.translate(raw)
        if raw.startswith("/workspace/ejepa_bundle/"):
            if not self.maps:
                raise ValueError("No verified cloud input mapping")
            translated = {mapping.translate(raw) for mapping in self.maps}
            if len(translated) != 1:
                raise ValueError("Cloud input mappings disagree")
            return translated.pop()
        if raw.startswith("/workspace/"):
            raise ValueError(f"Unregistered cloud path: {raw}")
        return raw


@contextmanager
def verified_cloud_consumption(
    roots: list[Path], repository: Path, *, downstream_roots: list[Path] | None = None
) -> Iterator[dict[str, Any]]:
    """Retain original scientific checks and add cloud-specific runtime evidence."""
    import torch

    torch_runtime: Any = torch

    from operational.final_refit import (
        fold_training,
        oof_lineage,
        scientific_authority,
        scientific_fold_heads,
        training,
    )

    reports = [
        verify_cloud_snapshot(root, repository=repository, require_endpoint=True) for root in roots
    ]
    from .return_rgb_downstream import verify_downstream_authority

    downstream_roots = downstream_roots or []
    downstream_reports = [
        verify_downstream_authority(root, repository) for root in downstream_roots
    ]
    mapping = CloudMappings([*roots, *downstream_roots])
    path_type = mapped_path_class(mapping)
    modules = [fold_training, oof_lineage, scientific_authority, scientific_fold_heads, training]
    original_paths = [(module, module.Path) for module in modules]
    original_equal = oof_lineage._json_equivalent
    original_verify = scientific_fold_heads.verify_scientific_admission
    original_load = torch_runtime.load
    original_ancestors = fold_training._validate_ancestors
    original_head_ancestors = scientific_fold_heads._validate_ancestors
    runtime_modules = []
    for name in ("operational.scientific_oof_isolated", "operational.scientific_oof_transport"):
        import importlib

        module = importlib.import_module(name)
        runtime_modules.append((module, module.verify_runtime_endpoint))

    def load(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            file = mapping.translate(file)
        return original_load(file, *args, **kwargs)

    def verify(path: Path) -> dict[str, Any]:
        return original_verify(path_type(path))

    def role_equal(left: object, right: object) -> bool:
        return canonical_role_equal(left, right, mapping)

    def runtime_verify(path: Path, admission: dict[str, Any], original: Any) -> None:
        owner = mapping.owner(admission["checkpoint_path"])
        if owner is None:
            original(path, admission)
            return
        root = owner.downloaded
        if (root / "RGB_DOWNSTREAM_ADMISSION.json").is_file():
            task = read(root / "RGB_DOWNSTREAM_ADMISSION.json")["task"]
            if admission["fit_id"] not in {"PAIR_R", "R_CTX"}:
                raise ValueError("RGB downstream return cannot authorize another fit")
            task = {"rotation": task["rotation"], "fit": admission["fit_id"]}
        else:
            task = read(root / "SCIENTIFIC_CLOUD_ADMISSION.json")["task"]
        frozen = root / f"rotations/{task['rotation']}/fits/{task['fit']}/ENDPOINT_ADMISSION.json"
        if digest(path) != digest(frozen):
            raise ValueError(
                "Local cloud endpoint metadata differs from its verified immutable return"
            )
        if admission["rotation"] != task["rotation"] or admission["fit_id"] != task["fit"]:
            raise ValueError("Cloud endpoint task differs from its registered return")

    def ancestors(unit: Any, stage: Any, receipts: Any, admissions: Any) -> dict[str, str]:
        if set(receipts) != set(stage.depends_on) or set(admissions) != set(stage.depends_on):
            raise ValueError("Fold parent receipts/admissions differ from dependencies")
        cloud_names = {
            name
            for name, path in admissions.items()
            if mapping.owner(read(path)["checkpoint_path"]) is not None
        }
        if not cloud_names:
            return original_ancestors(unit, stage, receipts, admissions)
        local_names = set(stage.depends_on) - cloud_names
        result = original_ancestors(
            unit,
            replace(
                stage, depends_on=tuple(name for name in stage.depends_on if name in local_names)
            ),
            {name: path for name, path in receipts.items() if name in local_names},
            {name: path for name, path in admissions.items() if name in local_names},
        )
        for name in cloud_names:
            # The original verifier first checks original JSON payload hashes,
            # checkpoint bytes, exact row identities and transitive lineage.
            # Only the location of an explicitly verified byte-identical copy
            # differs from the historical validator's string path equality.
            admission = verify(admissions[name])
            runtime_verify(admissions[name], admission, None)
            producer, generation = unit.roles["producer"], unit.roles["generation"]
            if not (
                admission["rotation"] == unit.rotation
                and admission["fit_id"] == name
                and name in {"R_A5", "R_C2F", "PAIR_R"}
                and admission["fold_contract_sha256"] == unit.fold_contract_sha256
                and admission["rotation_manifest_sha256"] == unit.rotation_manifest_sha256
                and admission["train_rows_sha256"] == producer["sha256"]
                and admission["ordered_sample_token_sha256"] == producer["sample_token_sha256"]
                and admission["sample_token_set_sha256"] == producer["sample_token_set_sha256"]
                and admission["ordered_native_ordinal_sha256"]
                == producer["ordered_native_ordinal_sha256"]
                and admission["train_group_ids"] == producer["group_ids"]
                and admission["generation_group_ids"] == generation["group_ids"]
                and admission["generation_token_set_sha256"]
                == generation["sample_token_set_sha256"]
                and admission["checkpoint_receipt_sha256"] == digest(receipts[name])
                and digest(Path(mapping.translate(admission["checkpoint_receipt_path"])))
                == digest(receipts[name])
            ):
                raise ValueError("Cloud parent differs from the consuming fold's exact role")
            result[name] = admission["checkpoint_sha256"]
        return result

    try:
        for module, _original in original_paths:
            module.__dict__["Path"] = path_type
        oof_lineage._json_equivalent = role_equal
        scientific_fold_heads.verify_scientific_admission = verify
        torch_runtime.load = load
        fold_training._validate_ancestors = ancestors
        scientific_fold_heads._validate_ancestors = ancestors
        for module, original in runtime_modules:
            module.verify_runtime_endpoint = lambda path, admission, original=original: (
                runtime_verify(path, admission, original)
            )
        for root, report in zip(roots, reports, strict=True):
            path = (
                root
                / f"rotations/{report['rotation']}/fits/{report['fit']}/ENDPOINT_ADMISSION.json"
            )
            admitted = verify(path)
            report["original_scientific_endpoint_verifier"] = "PASS"
            report["endpoint_admission_sha256"] = digest(path)
            report["admitted_updates"] = admitted["updates"]
        for root, report in zip(downstream_roots, downstream_reports, strict=True):
            verified = {}
            for fit in report["fits"]:
                path = root / f"rotations/{report['rotation']}/fits/{fit}/ENDPOINT_ADMISSION.json"
                admitted = verify(path)
                runtime_verify(path, admitted, None)
                verified[fit] = {
                    "admission_sha256": digest(path),
                    "checkpoint_sha256": admitted["checkpoint_sha256"],
                    "admitted_updates": admitted["updates"],
                }
            report["original_scientific_endpoint_verifier"] = "PASS"
            report["endpoints"] = verified
            rotation_root = root / f"rotations/{report['rotation']}"
            cache = read(rotation_root / "caches/R_CTX.json")
            normalizer_path = rotation_root / "normalizers/R_CTX.json"
            normalizer = read(normalizer_path)
            context = read(rotation_root / "fits/R_CTX/ENDPOINT_ADMISSION.json")
            fragment = read(rotation_root / "generation/R_CTX/fragment.json")
            parents = context["parent_checkpoint_sha256"]
            if not (
                cache["normalizers"] == {"R_CTX": digest(normalizer_path)}
                and cache["parents"] == normalizer["parents"] == fragment["parents"] == parents
                and cache["output_sha256"] == digest(rotation_root / "caches/R_CTX.npz")
                and normalizer["raw_sha256"]
                == digest(rotation_root / "refiner/R_CTX/raw_phase17.npz")
                and normalizer["role"] == "refiner"
                and normalizer["targets_read"] is False
                and normalizer["groups"] == context["train_group_ids"]
                and fragment["targets_used"] is False
                and fragment["role"] == "generation"
                and fragment["interface"] == "PHASE17_RAW_RGB"
                and fragment["output_sha256"]
                == digest(rotation_root / "generation/R_CTX/fragment.npz")
            ):
                raise ValueError("Returned RGB normalizer, role cache or OOF fragment differs")
            report["normalizer_and_fragment_ancestry"] = "PASS"
        yield {
            "status": "PASS",
            "endpoints": reports,
            "rgb_downstream": downstream_reports,
            "scientific_files_rewritten": False,
            "cloud_runtime_not_misrepresented_as_historical_prefetch": True,
        }
    finally:
        for module, original in original_paths:
            module.__dict__["Path"] = original
        oof_lineage._json_equivalent = original_equal
        scientific_fold_heads.verify_scientific_admission = original_verify
        torch_runtime.load = original_load
        fold_training._validate_ancestors = original_ancestors
        scientific_fold_heads._validate_ancestors = original_head_ancestors
        for module, original in runtime_modules:
            module.verify_runtime_endpoint = original


def main() -> int:
    """Use the explicit return registry while preserving the existing metadata repair."""
    repository = Path(__file__).resolve().parents[2]
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    from operational.final_refit.scientific_oof_queue import validate_plan

    plan = read(run / "SCIENTIFIC_OOF_QUEUE_PLAN.json")
    validate_plan(plan)
    for path in (
        Path(__file__),
        Path(__file__).with_name("return_endpoint.py"),
        Path(__file__).with_name("return_rgb_downstream.py"),
    ):
        if plan["source_bindings"].get(str(path.resolve())) != digest(path):
            raise ValueError("Cloud consumer is not bound by the active queue plan")
    registry_path = run / "CLOUD_RETURN_REGISTRY.json"
    registry = read(registry_path)
    if registry.get("schema") != "ejepa_cloud_return_registry_v1" or registry.get(
        "scientific_config_sha256"
    ) != digest(run / "SCIENTIFIC_CONFIG.json"):
        raise ValueError("Cloud return registry differs from this scientific authority")
    roots = []
    for entry in registry.get("endpoints", []):
        root = Path(entry["snapshot_root"]).resolve(strict=True)
        if digest(root / "MANIFEST.json") != entry["snapshot_manifest_sha256"]:
            raise ValueError("Registered cloud endpoint snapshot changed")
        roots.append(root)
    downstream_roots = []
    for entry in registry.get("rgb_downstream", []):
        root = Path(entry["snapshot_root"]).resolve(strict=True)
        if digest(root.parents[1] / "MANIFEST.json") != entry["archive_manifest_sha256"]:
            raise ValueError("Registered RGB downstream archive changed")
        downstream_roots.append(root)
    from operational.scientific_oof_rgb_fixed import main as original_main

    with verified_cloud_consumption(roots, repository, downstream_roots=downstream_roots):
        return original_main()


if __name__ == "__main__":
    raise SystemExit(main())
