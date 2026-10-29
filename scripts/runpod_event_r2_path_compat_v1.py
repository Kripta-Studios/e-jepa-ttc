"""Run R2/E_A5 with an explicit proof for two Windows-path-derived identity hashes."""

from __future__ import annotations

import copy
import json
import os
import runpy
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _argument(name: str) -> str:
    if sys.argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing {name} value")
    return sys.argv[index + 1]


def main() -> None:
    runtime = Path("/workspace/ejepa_event_operations/runtime_v12")
    base = Path("/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement")
    sys.path[:0] = [str(runtime), str(base), str(base / "src")]
    import operational.runpod_migration as migration

    extension = runtime / "operational/runpod_migration"
    migration.__path__.insert(0, str(extension))
    from operational.final_refit.recipe import canonical_sha256
    from operational.rgb_port.accounting import atomic_write_json
    from operational.runpod_migration import event_handoff as handoff
    from operational.runpod_migration.inventory import digest

    if _argument("--rotation") != "2" or _argument("--fit") != "E_A5":
        raise ValueError("Path compatibility is restricted to R2/E_A5")
    if sys.argv[1] not in ("probe", "producer"):
        raise ValueError("Path compatibility is restricted to probe/producer")
    run = Path(_argument("--run")).resolve(strict=True)
    migration_path = run / "EVENT_MIGRATION_BINDING.json"
    receipt_path = run / "R2_E_A5_WINDOWS_PATH_EQUIVALENCE.json"
    original_validate = handoff.validate_origin_identity
    original_create = handoff.create_binding

    def convert(value: Any, linux_prefix: str, windows_root: str) -> Any:
        if isinstance(value, str) and value.startswith(linux_prefix):
            return windows_root + value[len(linux_prefix) :].replace("/", "\\")
        if isinstance(value, dict):
            return {key: convert(item, linux_prefix, windows_root) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item, linux_prefix, windows_root) for item in value]
        return value

    def compatible_validate(
        identity: Mapping[str, Any], source: object, recipe: object, *, fit: str
    ) -> None:
        origin_source = dict(identity["source_identity"])
        current_source = dict(source.identity)  # type: ignore[attr-defined]
        origin_inventory = str(origin_source["inventory_sha256"])
        current_inventory = str(current_source["inventory_sha256"])
        roles = origin_source["storage_source_identity"]
        role_path = str(next(iter(roles.values()))["role_manifest_path"])
        marker = "\\e-jepa-ttc-"
        position = role_path.lower().find(marker)
        if position < 0:
            raise ValueError("Origin role manifest does not identify its Windows workspace")
        windows_root = role_path[: position + 1]
        payload = asdict(source.inventory)  # type: ignore[attr-defined]
        payload.pop("inventory_sha256")
        reconstructed = canonical_sha256(
            convert(payload, "/opt/ejepa_event_bundle/workspace/", windows_root)
        )
        origin_recipe = dict(identity["producer_recipe"])
        current_recipe = dict(recipe.identity)  # type: ignore[attr-defined]
        if not (
            reconstructed == origin_inventory
            and origin_recipe.pop("role_manifest_sha256") == origin_inventory
            and current_recipe.pop("role_manifest_sha256") == current_inventory
            and origin_recipe == current_recipe
        ):
            raise ValueError("R2/E_A5 identity differs beyond proven physical path hashes")
        candidate = copy.deepcopy(dict(identity))
        candidate["source_identity"]["inventory_sha256"] = current_inventory
        candidate["producer_recipe"]["role_manifest_sha256"] = current_inventory
        candidate["recipe_identity_sha256"] = recipe.identity_sha256  # type: ignore[attr-defined]
        candidate["identity_sha256"] = canonical_sha256(
            {key: value for key, value in candidate.items() if key != "identity_sha256"}
        )
        original_validate(candidate, source, recipe, fit=fit)
        receipt = {
            "schema": "ejepa_event_windows_path_identity_equivalence_v1",
            "status": "VERIFIED",
            "rotation": 2,
            "fit": "E_A5",
            "origin_identity_sha256": identity["identity_sha256"],
            "origin_inventory_sha256": origin_inventory,
            "linux_inventory_sha256": current_inventory,
            "windows_reconstructed_inventory_sha256": reconstructed,
            "normalized_fields": [
                "source_identity.inventory_sha256",
                "producer_recipe.role_manifest_sha256",
                "recipe_identity_sha256",
            ],
            "all_other_source_and_recipe_fields_exact": True,
            "identity_bytes_rewritten": False,
            "migration_binding_sha256": digest(migration_path),
            "compatibility_source_sha256": digest(Path(__file__)),
        }
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if receipt_path.exists():
            if receipt_path.read_text(encoding="utf-8") != encoded:
                raise ValueError("Existing R2 path-equivalence receipt differs")
        else:
            atomic_write_json(receipt_path, receipt)

    def compatible_create(
        directory: Path,
        source: object,
        recipe: object,
        **kwargs: Any,
    ) -> dict[str, Any]:
        value = original_create(directory, source, recipe, **kwargs)
        expected = digest(receipt_path.resolve(strict=True))
        path = directory.resolve(strict=True) / "EVENT_EXECUTION_BINDING.json"
        observed = value.get("path_equivalence_admission_sha256")
        if observed is None:
            value = dict(value)
            value["path_equivalence_admission_sha256"] = expected
            value.pop("payload_sha256", None)
            value["payload_sha256"] = canonical_sha256(value)
            atomic_write_json(path, value)
        elif observed != expected:
            raise ValueError("Existing R2 execution binding has another path admission")
        return value

    handoff.validate_origin_identity = compatible_validate
    handoff.create_binding = compatible_create

    @contextmanager
    def compatible_activated(
        science: Any,
        directory: Path,
        source: object,
        recipe: object,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        import torch

        value = compatible_create(directory, source, recipe, **kwargs)
        binding_path = directory / "EVENT_EXECUTION_BINDING.json"
        saved = torch.load(
            Path(str(value["origin_checkpoint"])), map_location="cpu", weights_only=False
        )
        origin_identity = dict(saved["identity"])
        original_identity = science.identity_for
        original_authority = science.execution_binding
        original_checkpoint = science.ProducerCheckpoint
        checkpoint_kwargs = handoff._binding_kwargs(binding_path, value)

        def identity(*args: Any, **named: Any) -> dict[str, Any]:
            current = original_identity(*args, **named)
            compatible_validate(origin_identity, source, recipe, fit=kwargs["fit"])
            for field in (
                "fit_id",
                "rotation",
                "fold_contract_sha256",
                "rotation_manifest_sha256",
                "training_cache_sha256",
                "training_sample_token_sha256",
                "train_group_ids",
                "generation_group_ids",
                "endpoint_updates",
                "checkpoint_interval_updates",
            ):
                if current.get(field) != origin_identity.get(field):
                    raise ValueError(f"Cloud scientific identity changed field {field}")
            return dict(origin_identity)

        class BoundCheckpoint(original_checkpoint):
            def __init__(self, *args: object, **named: object) -> None:
                injected = dict(named)
                injected.update(checkpoint_kwargs)
                super().__init__(*args, **injected)

        def authority(_run: Path) -> dict[str, Any]:
            return dict(origin_identity["execution_authority"])

        science.identity_for = identity
        science.execution_binding = authority
        science.ProducerCheckpoint = BoundCheckpoint
        try:
            yield value
        finally:
            science.identity_for = original_identity
            science.execution_binding = original_authority
            science.ProducerCheckpoint = original_checkpoint

    handoff.activated = compatible_activated
    os.environ["EJEPA_EVENT_R2_PATH_EQUIVALENCE"] = str(receipt_path)
    runpy.run_path(str(runtime / "scripts/runpod_event_scientific_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
