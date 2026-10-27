"""Apply the audited RGB observation metadata comparison only during queue calls."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

V3_SHA256 = "b42cd136eb08541d4d5676403258bc29684d03fb419e25fd0836625d67bead17"
REPAIR_SHA256 = "0e4167cf46c483ba777c076e6039a334c999085eb87207505bc45aa26b466b49"
AUDIT_SHA256 = "572a1a6c17bfa6be26cd403c23ca0f652ee1dfc5cf08480988de792224b95e07"
ORIGINAL_MATERIALIZE_SHA256 = (
    "aaf47f9f3c211b5e46ebdb67a4a980347c6e59524ff2b84ee4a85a0d1ce268c5"
)
QUERY_FIELDS = ["query_age_us", "query_time_us", "sample_token"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        import hashlib

        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_repair(path: Path) -> ModuleType:
    """Load the external hash-bound repair only after the RGB runtime is installed."""
    path = path.resolve(strict=True)
    if digest(path) != REPAIR_SHA256:
        raise ValueError("RGB metadata repair bytes differ")
    name = "ejepa_rgb_context_v5_metadata_fix"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def patched_materializer_context(repair: Any) -> Iterator[None]:
    """Patch exactly the materializer and its imported alias for one queue call."""
    from operational.final_refit import materialize, scientific_oof_features

    original = materialize.materialize_rgb_observations
    alias = scientific_oof_features.materialize_rgb_observations
    if alias is not original:
        raise ValueError("RGB materializer alias differs before metadata-only patch")
    replacement = repair.patched_materializer()
    materialize.materialize_rgb_observations = replacement
    scientific_oof_features.materialize_rgb_observations = replacement
    try:
        yield
    finally:
        materialize.materialize_rgb_observations = original
        scientific_oof_features.materialize_rgb_observations = alias


def install_queue_heartbeat(
    module: Any,
    clock: Any = time.monotonic,
    *,
    repair_path: Path | None = None,
    heartbeat_installer: Any | None = None,
) -> None:
    """Preserve V3 heartbeat semantics and scope the metadata patch to each call."""
    if heartbeat_installer is None:
        try:
            import runpod_rgb_recovery_v3 as v3
        except ModuleNotFoundError:  # pragma: no cover - repository import path
            from scripts import runpod_rgb_recovery_v3 as v3

        heartbeat_installer = v3.install_queue_heartbeat
    heartbeat_installer(module, clock=clock)
    original = module.run_rgb_downstream
    source = repair_path or Path(__file__).resolve().parent / "oof_rgb_metadata_fix.py"

    def run(*args: Any, **kwargs: Any) -> Any:
        repair = load_repair(source)
        with patched_materializer_context(repair):
            return original(*args, **kwargs)

    module.run_rgb_downstream = run


@contextmanager
def composed_v3_hook(v3: Any, repair_path: Path) -> Iterator[None]:
    """Install V5 over one captured V3 hook, then restore the exact original."""
    original = v3.install_queue_heartbeat

    def install(module: Any, clock: Any = time.monotonic) -> None:
        install_queue_heartbeat(
            module,
            clock=clock,
            repair_path=repair_path,
            heartbeat_installer=original,
        )

    v3.install_queue_heartbeat = install
    try:
        yield
    finally:
        v3.install_queue_heartbeat = original


def validate_audit(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if digest(path) != AUDIT_SHA256:
        raise ValueError("RGB metadata stability audit bytes differ")
    value = json.loads(path.read_text(encoding="utf-8"))
    invariants = value.get("invariants")
    if not (
        value.get("schema") == "ejepa_rgb_observation_metadata_stability_audit_v1"
        and value.get("status") == "PASS"
        and value.get("global_changed_keys")
        == {field: 352_932 for field in QUERY_FIELDS}
        and isinstance(invariants, dict)
        and invariants.get("all_observed_duplicate_differences_are_exact_three_query_fields")
        is True
        and invariants.get("sensor_roi_group_role_conflicts") == 0
        and invariants.get("model_forward_calls") == 0
        and invariants.get("remote_updates") == 0
    ):
        raise ValueError("RGB metadata stability audit contract differs")
    return value


def write_sidecar(
    run: Path,
    source: Path,
    v3_path: Path,
    repair_path: Path,
    audit_path: Path,
) -> None:
    binding = {
        "schema": "ejepa_rgb_context_metadata_recovery_v5",
        "status": "PASS",
        "source_sha256": digest(source),
        "recovery_v3_sha256": digest(v3_path),
        "metadata_repair_sha256": digest(repair_path),
        "metadata_stability_audit_sha256": digest(audit_path),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "original_materialize_sha256": ORIGINAL_MATERIALIZE_SHA256,
        "excluded_query_reference_fields": QUERY_FIELDS,
        "materializer_ast_comparisons_replaced": 1,
        "materializer_and_imported_alias_scoped_to_queue_call": True,
        "rgb_pixels_changed": False,
        "scientific_recipes_changed": False,
        "scientific_inputs_changed": False,
        "scientific_sources_rewritten": False,
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V5.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != binding:
            raise ValueError("RGB context recovery V5 binding differs")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def main() -> None:
    source = Path(__file__).resolve()
    ops = source.parent
    v3_path = ops / "runpod_rgb_recovery_v3.py"
    repair_path = ops / "oof_rgb_metadata_fix.py"
    audit_path = ops / "RGB_OBSERVATION_METADATA_STABILITY_AUDIT.json"
    if digest(v3_path) != V3_SHA256:
        raise ValueError("Frozen RGB recovery V3 bytes differ")
    if digest(repair_path) != REPAIR_SHA256:
        raise ValueError("External RGB metadata repair bytes differ")
    validate_audit(audit_path)
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    write_sidecar(run, source, v3_path, repair_path, audit_path)
    sys.path.insert(0, str(ops))
    import runpod_rgb_recovery_v3 as v3

    with composed_v3_hook(v3, repair_path):
        v3.main()


if __name__ == "__main__":
    main()
