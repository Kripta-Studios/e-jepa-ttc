"""Admit explicit cross-scope RGB byte reuse without relabelling physical evidence."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

V7_SHA256 = "ae66d985c5a2bfafac2050d5e75a32e85de9f207509fadc34d978301a43c916b"
LAYOUT_SCHEMA = "ejepa_rgb_compositional_media_layout_v8"
ALLOWED_ROOTS = {Path("/opt/ejepa_rgb_frames"), Path("/workspace/ejepa_rgb_frames")}
ALLOWED_REUSE = {(2, "refiner", 3, "generation"), (4, "generation", 3, "refiner")}
ALLOWED_DIRECT = {(2, "generation"), (4, "refiner")}


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bound_path(item: Mapping[str, Any], name: str) -> Path:
    path = Path(str(item[f"{name}_path"])).resolve(strict=True)
    if digest(path) != item.get(f"{name}_sha256"):
        raise ValueError(f"Compositional RGB {name} bytes differ")
    return path


def _declared_path(item: Mapping[str, Any], name: str) -> Path:
    path = Path(str(item[f"{name}_path"]))
    if not path.is_absolute():
        raise ValueError(f"Compositional RGB {name} path is not absolute")
    return path


def member_bytes(plan: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for tar in plan.get("tars", []):
        relative = str(tar["relative"]).replace("\\", "/")
        for member in tar.get("members", []):
            key = relative + "\0" + str(member["member"])
            if key in result:
                raise ValueError("Compositional RGB plan duplicates a TAR/member identity")
            result[key] = int(member["payload_bytes"])
    if not result or any(value <= 0 for value in result.values()):
        raise ValueError("Compositional RGB plan member map is empty or invalid")
    return result


def load_layout(path: Path, original_config: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = read(path)
    roots = {Path(str(item)) for item in value.get("allowed_physical_roots", [])}
    if not (
        value.get("schema") == LAYOUT_SCHEMA
        and value.get("status") == "PASS"
        and roots == ALLOWED_ROOTS
        and Path(str(value.get("original_config_path"))).resolve(strict=True)
        == original_config.resolve(strict=True)
        and value.get("original_config_sha256") == digest(original_config)
        and isinstance(value.get("scopes"), list)
    ):
        raise ValueError("Compositional RGB layout authority differs")
    scopes: dict[str, dict[str, Any]] = {}
    for raw in value["scopes"]:
        item = dict(raw)
        identity = (
            int(item.get("rotation")),
            str(item.get("role")),
            int(item.get("physical_rotation")),
            str(item.get("physical_role")),
        )
        key = f"R{identity[0]}_{identity[1]}"
        root = Path(str(item.get("physical_root")))
        mode = str(item.get("mode"))
        allowed_identity = (mode == "REUSE" and identity in ALLOWED_REUSE) or (
            mode == "DIRECT"
            and (identity[0], identity[1]) in ALLOWED_DIRECT
            and identity[:2] == identity[2:]
        )
        if not allowed_identity or root not in ALLOWED_ROOTS or key in scopes:
            raise ValueError("Compositional RGB scope/root is unauthorized or duplicated")
        required = ["target_plan", "source_plan", "source_manifest"]
        if mode == "REUSE":
            required.append("reuse_proof")
        for name in required:
            if name == "source_manifest" and not _declared_path(item, name).is_file():
                continue
            _bound_path(item, name)
        if mode == "DIRECT" and (
            item["target_plan_sha256"] != item["source_plan_sha256"]
            or root != Path("/workspace/ejepa_rgb_frames")
        ):
            raise ValueError("Direct RGB scopes must retain their own plan in workspace root")
        scopes[key] = item
    if not scopes or set(scopes) - {
        "R2_refiner",
        "R2_generation",
        "R4_refiner",
        "R4_generation",
    }:
        raise ValueError("Compositional RGB layout has no authorized R2/R4 scopes")
    by_root: dict[Path, set[str]] = {}
    for item in scopes.values():
        source_plan = read(_bound_path(item, "source_plan"))
        relatives = {
            str(tar["relative"]).replace("\\", "/") for tar in source_plan["tars"]
        }
        root = Path(str(item["physical_root"]))
        overlap = by_root.setdefault(root, set()) & relatives
        if overlap:
            raise ValueError(f"Compositional RGB physical paths collide: {sorted(overlap)[:3]}")
        by_root[root].update(relatives)
    return {
        **value,
        "scope_map": scopes,
        "layout_sha256": digest(path),
        "layout_authority_sha256": canonical(
            {key: item for key, item in value.items() if key != "scopes"}
        ),
    }


def validate_equivalence(item: Mapping[str, Any]) -> dict[str, Any]:
    proof = read(_bound_path(item, "reuse_proof"))
    target_plan = read(_bound_path(item, "target_plan"))
    source_plan = read(_bound_path(item, "source_plan"))
    _bound_path(item, "source_manifest")
    identity = (
        int(item["rotation"]),
        str(item["role"]),
        int(item["physical_rotation"]),
        str(item["physical_role"]),
    )
    target, physical = proof.get("target", {}), proof.get("physical", {})
    proof_payload = {key: value for key, value in proof.items() if key != "receipt_sha256"}
    if not (
        proof.get("schema") == "ejepa_rgb_media_scope_reuse_proof_v1"
        and proof.get("status") == "PASS"
        and proof.get("same_exact_member_names_and_payload_bytes") is True
        and proof.get("same_logical_tar_paths") is True
        and proof.get("physical_sidecar_roles_relabelled") is False
        and proof.get("optimizer_updates") == 0
        and proof.get("receipt_sha256") == canonical(proof_payload)
        and target.get("rotation") == identity[0]
        and target.get("role") == identity[1]
        and target.get("plan_sha256") == item["target_plan_sha256"]
        and physical.get("rotation") == identity[2]
        and physical.get("role") == identity[3]
        and physical.get("plan_sha256") == item["source_plan_sha256"]
        and physical.get("manifest_sha256") == item["source_manifest_sha256"]
        and member_bytes(target_plan) == member_bytes(source_plan)
        and len(member_bytes(target_plan)) == int(proof.get("member_count"))
    ):
        raise ValueError("Compositional RGB target/physical equivalence differs")
    return proof


@contextmanager
def install(v6: Any, v7: Any, layout_path: Path, original_config_path: Path) -> Iterator[None]:
    original_load = v6.load_config
    original_verify = v6.verify_scope
    original_runtime = v6.routed_runtime
    original_wait = v6.wait_scope
    original_barrier = v7._barrier_paths
    initial_layout = load_layout(layout_path, original_config_path)
    observed_scopes: dict[str, str] = {}

    def load_config(path: Path) -> dict[str, Any]:
        original = original_load(path)
        if Path(path).resolve(strict=True) != original_config_path.resolve(strict=True):
            raise ValueError("V8 received a different immutable V6 config")
        layout = load_layout(layout_path, original_config_path)
        if (
            layout["layout_authority_sha256"]
            != initial_layout["layout_authority_sha256"]
        ):
            raise ValueError("Compositional RGB layout header authority changed")
        current_scopes = {
            key: canonical(item) for key, item in layout["scope_map"].items()
        }
        if any(current_scopes.get(key) != value for key, value in observed_scopes.items()):
            raise ValueError("Existing compositional RGB scope changed or vanished")
        observed_scopes.update(current_scopes)
        combined = dict(original["scope_map"])
        for key, item in layout["scope_map"].items():
            combined[key] = {
                "rotation": item["rotation"],
                "role": item["role"],
                "plan_path": item["target_plan_path"],
                "plan_sha256": item["target_plan_sha256"],
                "manifest_path": item["source_manifest_path"],
                "manifest_sha256": item["source_manifest_sha256"],
                "v8_composition": item,
            }
        return {
            **original,
            "scope_map": combined,
            "authority_sha256": canonical(
                {
                    "original_authority_sha256": original["authority_sha256"],
                    "layout_authority_sha256": initial_layout[
                        "layout_authority_sha256"
                    ],
                }
            ),
        }

    def verify(config: dict[str, Any], rotation: int, role: str) -> dict[str, Any] | None:
        composition = config["scope_map"].get(f"R{rotation}_{role}", {}).get(
            "v8_composition"
        )
        if composition is None:
            return original_verify(config, rotation, role)
        if not _declared_path(composition, "source_manifest").is_file():
            return None
        proof = (
            validate_equivalence(composition)
            if composition["mode"] == "REUSE"
            else None
        )
        physical_rotation = int(composition["physical_rotation"])
        physical_role = str(composition["physical_role"])
        physical_scope = {
            "rotation": physical_rotation,
            "role": physical_role,
            "plan_path": composition["source_plan_path"],
            "plan_sha256": composition["source_plan_sha256"],
            "manifest_path": composition["source_manifest_path"],
            "manifest_sha256": composition["source_manifest_sha256"],
        }
        physical_config = {
            **config,
            "scope_map": {f"R{physical_rotation}_{physical_role}": physical_scope},
        }
        old_root = v6.MEDIA_ROOT
        v6.MEDIA_ROOT = Path(str(composition["physical_root"]))
        try:
            evidence = original_verify(
                physical_config, physical_rotation, physical_role
            )
        finally:
            v6.MEDIA_ROOT = old_root
        if evidence is None:
            return None
        return {
            "schema": "ejepa_rgb_compositional_scope_admission_v8",
            "status": "PASS",
            "rotation": rotation,
            "role": role,
            "target_plan_sha256": composition["target_plan_sha256"],
            "physical_rotation": physical_rotation,
            "physical_role": physical_role,
            "physical_root": composition["physical_root"],
            "physical_admission": evidence,
            "composition_mode": composition["mode"],
            "reuse_proof_sha256": (
                composition["reuse_proof_sha256"] if proof is not None else None
            ),
            "reuse_proof_receipt_sha256": (
                proof["receipt_sha256"] if proof is not None else None
            ),
            "authority_sha256": config["authority_sha256"],
            "physical_sidecar_roles_relabelled": False,
            "optimizer_updates_inferred": False,
        }

    def barrier(config: dict[str, Any], rotation: int, role: str) -> tuple[Path, ...]:
        composition = config["scope_map"].get(f"R{rotation}_{role}", {}).get(
            "v8_composition"
        )
        if composition is None:
            return original_barrier(config, rotation, role)
        names = ["target_plan", "source_plan", "source_manifest"]
        if composition["mode"] == "REUSE":
            names.insert(0, "reuse_proof")
        paths = [
            (
                _declared_path(composition, name)
                if name == "source_manifest"
                else _bound_path(composition, name)
            )
            for name in names
        ]
        root = Path(str(composition["physical_root"]))
        for tar in read(_bound_path(composition, "source_plan"))["tars"]:
            target = root / str(tar["relative"])
            paths.extend((target, target.with_suffix(target.suffix + ".json")))
        return tuple(paths)

    def wait_scope(
        run: Path,
        config_path: Path,
        rotation: int,
        role: str,
        *,
        sleeper: Any = time.sleep,
    ) -> dict[str, Any]:
        key = f"R{rotation}_{role}"
        initial_authority: str | None = None
        initial_scope: str | None = None
        while True:
            config = load_config(config_path)
            scope = config["scope_map"].get(key)
            authority = str(config["authority_sha256"])
            scope_sha = canonical(scope) if scope is not None else None
            if initial_authority is None:
                initial_authority, initial_scope = authority, scope_sha
            elif authority != initial_authority or (
                initial_scope is not None and scope_sha != initial_scope
            ):
                raise ValueError("Current compositional RGB scope authority changed")
            evidence = v6.verify_scope(config, rotation, role)
            if evidence is not None:
                target = run / f"RGB_MEDIA_SCOPE_ADMISSION_R{rotation}_{role}.json"
                if target.exists() and read(target) != evidence:
                    raise ValueError("Existing compositional RGB admission differs")
                if not target.exists():
                    temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
                    with temporary.open("w", encoding="utf-8") as stream:
                        json.dump(evidence, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                return evidence
            stops = (
                run / "STOP_REQUEST",
                run / f"rotations/{rotation}/STOP_REQUEST",
                run / f"rotations/{rotation}/fits/R_CTX/STOP_REQUEST",
            )
            if any(path.exists() for path in stops):
                raise InterruptedError("STOP_REQUEST while waiting for compositional RGB media")
            deadline = datetime.fromisoformat(str(config["deadline_utc"]))
            if datetime.now(UTC) >= deadline:
                raise TimeoutError("Authorized RGB deadline reached while waiting for media")
            target = run / f"RGB_MEDIA_WAIT_R{rotation}_{role}.json"
            temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema": "ejepa_rgb_compositional_media_wait_v8",
                        "status": "WAITING_MEDIA_SCOPE",
                        "rotation": rotation,
                        "role": role,
                        "pid": os.getpid(),
                        "model_inference_started": False,
                        "optimizer_updates_inferred": False,
                        "updated_utc": datetime.now(UTC).isoformat(),
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            sleeper(float(config.get("poll_seconds", 15)))

    @contextmanager
    def routed(run: Path, rotation: int, config_path: Path) -> Iterator[None]:
        from operational.final_refit import materialize
        from operational.final_refit import scientific_oof_pipeline as pipeline

        reader, rgb_raw = materialize.TarFrameReader, pipeline.rgb_raw
        selected = [v6.MEDIA_ROOT]

        def routed_reader(root: Path, *args: Any, **kwargs: Any) -> Any:
            requested = Path(root)
            return reader(
                selected[0] if requested == v6.HISTORICAL_ROOT else requested,
                *args,
                **kwargs,
            )

        def gated(*args: Any, **kwargs: Any) -> Any:
            output = kwargs.get("output") if "output" in kwargs else args[3]
            role = Path(output).parent.name
            evidence = v6.wait_scope(run, config_path, rotation, role)
            config = load_config(config_path)
            composition = config["scope_map"].get(
                f"R{rotation}_{role}", {}
            ).get("v8_composition")
            selected[0] = (
                Path(str(evidence.get("physical_root", composition["physical_root"])))
                if composition is not None
                else v6.MEDIA_ROOT
            )
            try:
                return rgb_raw(*args, **kwargs)
            finally:
                selected[0] = v6.MEDIA_ROOT

        materialize.TarFrameReader, pipeline.rgb_raw = routed_reader, gated
        try:
            yield
        finally:
            materialize.TarFrameReader, pipeline.rgb_raw = reader, rgb_raw

    v6.load_config, v6.verify_scope, v6.routed_runtime, v6.wait_scope = (
        load_config,
        verify,
        routed,
        wait_scope,
    )
    v7._barrier_paths = barrier
    try:
        yield
    finally:
        v7._barrier_paths = original_barrier
        v6.load_config, v6.verify_scope, v6.routed_runtime, v6.wait_scope = (
            original_load,
            original_verify,
            original_runtime,
            original_wait,
        )


def _consume(argv: list[str], name: str) -> tuple[str, list[str]]:
    if argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = argv.index(name)
    return argv[index + 1], [*argv[:index], *argv[index + 2 :]]


def main() -> None:
    source, ops = Path(__file__).resolve(), Path(__file__).resolve().parent
    v7_path = ops / "runpod_rgb_context_v7.py"
    if digest(v7_path) != V7_SHA256:
        raise ValueError("Frozen RGB context V7 bytes differ")
    raw_layout, forwarded = _consume(sys.argv, "--media-layout")
    raw_config = forwarded[forwarded.index("--media-config") + 1]
    sys.path.insert(0, str(ops))
    import runpod_rgb_context_v6 as v6
    import runpod_rgb_context_v7 as v7

    run = Path(forwarded[forwarded.index("--run") + 1]).resolve(strict=True)
    layout = load_layout(Path(raw_layout), Path(raw_config))
    binding = {
        "schema": "ejepa_rgb_compositional_context_v8",
        "status": "PASS",
        "source_sha256": digest(source),
        "context_v7_sha256": digest(v7_path),
        "layout_sha256": layout["layout_sha256"],
        "layout_authority_sha256": layout["layout_authority_sha256"],
        "original_config_sha256": digest(Path(raw_config)),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "allowed_physical_roots": sorted(str(root) for root in ALLOWED_ROOTS),
        "physical_sidecar_roles_relabelled": False,
        "existing_r3_admissions_changed": False,
        "optimizer_updates_inferred": False,
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V8.json"
    if target.exists() and read(target) != binding:
        raise ValueError("RGB compositional recovery V8 binding differs")
    if not target.exists():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(binding, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    original_argv, sys.argv = sys.argv, forwarded
    try:
        with install(v6, v7, Path(raw_layout), Path(raw_config)):
            v7.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
