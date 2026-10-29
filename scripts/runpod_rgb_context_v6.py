"""Gate RGB scopes on verified sparse frames and route only materializer readers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

V5_SHA256 = "cb349a6d514245414f508800b878f9eb3cae44f167ce6c5ec8076696d299f357"
CONFIG_SCHEMA = "ejepa_rgb_media_scope_config_v1"
HISTORICAL_ROOT = Path("/workspace/ejepa_bundle/drives/E/eAP_dataset")
MEDIA_ROOT = Path("/opt/ejepa_rgb_frames")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _relative(value: object) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe RGB supplement path: {value}")
    return path.as_posix()


def load_config(path: Path) -> dict[str, Any]:
    """Load append-only scope evidence under one immutable six-scope authority."""
    path = path.resolve(strict=True)
    value = read(path)
    budget_path = Path(str(value.get("budget_path"))).resolve(strict=True)
    budget = read(budget_path)
    deadline = datetime.fromisoformat(str(value.get("deadline_utc")))
    authorized = [
        {"rotation": rotation, "role": role}
        for rotation in (2, 3, 4)
        for role in ("refiner", "generation")
    ]
    if not (
        value.get("schema") == CONFIG_SCHEMA
        and value.get("media_root") == str(MEDIA_ROOT)
        and value.get("historical_media_root") == str(HISTORICAL_ROOT)
        and value.get("authorized_scopes") == authorized
        and value.get("scope_evidence_append_only") is True
        and value.get("budget_sha256") == digest(budget_path)
        and budget.get("schema") == "ejepa_cloud_budget_authorization_v1"
        and budget.get("maximum_total_usd") == 24
        and budget.get("execution_deadline_utc") == value.get("deadline_utc")
        and deadline.tzinfo is not None
        and isinstance(value.get("scopes"), list)
    ):
        raise ValueError("RGB media scope authority is invalid")
    scope_map: dict[str, dict[str, Any]] = {}
    for raw in value["scopes"]:
        if not isinstance(raw, dict):
            raise ValueError("RGB scope evidence must be an object")
        item = dict(raw)
        rotation, role = item.get("rotation"), item.get("role")
        key = f"R{rotation}_{role}"
        if {"rotation": rotation, "role": role} not in authorized or key in scope_map:
            raise ValueError("RGB scope evidence identity is invalid or duplicated")
        for field in ("plan_sha256", "manifest_sha256"):
            if not isinstance(item.get(field), str) or len(item[field]) != 64:
                raise ValueError(f"RGB scope {field} is invalid: {key}")
        scope_map[key] = item
    header = {key: item for key, item in value.items() if key != "scopes"}
    return {
        **value,
        "scope_map": scope_map,
        "authority_sha256": canonical(header),
        "config_sha256": digest(path),
    }


def _decode(payload: bytes) -> tuple[list[int], str]:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return list(pixels.shape), hashlib.sha256(pixels.tobytes()).hexdigest()


def _member_covers(actual: Mapping[str, Any], planned: Mapping[str, Any]) -> bool:
    return bool(
        actual.get("member") == planned.get("member")
        and actual.get("payload_bytes") == planned.get("payload_bytes")
        and set(planned.get("rotations", [])) <= set(actual.get("rotations", []))
        and set(planned.get("roles", [])) <= set(actual.get("roles", []))
    )


def verify_scope(config: Mapping[str, Any], rotation: int, role: str) -> dict[str, Any] | None:
    """Return PASS evidence; only absent authorized evidence/files return None."""
    scope = config["scope_map"].get(f"R{rotation}_{role}")
    if scope is None:
        return None
    plan_path, manifest_path = Path(str(scope["plan_path"])), Path(str(scope["manifest_path"]))
    if not plan_path.is_file() or not manifest_path.is_file():
        return None
    if digest(plan_path) != scope["plan_sha256"] or digest(manifest_path) != scope[
        "manifest_sha256"
    ]:
        raise ValueError("RGB scope plan/manifest bytes differ")
    plan, manifest = read(plan_path), read(manifest_path)
    plan_payload = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "receipt_sha256"
    }
    if not (
        plan.get("schema") == "ejepa_rgb_frame_supplement_plan_v1"
        and plan.get("status") == "PLANNED_HEADERS_VERIFIED"
        and plan.get("rotations") == [rotation]
        and plan.get("roles") == [role]
        and plan.get("receipt_sha256") == canonical(plan_payload)
        and manifest.get("schema") == "ejepa_rgb_frame_supplement_v1"
        and manifest.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
        and manifest.get("plan_sha256") == scope["plan_sha256"]
        and manifest.get("tar_count") == plan.get("tar_count")
        and manifest.get("member_count") == plan.get("member_count")
        and manifest.get("payload_recompressed") is False
        and manifest.get("decoded_pixel_parity") is True
        and manifest.get("optimizer_updates") == 0
        and manifest.get("receipt_sha256") == canonical(manifest_payload)
    ):
        raise ValueError("RGB scope plan/manifest contract differs")
    planned = {_relative(item["relative"]): item for item in plan["tars"]}
    records = {_relative(item["relative"]): item for item in manifest.get("tars", [])}
    if planned.keys() != records.keys():
        raise ValueError("RGB scope TAR set differs")
    paths: dict[str, tuple[Path, Path]] = {}
    for relative in planned:
        tar_path = (MEDIA_ROOT / relative).resolve()
        if not tar_path.is_relative_to(MEDIA_ROOT.resolve()):
            raise ValueError("RGB sparse TAR escapes media root")
        sidecar = tar_path.with_suffix(tar_path.suffix + ".json")
        paths[relative] = (tar_path, sidecar)
    if any(
        not tar_path.is_file() or not sidecar.is_file()
        for tar_path, sidecar in paths.values()
    ):
        return None
    admitted: list[dict[str, str]] = []
    for relative, expected in planned.items():
        record = records[relative]
        tar_path, sidecar = paths[relative]
        if read(sidecar) != record:
            raise ValueError("RGB sparse TAR sidecar differs from authorized manifest")
        record_payload = {key: value for key, value in record.items() if key != "receipt_sha256"}
        if not (
            record.get("schema") == "ejepa_rgb_sparse_tar_receipt_v1"
            and record.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
            and record.get("sparse_tar_bytes") == tar_path.stat().st_size
            and record.get("sparse_tar_sha256") == digest(tar_path)
            and record.get("receipt_sha256") == canonical(record_payload)
        ):
            raise ValueError("RGB sparse TAR receipt/hash differs")
        actual_members = {str(item["member"]): item for item in record.get("members", [])}
        for planned_member in expected["members"]:
            actual = actual_members.get(str(planned_member["member"]))
            if actual is None or not _member_covers(actual, planned_member):
                raise ValueError("RGB sparse TAR does not cover scoped plan member")
        samples = [item for item in record["members"] if item.get("decoded_pixel_parity_sample")]
        if len(samples) != 1:
            raise ValueError("RGB sparse TAR lacks exactly one decoded proof")
        sample = samples[0]
        with tarfile.open(tar_path, "r") as archive:
            stream = archive.extractfile(str(sample["member"]))
            if stream is None:
                raise FileNotFoundError(str(sample["member"]))
            with stream:
                shape, pixel_sha = _decode(stream.read())
        if shape != sample.get("decoded_rgb_shape") or pixel_sha != sample.get(
            "decoded_rgb_sha256"
        ):
            raise ValueError("RGB sparse TAR decoded proof differs")
        admitted.append({"relative": relative, "tar_sha256": record["sparse_tar_sha256"]})
    return {
        "schema": "ejepa_rgb_media_scope_admission_v1",
        "status": "PASS",
        "rotation": rotation,
        "role": role,
        "scope_evidence_sha256": canonical(scope),
        "plan_sha256": scope["plan_sha256"],
        "manifest_sha256": scope["manifest_sha256"],
        "authority_sha256": config["authority_sha256"],
        "media_root": str(MEDIA_ROOT),
        "verified_tars": admitted,
        "model_inference_before_admission": False,
        "optimizer_updates_inferred": False,
    }


def wait_scope(
    run: Path,
    config_path: Path,
    rotation: int,
    role: str,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    original_authority: str | None = None
    admitted_entries: dict[str, str] = {}
    while True:
        config = load_config(config_path)
        if original_authority is None:
            original_authority = config["authority_sha256"]
        elif config["authority_sha256"] != original_authority:
            raise ValueError("RGB media authority changed while waiting")
        for key, item in config["scope_map"].items():
            observed = canonical(item)
            if key in admitted_entries and admitted_entries[key] != observed:
                raise ValueError("Existing RGB scope evidence changed")
        for admission_path in run.glob("RGB_MEDIA_SCOPE_ADMISSION_R*_*.json"):
            prior = read(admission_path)
            key = f"R{prior.get('rotation')}_{prior.get('role')}"
            current = config["scope_map"].get(key)
            if (
                current is None
                or prior.get("authority_sha256") != original_authority
                or prior.get("scope_evidence_sha256") != canonical(current)
            ):
                raise ValueError("Previously admitted RGB scope evidence changed or vanished")
        evidence = verify_scope(config, rotation, role)
        if evidence is not None:
            admitted_entries[f"R{rotation}_{role}"] = canonical(
                config["scope_map"][f"R{rotation}_{role}"]
            )
            target = run / f"RGB_MEDIA_SCOPE_ADMISSION_R{rotation}_{role}.json"
            if target.is_file() and read(target) != evidence:
                raise ValueError("Existing RGB media scope admission differs")
            if not target.exists():
                atomic_json(target, evidence)
            return evidence
        deadline = datetime.fromisoformat(str(config["deadline_utc"]))
        stops = [
            run / "STOP_REQUEST",
            run / f"rotations/{rotation}/STOP_REQUEST",
            run / f"rotations/{rotation}/fits/R_CTX/STOP_REQUEST",
        ]
        if any(path.exists() for path in stops):
            raise InterruptedError("STOP_REQUEST while waiting for RGB media")
        if datetime.now(UTC) >= deadline:
            raise TimeoutError("Authorized RGB deadline reached while waiting for media")
        atomic_json(
            run / f"RGB_MEDIA_WAIT_R{rotation}_{role}.json",
            {
                "schema": "ejepa_rgb_media_wait_v1",
                "status": "WAITING_MEDIA_SCOPE",
                "rotation": rotation,
                "role": role,
                "pid": os.getpid(),
                "model_inference_started": False,
                "optimizer_updates_inferred": False,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        sleeper(float(config.get("poll_seconds", 15)))


@contextmanager
def routed_runtime(run: Path, rotation: int, config_path: Path) -> Iterator[None]:
    from operational.final_refit import materialize
    from operational.final_refit import scientific_oof_pipeline as pipeline

    original_reader, original_rgb_raw = materialize.TarFrameReader, pipeline.rgb_raw

    def routed_reader(root: Path, *args: Any, **kwargs: Any) -> Any:
        requested = Path(root)
        return original_reader(
            MEDIA_ROOT if requested == HISTORICAL_ROOT else requested, *args, **kwargs
        )

    def gated_rgb_raw(*args: Any, **kwargs: Any) -> Any:
        output = kwargs.get("output") if "output" in kwargs else args[3]
        if not isinstance(output, (str, os.PathLike)):
            raise TypeError("RGB output path is absent")
        role = Path(output).parent.name
        if role not in {"refiner", "generation"}:
            raise ValueError(f"Unexpected RGB role: {role}")
        wait_scope(run, config_path, rotation, role)
        return original_rgb_raw(*args, **kwargs)

    materialize.TarFrameReader, pipeline.rgb_raw = routed_reader, gated_rgb_raw
    try:
        yield
    finally:
        materialize.TarFrameReader, pipeline.rgb_raw = original_reader, original_rgb_raw


def install_queue_heartbeat(
    module: Any,
    clock: Any = time.monotonic,
    *,
    media_config: Path,
    heartbeat_installer: Any,
) -> None:
    heartbeat_installer(module, clock=clock)
    original = module.run_rgb_downstream

    def run(run: Path, rotation: int, **kwargs: Any) -> Any:
        with routed_runtime(run, rotation, media_config):
            return original(run, rotation, **kwargs)

    module.run_rgb_downstream = run


@contextmanager
def composed_v5_hook(v5: Any, media_config: Path) -> Iterator[None]:
    original = v5.install_queue_heartbeat

    def install(module: Any, clock: Any = time.monotonic) -> None:
        install_queue_heartbeat(
            module,
            clock=clock,
            media_config=media_config,
            heartbeat_installer=original,
        )

    v5.install_queue_heartbeat = install
    try:
        yield
    finally:
        v5.install_queue_heartbeat = original


def _consume(argv: list[str], name: str) -> tuple[str, list[str]]:
    if argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = argv.index(name)
    return argv[index + 1], [*argv[:index], *argv[index + 2 :]]


def main() -> None:
    source, ops = Path(__file__).resolve(), Path(__file__).resolve().parent
    v5_path = ops / "runpod_rgb_context_v5.py"
    if digest(v5_path) != V5_SHA256:
        raise ValueError("Frozen RGB context V5 bytes differ")
    raw_config, forwarded = _consume(sys.argv, "--media-config")
    config_path = Path(raw_config).resolve(strict=True)
    config = load_config(config_path)
    run = Path(forwarded[forwarded.index("--run") + 1]).resolve(strict=True)
    atomic_json(
        run / "RGB_OPERATIONAL_RECOVERY_V6.json",
        {
            "schema": "ejepa_rgb_frame_gate_routing_v6",
            "status": "PASS",
            "source_sha256": digest(source),
            "context_v5_sha256": digest(v5_path),
            "media_authority_sha256": config["authority_sha256"],
            "initial_media_config_sha256": config["config_sha256"],
            "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
            "reader_routing_scope": "materializer_call_only",
            "original_frames_crops_clocks_changed": False,
            "model_inference_requires_scope_admission": True,
        },
    )
    sys.path.insert(0, str(ops))
    import runpod_rgb_context_v5 as v5

    original_argv, sys.argv = sys.argv, forwarded
    try:
        with composed_v5_hook(v5, config_path):
            v5.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
