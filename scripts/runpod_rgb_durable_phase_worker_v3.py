"""Bind uploader V6 and stdin eviction to the frozen durable RGB worker V1."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import psutil

V1_SHA256 = "cc8648fb050cd84cf880b98def48e2aac66694612cfb13b14f86caf9486d51f4"
BASE_CONFIG_SHA256 = "f4078105d9cb3398b033d47123b3b643734c70ada26d0b13914c2a1c92415f3a"
V6_SHA256 = "753c8899a72a7f2dcbc692e79fe18451f70fe13c6d75002771fe6a9f36a0c937"
V5_NAME = "upload_rgb_sparse_frames_v5.py"
V6_NAME = "upload_rgb_sparse_frames_v6.py"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_matches(pid: int, argv: list[str]) -> bool:
    """Match exact live argv through psutil without a locale-sensitive shell."""
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and process.cmdline() == argv
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError) as error:
        raise RuntimeError(f"Cannot prove process identity for PID {pid}") from error


def _load_frozen_v1() -> Any:
    source = Path(__file__).resolve().with_name("runpod_rgb_durable_phase_worker_v1.py")
    if digest(source) != V1_SHA256:
        raise ValueError("Frozen durable RGB worker V1 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_rgb_durable_phase_worker_v1_v3", source)
    if spec is None or spec.loader is None:
        raise ImportError(str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_v3() -> Any:
    module: Any = _load_frozen_v1()
    original_load_config = module.load_config
    remote_eviction = module.REMOTE_EVICTION
    needle = "payload=json.loads(sys.argv[2])"
    if remote_eviction.count(needle) != 1:
        raise ValueError("Frozen eviction payload protocol differs")
    remote_eviction = remote_eviction.replace(needle, "payload=json.load(sys.stdin)")

    def load_config(path: Path) -> dict[str, Any]:
        overlay = module.read_json(path.resolve(strict=True))
        if overlay.get("schema") != "ejepa_rgb_durable_phase_worker_config_v2":
            raise ValueError("Durable RGB worker V2 config schema differs")
        base_path = Path(str(overlay.get("base_config_path"))).resolve(strict=True)
        if (
            digest(base_path) != BASE_CONFIG_SHA256
            or overlay.get("base_config_sha256") != BASE_CONFIG_SHA256
        ):
            raise ValueError("Frozen durable worker base config differs")
        v6_path = Path(str(overlay.get("uploader_v6_path"))).resolve(strict=True)
        if digest(v6_path) != V6_SHA256 or overlay.get("uploader_v6_sha256") != V6_SHA256:
            raise ValueError("Uploader V6 bytes differ")
        value = copy.deepcopy(original_load_config(base_path))
        for key in ("r4_refiner", "r2_generation"):
            argv = value["uploads"][key]["argv"]
            matches = [index for index, item in enumerate(argv) if item.endswith(V5_NAME)]
            if len(matches) != 1:
                raise ValueError(f"{key} frozen V5 entrypoint differs")
            argv[matches[0]] = str(v6_path)
            if "--complete-manifest" not in argv or "--complete-manifest-sha256" not in argv:
                raise ValueError(f"{key} lacks complete manifest closure")
        value["config_overlay_path"] = str(path.resolve(strict=True))
        value["config_overlay_sha256"] = digest(path.resolve(strict=True))
        value["uploader_v6_sha256"] = V6_SHA256
        return value

    def evict(self: Any) -> None:
        eviction = self.config["eviction"]
        authority_path = Path(eviction["authority_path"])
        module.validate_eviction_authority(authority_path)
        r4, records = self._upload("r4_refiner")
        _, r2_records = self._upload("r2_generation")
        if set(records) & set(r2_records):
            raise ValueError("R4 raw eviction overlaps R2 generation inputs")
        payload = {
            "authority_sha256": module.sha256(authority_path),
            "manifest_sha256": r4["manifest_sha256"],
            "records": [
                {
                    "relative": relative,
                    "bytes": item["sparse_tar_bytes"],
                    "sha256": item["sparse_tar_sha256"],
                    "receipt_sha256": item["receipt_sha256"],
                }
                for relative, item in sorted(records.items())
            ],
        }
        encoded = base64.b64encode(remote_eviction.encode()).decode()
        argv = list(eviction["ssh_argv_prefix"]) + [
            f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\" "
            + eviction["remote_root"]
        ]
        self.run(
            argv,
            input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            text=True,
            check=True,
            timeout=1800,
        )

    module.process_matches = process_matches
    module.load_config = load_config
    module.Worker.evict = evict
    module.REMOTE_EVICTION = remote_eviction
    return module


def main() -> int:
    return int(install_v3().main())


if __name__ == "__main__":
    raise SystemExit(main())
