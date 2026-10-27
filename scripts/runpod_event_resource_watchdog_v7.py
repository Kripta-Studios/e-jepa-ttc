"""Reset downstream retries only after new verified PAIR_E fragments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

V6_SHA256 = "576b38484ae5f63932cf85d07fa0f9049f717cf572b08867c34f4425c66626f9"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def durable_pair_fragments(
    module: Any, task: dict[str, Any], anchor: object
) -> dict[str, Any] | None:
    """Return a strictly newer, stable FragmentStore prefix or no reset authority."""
    if task.get("fit") != module.HEAD:
        return None
    root = (
        Path(task["run"])
        / f"rotations/{int(task['rotation'])}/caches/PAIR_E.parts"
    )
    binding_path = root / "BINDING.json"
    if not binding_path.is_file() or binding_path.is_symlink():
        return None
    try:
        binding = module.read(binding_path)
        parents = binding.get("parents")
        if not (
            binding.get("schema") == "scientific_fold_pair133_v1"
            and binding.get("precision") == "float32"
            and isinstance(binding.get("rows_sha256"), str)
            and _SHA256.fullmatch(binding["rows_sha256"])
            and isinstance(parents, dict)
            and set(parents) == {"E_A5"}
            and isinstance(parents["E_A5"], str)
            and _SHA256.fullmatch(parents["E_A5"])
        ):
            return None
        binding_sha = canonical(binding)
        receipts = sorted(root.glob("part_*.json"))
        if not receipts:
            return None
        indices = []
        identities: list[dict[str, Any]] = []
        import numpy as np

        for receipt_path in receipts:
            match = re.fullmatch(r"part_(\d{6})\.json", receipt_path.name)
            if match is None or receipt_path.is_symlink():
                return None
            index = int(match.group(1))
            indices.append(index)
            part_path = receipt_path.with_suffix(".npz")
            if part_path.is_symlink() or not part_path.is_file():
                return None
            before = part_path.stat()
            part_sha = digest(part_path)
            after = part_path.stat()
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                return None
            receipt = module.read(receipt_path)
            rows = receipt.get("rows")
            if not (
                receipt.get("binding_sha256") == binding_sha
                and receipt.get("sha256") == part_sha
                and receipt.get("optimizer_updates") == 0
                and isinstance(rows, int)
                and 1 <= rows <= 32
            ):
                return None
            with np.load(part_path, allow_pickle=False) as archive:
                if set(archive.files) != {"features"}:
                    return None
                features = archive["features"]
                if features.shape != (rows, 133) or str(features.dtype) != "float32":
                    return None
            identities.append(
                {
                    "index": index,
                    "rows": rows,
                    "part_sha256": part_sha,
                    "receipt_sha256": digest(receipt_path),
                }
            )
        if indices != list(range(len(indices))):
            return None
        if anchor is not None:
            if not isinstance(anchor, dict):
                return None
            old_count = anchor.get("parts")
            if (
                anchor.get("binding_sha256") != binding_sha
                or not isinstance(old_count, int)
                or old_count < 0
                or old_count >= len(identities)
                or anchor.get("prefix_sha256")
                != canonical(
                    {"binding_sha256": binding_sha, "parts": identities[:old_count]}
                )
            ):
                return None
        return {
            "binding_sha256": binding_sha,
            "parts": len(identities),
            "rows": sum(int(item["rows"]) for item in identities),
            "prefix_sha256": canonical(
                {"binding_sha256": binding_sha, "parts": identities}
            ),
        }
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return None


def install_v7() -> Any:
    source = Path(__file__).resolve()
    v6_path = source.with_name("runpod_event_resource_watchdog_v6.py")
    if digest(v6_path) != V6_SHA256:
        raise ValueError("Frozen EVENT watchdog V6 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v6_v7", v6_path)
    if spec is None or spec.loader is None:
        raise ImportError(v6_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v6()
    original_validate = module.validate_config
    original_adopt = module.Watchdog._adopt
    original_restart = module.Watchdog._restart

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_fragment_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V7 fragment overlay binding differs")
        return value

    def observe(task: dict[str, Any], current: dict[str, Any]) -> None:
        anchor = current.get("retry_anchor_pair_fragments")
        progress = durable_pair_fragments(module, task, anchor)
        if progress is None:
            return
        if anchor is None:
            current.update(
                retry_anchor_pair_fragments=progress,
                retry_fragment_anchor_initialized_without_reset=True,
            )
            return
        previous = int(current.get("retries", 0))
        current.update(
            retries=0,
            retry_anchor_pair_fragments=progress,
            retry_reset_previous_retries=previous,
            retry_reset_fragment_parts=progress["parts"],
            retry_reset_fragment_rows=progress["rows"],
            retry_fragment_reset_count=int(current.get("retry_fragment_reset_count", 0)) + 1,
        )

    def adopt(self: Any, task: dict[str, Any], current: dict[str, Any]) -> Any:
        observe(task, current)
        return original_adopt(self, task, current)

    def restart(self: Any, task: dict[str, Any], current: dict[str, Any]) -> None:
        observe(task, current)
        original_restart(self, task, current)

    module.validate_config = validate_config
    module.Watchdog._adopt = adopt
    module.Watchdog._restart = restart
    module.Watchdog._observe_downstream_fragment_retry_boundary = staticmethod(observe)
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v7().main())


if __name__ == "__main__":
    raise SystemExit(main())
