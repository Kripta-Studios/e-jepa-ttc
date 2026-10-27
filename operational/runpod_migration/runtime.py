"""Explicit Linux path/resource overlay preserving frozen scientific file bytes."""

# ruff: noqa: ANN401 -- importlib delegates and frozen module APIs are heterogeneous.

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .inventory import digest
from .paths import PathMap, mapped_path_class


class ProjectLoader(importlib.abc.Loader):
    """Only substitute the imported Path alias in project modules after execution."""

    def __init__(self, wrapped: Any, path_type: type[Path]) -> None:
        self.wrapped, self.path_type = wrapped, path_type

    def create_module(self, spec: Any) -> ModuleType | None:
        return self.wrapped.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        if vars(module).get("Path") is Path:
            vars(module)["Path"] = self.path_type


class ProjectFinder(importlib.abc.MetaPathFinder):
    """Never intercept third-party packages or the migration implementation itself."""

    def __init__(self, path_type: type[Path]) -> None:
        self.path_type = path_type

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if not fullname.startswith(("operational.", "e_jepa_ttc.")) or fullname.startswith(
            "operational.runpod_migration"
        ):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = ProjectLoader(spec.loader, self.path_type)
        return spec


def install(
    mapping: PathMap, *, budget_bytes: int = 31_000_000_000, origin_git_commit: str | None = None
) -> None:
    """Install the explicitly recorded overlay before importing scientific runners."""
    from .resources import LinuxResourceMonitor

    path_type = mapped_path_class(mapping)
    sys.meta_path.insert(0, ProjectFinder(path_type))
    # Do not permit silently patching an already running training process.
    if "operational.final_refit.scientific_fold_producers" in sys.modules:
        raise RuntimeError("Migration runtime must be installed before scientific modules")
    from operational.final_refit import scientific_authority, shared_resources

    def load_config(run: Path) -> dict[str, Any]:
        # Verify the original canonical payload and every original file digest.
        # Only returned routing strings change; original JSON and hashes remain intact.
        from operational.final_refit.recipe import canonical_sha256

        original = json.loads((run / "SCIENTIFIC_CONFIG.json").read_text(encoding="utf-8"))
        if original.get("schema") != "scientific_oof_execution_v1" or canonical_sha256(
            {k: v for k, v in original.items() if k != "config_sha256"}
        ) != original.get("config_sha256"):
            raise ValueError("Original scientific authority changed")
        for category in ("input_sha256", "source_sha256"):
            if not isinstance(original.get(category), dict) or not original[category]:
                raise ValueError(f"Missing {category}")
            for source, expected in original[category].items():
                if digest(path_type(source)) != expected:
                    raise ValueError(f"Original authority binding changed: {source}")
        for key in ("fold_contract", "config", "rgb_source", "teacher", "recipe"):
            if original["input_sha256"].get(original[key]) != digest(path_type(original[key])):
                raise ValueError(f"Original authority omits {key}")
        old = json.loads(
            (path_type(original["active_rgb_runtime"]) / "PRODUCER_PLAN.json").read_text(
                encoding="utf-8"
            )
        )
        repo = Path(__file__).resolve().parents[2]
        for relative, expected in old["executed_source_sha256"].items():
            if digest(repo / relative) != expected:
                raise ValueError(f"Historical source changed: {relative}")
        if digest(path_type(original["config"])) != old["input_sha256"]["producer_config"]:
            raise ValueError("Historical producer recipe changed")
        result = dict(original)
        for key, value in original.items():
            if isinstance(value, str) and (":\\" in value or ":/" in value):
                result[key] = mapping.translate(value)
        return result

    scientific_authority.load_config = load_config

    class BoundMonitor(LinuxResourceMonitor):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, budget_bytes=budget_bytes, **kwargs)

    shared_resources.SharedResourceMonitor = BoundMonitor
    if origin_git_commit is not None:
        if len(origin_git_commit) != 40 or any(
            c not in "0123456789abcdef" for c in origin_git_commit
        ):
            raise ValueError("Invalid original Git base commit")
        from operational.final_refit import training

        # A bundle deliberately omits .git and credentials. Preserve the base
        # commit as provenance only; exact dirty/untracked execution bytes are
        # identified by original scientific SHA bindings and the bundle digest.
        training._git_commit = lambda: origin_git_commit
