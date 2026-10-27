"""Admit immutable legacy RGB decoded proofs without weakening V6 verification."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

V6_SHA256 = "88bf04a2ad2b2938b93b1a1a2a8099de07365992ea0a1ff75ef6bd81684d19ca"
V7_SHA256 = "ae66d985c5a2bfafac2050d5e75a32e85de9f207509fadc34d978301a43c916b"
V8_SHA256 = "53d170fb9c880c983cf22a64659d5b986abb4d058d98cbc1acff3823d3458efe"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_decoded_proof(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve new receipts; narrowly recognize the immutable legacy format."""
    members = record.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("RGB sparse TAR decoded proof members differ")
    flagged = [item for item in members if item.get("decoded_pixel_parity_sample")]
    if len(flagged) == 1:
        return flagged
    if flagged:
        raise ValueError("RGB sparse TAR lacks exactly one decoded proof")
    legacy = (
        "decoded_pixel_samples" not in record
        and "decoded_pixel_parity_basis" not in record
        and record.get("decoded_pixel_parity") is True
        and record.get("payload_recompressed") is False
        and all("decoded_pixel_parity_sample" not in item for item in members)
    )
    if not legacy:
        raise ValueError("RGB sparse TAR lacks exactly one decoded proof")
    for item in members:
        shape = item.get("decoded_rgb_shape")
        if not (
            isinstance(item.get("member"), str)
            and item["member"]
            and isinstance(item.get("payload_bytes"), int)
            and item["payload_bytes"] > 0
            and isinstance(item.get("compressed_payload_sha256"), str)
            and _SHA256.fullmatch(item["compressed_payload_sha256"])
            and isinstance(item.get("decoded_rgb_sha256"), str)
            and _SHA256.fullmatch(item["decoded_rgb_sha256"])
            and isinstance(shape, list)
            and len(shape) == 3
            and all(isinstance(value, int) and value > 0 for value in shape)
            and shape[2] == 3
        ):
            raise ValueError("Legacy RGB decoded proof closure differs")
    return [members[0]]


class _ProofPatch(ast.NodeTransformer):
    def __init__(self) -> None:
        self.selector_replaced = 0
        self.guard_removed = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "samples"
            and "decoded_pixel_parity_sample" in ast.unparse(node.value)
        ):
            self.selector_replaced += 1
            node.value = ast.Call(
                func=ast.Name(id="_select_decoded_samples_v9", ctx=ast.Load()),
                args=[ast.Name(id="record", ctx=ast.Load())],
                keywords=[],
            )
        return node

    def visit_If(self, node: ast.If) -> ast.AST | None:
        rendered = ast.unparse(node)
        if (
            "len(samples) != 1" in rendered
            and "RGB sparse TAR lacks exactly one decoded proof" in rendered
        ):
            self.guard_removed += 1
            return None
        return self.generic_visit(node)


def patched_verify(v6: Any) -> Callable[..., dict[str, Any] | None]:
    """Compile only V6 verify_scope with the proof selector replaced."""
    source = Path(v6.__file__).resolve(strict=True)
    if digest(source) != V6_SHA256:
        raise ValueError("Frozen RGB context V6 bytes differ")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_scope"
    )
    patch = _ProofPatch()
    transformed = patch.visit(function)
    if patch.selector_replaced != 1 or patch.guard_removed != 1:
        raise ValueError("RGB V6 decoded-proof AST contract differs")
    module = ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[]))
    namespace = dict(vars(v6))
    namespace["_select_decoded_samples_v9"] = select_decoded_proof
    exec(compile(module, str(source), "exec"), namespace)
    verified = namespace["verify_scope"]

    def verify(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        # V8 temporarily selects a physical root around each compositional scope.
        # Keep the compiled V6 function's module namespace synchronized per call.
        namespace["MEDIA_ROOT"] = v6.MEDIA_ROOT
        return verified(*args, **kwargs)

    return verify


def main() -> None:
    source, scripts = Path(__file__).resolve(), Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts))
    import runpod_rgb_context_v6 as v6

    compositional = "--media-layout" in sys.argv
    parent_path = scripts / (
        "runpod_rgb_context_v8.py" if compositional else "runpod_rgb_context_v7.py"
    )
    expected_parent = V8_SHA256 if compositional else V7_SHA256
    if digest(parent_path) != expected_parent:
        raise ValueError("Frozen RGB context parent bytes differ")
    compatibility_verify = patched_verify(v6)
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    binding = {
        "schema": "ejepa_rgb_legacy_decoded_proof_compatibility_v9",
        "status": "PASS",
        "source_sha256": digest(source),
        "context_v6_sha256": digest(Path(v6.__file__)),
        "parent_context_sha256": digest(parent_path),
        "parent_context_version": 8 if compositional else 7,
        "policy": "NEW_EXACTLY_ONE; LEGACY_ALL_MEMBERS_CLOSED_THEN_FIRST_DECODED",
        "original_markers_rewritten": False,
        "optimizer_updates_inferred": False,
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V9.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != binding:
        raise ValueError("Existing RGB V9 recovery binding differs")
    if not target.exists():
        atomic_json(target, binding)
    original = v6.verify_scope
    v6.verify_scope = compatibility_verify
    try:
        if compositional:
            import runpod_rgb_context_v8 as parent
        else:
            import runpod_rgb_context_v7 as parent
        parent.main()
    finally:
        v6.verify_scope = original


if __name__ == "__main__":
    main()

