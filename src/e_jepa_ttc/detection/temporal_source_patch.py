"""Materialize a hash-bound MonoDGP source split for temporal feature fusion."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from e_jepa_ttc.detection.temporal_detector import TEMPORAL_PORTABLE_VARIANT

R0_MONODGP_SHA256 = "ac16913d545b470cf1f741df2dd783f0ca82eaca515cc94a824a7074c4113c6e"
R0_PORTABLE_TREE_MANIFEST_SHA256 = (
    "0a7d44d6ec24e3a247a6d347460b59041740983f0a1d3384217e6279cd77a57f"
)
R0_RUNTIME_PREREGISTRATION_SHA256 = (
    "3a6438ab23ef51717af4143843af8e154266fbddad6fb05427dd22aa3bc85e5d"
)
MONODGP_RELATIVE_PATH = Path("lib/models/monodgp/monodgp.py")
IMPORT_ANCHOR = "from e_jepa_ttc.detection.velocity_extension import matched_velocity_l1\n"
CLASS_ANCHOR = "class MonoDGP(nn.Module):\n"
EXTRACTION_START = "        features, pos = self.backbone(images)\n"
DECODER_START = "        # region enhancement\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


@dataclass(frozen=True)
class TemporalSourceContract:
    """Exact structural and byte contract for a copied source transformation."""

    source_sha256: str = R0_MONODGP_SHA256
    portable_tree_manifest_sha256: str = R0_PORTABLE_TREE_MANIFEST_SHA256
    runtime_preregistration_sha256: str = R0_RUNTIME_PREREGISTRATION_SHA256
    import_anchor: str = IMPORT_ANCHOR
    class_anchor: str = CLASS_ANCHOR
    extraction_start: str = EXTRACTION_START
    decoder_start: str = DECODER_START


def _mono_forward(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MonoDGP"
    ]
    if len(classes) != 1:
        raise ValueError("Expected exactly one MonoDGP class")
    forwards = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1 or forwards[0].end_lineno is None:
        raise ValueError("Expected exactly one bounded MonoDGP.forward")
    return forwards[0]


def transform_monodgp_source(
    source: str,
    contract: TemporalSourceContract | None = None,
) -> str:
    """Split the exact forward at projected features without renaming modules."""
    contract = TemporalSourceContract() if contract is None else contract
    if _sha256(source.encode("utf-8")) != contract.source_sha256:
        raise ValueError("Temporal MonoDGP source SHA-256 differs from the contract")
    for name, marker in (
        ("import", contract.import_anchor),
        ("class", contract.class_anchor),
        ("extraction", contract.extraction_start),
        ("decoder", contract.decoder_start),
    ):
        if source.count(marker) != 1:
            raise ValueError(f"Temporal source {name} marker is absent or ambiguous")
    forward = _mono_forward(source)
    lines = source.splitlines(keepends=True)
    start = forward.lineno - 1
    end = forward.end_lineno
    forward_lines = lines[start:end]
    extraction_index = forward_lines.index(contract.extraction_start)
    decoder_index = forward_lines.index(contract.decoder_start)
    if extraction_index >= decoder_index:
        raise ValueError("Temporal extraction marker must precede decoder marker")
    extraction_body = "".join(forward_lines[extraction_index:decoder_index]).rstrip() + "\n"
    decoder_body = "".join(forward_lines[decoder_index:]).rstrip() + "\n"
    if not decoder_body.rstrip().endswith("return out"):
        raise ValueError("MonoDGP.forward no longer ends with the expected output return")

    replacement = f'''    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """Run the unchanged split producer for single-frame compatibility."""
        state = self.extract_projected_features(images)
        return self.decode_projected_features(
            state, state.features, calibs, targets, img_sizes, dn_args)

    def extract_projected_features(self, images):
        """Run backbone and all input projections for temporal fusion."""
{extraction_body}
        return ProjectedFeatureState(
            tuple(srcs), (tuple(pos), tuple(masks)))

    def decode_projected_features(
            self, state, projected_features, calibs, targets, img_sizes,
            dn_args=None):
        """Continue through the unchanged region, geometry and velocity heads."""
        del dn_args
        if not isinstance(state, ProjectedFeatureState):
            raise TypeError('state must be ProjectedFeatureState')
        pos, masks = state.decoder_context
        srcs = list(projected_features)
        if len(srcs) != self.num_feature_levels:
            raise ValueError('projected feature count differs from detector levels')
{decoder_body}'''
    transformed = "".join(lines[:start]) + replacement + "".join(lines[end:])
    transformed = transformed.replace(
        contract.import_anchor,
        contract.import_anchor
        + "from e_jepa_ttc.detection.temporal_detector import (\n"
        + "    ProjectedFeatureBackend, ProjectedFeatureState)\n",
        1,
    )
    transformed = transformed.replace(
        contract.class_anchor, "class MonoDGP(ProjectedFeatureBackend):\n", 1
    )
    compile(transformed, "monodgp_temporal.py", "exec")
    tree = ast.parse(transformed)
    mono = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MonoDGP"
    )
    methods = {
        node.name for node in mono.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {"forward", "extract_projected_features", "decode_projected_features"}
    if not required.issubset(methods):
        raise ValueError("Transformed MonoDGP lacks required split methods")
    return transformed


def unified_patch(source: str, transformed: str) -> str:
    """Return a deterministic reviewable unified diff."""
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            transformed.splitlines(keepends=True),
            fromfile=f"a/{MONODGP_RELATIVE_PATH.as_posix()}",
            tofile=f"b/{MONODGP_RELATIVE_PATH.as_posix()}",
        )
    )


def portable_tree_hashes(root: Path) -> dict[str, str]:
    """Hash the exact runtime files included by the campaign source snapshot."""
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def materialize_temporal_source(
    *,
    source_root: str | Path,
    destination: str | Path,
    checkpoint: str | Path,
    runtime_preregistration: str | Path,
    patch_output: str | Path,
    manifest_output: str | Path,
    forbidden_destination: str | Path,
    contract: TemporalSourceContract | None = None,
) -> dict[str, object]:
    """Copy, split and hash one new portable variant without overwriting artifacts."""
    source_root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    checkpoint = Path(checkpoint).resolve()
    runtime_preregistration = Path(runtime_preregistration).resolve()
    patch_output = Path(patch_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    forbidden_destination = Path(forbidden_destination).resolve()
    contract = TemporalSourceContract() if contract is None else contract
    if destination == forbidden_destination:
        raise ValueError("Temporal destination aliases the active single-frame portable source")
    named_paths = {
        "source_root": source_root,
        "destination": destination,
        "patch_output": patch_output,
        "manifest_output": manifest_output,
    }
    if len(set(named_paths.values())) != len(named_paths):
        raise ValueError("Source, destination, patch and manifest paths must be distinct")
    if destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
        raise ValueError("Temporal source and destination must not contain one another")
    for name, artifact in (("patch", patch_output), ("manifest", manifest_output)):
        if artifact.is_relative_to(source_root) or artifact.is_relative_to(destination):
            raise ValueError(f"Temporal {name} output must stay outside source and destination")
    if patch_output == manifest_output:
        raise ValueError("Temporal patch and manifest outputs must differ")
    if destination.exists() or patch_output.exists() or manifest_output.exists():
        raise FileExistsError("Use new temporal destination, patch and manifest paths")
    if (
        not source_root.is_dir()
        or not checkpoint.is_file()
        or not runtime_preregistration.is_file()
    ):
        raise FileNotFoundError("Frozen source, checkpoint or runtime preregistration is absent")
    preregistration_sha256 = _file_sha256(runtime_preregistration)
    if preregistration_sha256 != contract.runtime_preregistration_sha256:
        raise ValueError("Runtime preregistration SHA-256 differs from the frozen contract")
    preregistration = json.loads(runtime_preregistration.read_text(encoding="utf-8"))
    if preregistration.get("schema") != "eap_detection_campaign_runtime_preregistration_v1":
        raise ValueError("Runtime preregistration schema differs")
    source_file = source_root / MONODGP_RELATIVE_PATH
    source_bytes = source_file.read_bytes()
    source = source_bytes.decode("utf-8")
    transformed = transform_monodgp_source(source, contract)
    patch = unified_patch(source, transformed)
    if not patch:
        raise ValueError("Temporal transformation unexpectedly produced an empty patch")

    before = portable_tree_hashes(source_root)
    declared_tree = preregistration.get("portable_source_sha256")
    if not isinstance(declared_tree, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in declared_tree.items()
    ):
        raise ValueError("Runtime preregistration portable tree is malformed")
    declared_manifest = preregistration.get("portable_source_sha256_manifest")
    if (
        before != declared_tree
        or _canonical_sha256(before) != contract.portable_tree_manifest_sha256
        or declared_manifest != contract.portable_tree_manifest_sha256
    ):
        raise ValueError("Portable source tree differs from the frozen R0 manifest")
    shutil.copytree(source_root, destination)
    transformed_file = destination / MONODGP_RELATIVE_PATH
    transformed_file.write_text(transformed, encoding="utf-8", newline="")
    after = portable_tree_hashes(destination)
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    if changed != [MONODGP_RELATIVE_PATH.as_posix()]:
        raise ValueError(f"Temporal copy changed unexpected files: {changed}")

    patch_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    patch_output.write_text(patch, encoding="utf-8", newline="")
    manifest: dict[str, object] = {
        "schema": "monodgp_projected_temporal_source_v1",
        "variant": TEMPORAL_PORTABLE_VARIANT,
        "source_root": str(source_root),
        "destination": str(destination),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "runtime_preregistration": str(runtime_preregistration),
        "runtime_preregistration_sha256": preregistration_sha256,
        "input_monodgp_sha256": _sha256(source_bytes),
        "output_monodgp_sha256": _file_sha256(transformed_file),
        "input_tree_sha256": _canonical_sha256(before),
        "input_tree_file_count": len(before),
        "output_tree_sha256": _canonical_sha256(after),
        "patch_sha256": _file_sha256(patch_output),
        "changed_files": changed,
        "state_dict_module_names_changed": False,
        "real_model_numerical_parity": "PENDING_GPU_GATE",
        "status": "MATERIALIZED_NOT_EXECUTED",
    }
    manifest["manifest_canonical_sha256"] = _canonical_sha256(manifest)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
