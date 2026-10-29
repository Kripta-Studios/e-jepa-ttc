"""Isolate only the two summary linspaces on twelve bound transport snapshots.

The control retains the preceding opaque weighted-mean/sampler partition. The
candidate changes only x/y construction in a private copy of _summarise_one.
No RAW, checkpoint, normalizer or TTC head is loaded; no latency is admitted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import textwrap
import traceback
import types
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
CANDIDATE = "native_weighted_mean_and_linspace"


@lru_cache(maxsize=1)
def native_summary_linspace_operator() -> Any:
    """Preserve ATen coordinate generation, including the input dtype and device."""
    import torch

    @torch.library.custom_op(
        "train40_transport_diagnostics::native_summary_linspace",
        mutates_args=(),
        schema="(Tensor like, int steps) -> Tensor",
    )
    def coordinates(like: Tensor, steps: int) -> Tensor:
        if not like.is_floating_point() or steps < 1:
            raise ValueError("Summary coordinates require floating input and positive steps")
        return torch.linspace(-0.5, 0.5, steps, device=like.device, dtype=like.dtype)

    @coordinates.register_fake
    def fake(like: Tensor, steps: int) -> Tensor:
        if not like.is_floating_point() or steps < 1:
            raise ValueError("Summary coordinates require floating input and positive steps")
        return like.new_empty((steps,))

    return coordinates


def rewrite_summary_linspaces(source: str) -> tuple[ast.Module, str]:
    """Replace exactly the two frozen assignments, rejecting any linspace drift."""
    parsed = ast.parse(textwrap.dedent(source))
    expected = {
        ast.dump(
            ast.parse(
                f"{axis} = torch.linspace(-0.5, 0.5, {size}, device=device, dtype=dtype)"
            ).body[0]
        ): size
        for axis, size in (("x", "width"), ("y", "height"))
    }
    calls = [
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "linspace"
    ]
    replacements = []

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST:  # noqa: N802
            size = expected.get(ast.dump(node))
            if size is None:
                return self.generic_visit(node)
            replacements.append(size)
            node.value = ast.copy_location(
                ast.parse(f"_NATIVE_SUMMARY_LINSPACE(forward.dx, {size})", mode="eval").body,
                node.value,
            )
            return node

    transformed = Rewrite().visit(parsed)
    if len(calls) != 2 or sorted(replacements) != ["height", "width"]:
        raise ValueError("Expected exactly the two frozen summary linspace assignments")
    if not isinstance(transformed, ast.Module):
        raise TypeError("Expected a complete private summary module")
    ast.fix_missing_locations(transformed)
    return transformed, hashlib.sha256(ast.dump(transformed).encode()).hexdigest()


@lru_cache(maxsize=1)
def make_native_linspace_physical() -> tuple[Any, dict[str, Any]]:
    """Clone the admitted diagnostic partition without changing scientific globals."""
    from e_jepa_ttc.models import local_transport as oracle
    from scripts.diagnose_train40_weighted_mean import make_native_mean_physical

    base, base_binding = make_native_mean_physical()
    source = textwrap.dedent(inspect.getsource(oracle._summarise_one))
    parsed, ast_hash = rewrite_summary_linspaces(source)
    name = f"_train40_native_summary_linspace_{ast_hash[:16]}"
    module = types.ModuleType(name)
    module.__dict__.update(base.__globals__)
    module.__name__, module.__package__, module.__spec__ = name, "", None
    scope = vars(module)
    scope["_NATIVE_SUMMARY_LINSPACE"] = native_summary_linspace_operator()
    exec(compile(parsed, __file__, "exec"), scope)  # noqa: S102
    for key in ("_weighted_slope", "transport_physical_features"):
        original = base.__globals__[key]
        clone = FunctionType(
            original.__code__, scope, original.__name__, original.__defaults__, original.__closure__
        )
        clone.__kwdefaults__ = dict(original.__kwdefaults__ or {})
        clone.__annotations__ = dict(original.__annotations__)
        scope[key] = clone
    sys.modules[name] = module
    return scope["transport_physical_features"], {
        "private_module": name,
        "base_partition": base_binding,
        "original_summary_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "rewritten_ast_sha256": ast_hash,
        "replacements": 2,
        "only_new_boundary": "ATEN_LINSPACE_MINUS_HALF_TO_HALF_ORIGINAL_DTYPE_DEVICE",
        "scientific_globals_modified": False,
    }


def make_linspace_probe(
    radius: int,
    temperature: float,
    *,
    deterministic_sampling: bool,
    backend: str = "eager",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the preceding numerical controls and vary only summary coordinates."""
    import torch
    from torch import nn

    from e_jepa_ttc.models import local_transport as oracle
    from scripts.diagnose_train40_weighted_mean import make_weighted_mean_probe

    physical, _ = make_native_linspace_physical()

    class Summary(nn.Module):
        def forward(
            self, forward: Any, reverse: Any, previous: Tensor, current: Tensor
        ) -> tuple[Tensor, Tensor]:
            return (
                physical(
                    forward,
                    reverse,
                    foreground_weight=previous,
                    radius=radius,
                    deterministic_sampling=deterministic_sampling,
                    reuse_reverse_sample=True,
                ),
                physical(
                    reverse,
                    forward,
                    foreground_weight=current,
                    radius=radius,
                    deterministic_sampling=deterministic_sampling,
                    reuse_reverse_sample=True,
                ),
            )

    compiled = torch.compile(
        Summary().eval(), backend=backend, fullgraph=True, dynamic=False, options=options
    )

    def candidate(
        previous: Tensor, current: Tensor, foreground_previous: Tensor, foreground_current: Tensor
    ) -> tuple[Tensor, Tensor]:
        forward = oracle.local_correlation_match(
            previous, current, radius=radius, temperature=temperature
        )
        reverse = oracle.local_correlation_match(
            current, previous, radius=radius, temperature=temperature
        )
        return compiled(forward, reverse, foreground_previous, foreground_current)

    controls = make_weighted_mean_probe(
        radius,
        temperature,
        deterministic_sampling=deterministic_sampling,
        backend=backend,
        options=options,
    )
    return {
        "original": controls["original"],
        "native_weighted_mean": controls["native_weighted_mean"],
        CANDIDATE: candidate,
    }


def coordinate_probe(
    *, device: str, backend: str, options: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
    """Expose standalone 16/32-coordinate values and bits, separate from slopes."""
    import torch
    from torch import nn

    native = native_summary_linspace_operator()
    records, tensors = [], {}

    class Coordinates(nn.Module):
        def __init__(self, steps: int, opaque: bool) -> None:
            super().__init__()
            self.steps, self.opaque = steps, opaque

        def forward(self, like: Tensor) -> Tensor:
            if self.opaque:
                return native(like, self.steps)
            return torch.linspace(-0.5, 0.5, self.steps, dtype=like.dtype, device=like.device)

    like = torch.zeros((), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for steps in (16, 32):
            original = torch.linspace(-0.5, 0.5, steps, dtype=like.dtype, device=like.device)
            row: dict[str, Any] = {
                "steps": steps,
                "original_values": original.cpu().tolist(),
                "variants": {},
            }
            tensors[f"original_{steps}"] = original
            for name, opaque in (("compiled", False), ("opaque", True)):
                compiled = torch.compile(
                    Coordinates(steps, opaque),
                    backend=backend,
                    fullgraph=True,
                    dynamic=False,
                    options=options,
                )
                actual = compiled(like)
                a, b = actual.cpu(), original.cpu()
                differing = (
                    (a.view(torch.int32) != b.view(torch.int32)).nonzero().flatten().tolist()
                )
                row["variants"][name] = {
                    "exact": not differing,
                    "different_indices": differing,
                    "max_abs": float((a.double() - b.double()).abs().max()),
                    "values": a.tolist(),
                    "bits": a.view(torch.int32).tolist(),
                    "original_bits": b.view(torch.int32).tolist(),
                }
                tensors[f"{name}_{steps}"] = actual
            records.append(row)
    return records, tensors


def validate_cohort(records: Any) -> list[dict[str, Any]]:
    """Require the complete twelve-record diagnostic population before replay."""
    scales = {
        ("E_A5_MATCHED", "fine"),
        ("E_C2F_MATCHED", "fine"),
        ("E_C2F_MATCHED", "coarse"),
        ("R_A5", "fine"),
        ("R_C2F", "fine"),
        ("R_C2F", "coarse"),
    }
    expected = {(index, producer, scale) for index in (488, 505) for producer, scale in scales}
    if (
        not isinstance(records, list)
        or len(records) != 12
        or any(not isinstance(row, dict) or row.get("status") != "COMPARED" for row in records)
    ):
        raise ValueError("Expected all twelve completed source diagnostic records")
    identities = [(row.get("index"), row.get("producer"), row.get("scale")) for row in records]
    if len(set(identities)) != 12 or set(identities) != expected:
        raise ValueError("Source diagnostic population is missing, duplicated or unexpected")
    return records


def diagnostic_gate(records: Any, coordinates: Any) -> dict[str, bool]:
    """Require all saved-control and candidate coordinates, with no partial passes."""
    from e_jepa_ttc.models.local_transport import TRANSPORT_FEATURE_NAMES

    def exact(comparison: Any) -> bool:
        if not isinstance(comparison, dict) or comparison.get("exact") is not True:
            return False
        outputs = comparison.get("outputs", [])
        if len(outputs) != 2:
            return False
        for output, direction in zip(outputs, ("forward", "reverse"), strict=True):
            if output.get("direction") != direction or output.get("exact") is not True:
                return False
            features = output.get("coordinates", [])
            if len(features) != len(TRANSPORT_FEATURE_NAMES):
                return False
            for index, (feature, name) in enumerate(
                zip(features, TRANSPORT_FEATURE_NAMES, strict=True)
            ):
                if (
                    feature.get("index") != index
                    or feature.get("name") != name
                    or feature.get("exact") is not True
                    or feature.get("nonfinite_pairs") != 0
                    or feature.get("different_elements") != 0
                ):
                    return False
        return True

    try:
        rows = validate_cohort(records)
    except (TypeError, ValueError):
        rows = []
    controls = bool(rows) and all(
        exact(row.get("variants", {}).get("original"))
        and all(
            exact(row.get("control_reproduction", {}).get(name))
            for name in ("original", "native_weighted_mean")
        )
        for row in rows
    )
    candidate = bool(rows) and all(exact(row.get("variants", {}).get(CANDIDATE)) for row in rows)
    standalone = (
        isinstance(coordinates, list)
        and len(coordinates) == 2
        and all(
            row.get("steps") == steps
            and row.get("variants", {}).get("opaque", {}).get("exact") is True
            for row, steps in zip(coordinates, (16, 32), strict=True)
        )
    )
    return {
        "all_controls_reproduced_exactly": controls,
        "all_candidates_exact": candidate,
        "standalone_opaque_exact": standalone,
        "diagnostic_gate_passed": controls and candidate and standalone,
    }


def run(args: argparse.Namespace) -> None:
    """Verify both evidence generations and retain all negative numerical results."""
    from scripts.benchmark_train40_compiled_pipeline import compiler_cache_inventory
    from scripts.benchmark_train40_encoder import configure_compiler
    from scripts.diagnose_train40_weighted_mean import _sha

    source, tensor_source = (
        args.source.resolve(strict=True),
        args.tensor_source.resolve(strict=True),
    )
    output, cache = args.output.resolve(), args.compiler_cache.resolve()
    for protected in (source, tensor_source):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Output must be separate from both source evidence directories")
        if cache == protected or cache.is_relative_to(protected) or protected.is_relative_to(cache):
            raise ValueError("Compiler cache must not change source evidence")
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) for key, value in vars(args).items()}
    report: dict[str, Any] = {
        "status": "FAILED",
        "scope": "BOUND_TRANSPORT_SUMMARY_LINSPACE_ONLY_DIAGNOSTIC",
        "frozen_model_checkpoints_loaded": False,
        "normalizers_loaded": False,
        "raw_sensors_loaded": False,
        "final_ttc_admitted": False,
        "complete_update_admitted": False,
        "latency_measured": False,
        "config": config,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "start_time": datetime.now(UTC).isoformat(),
        "records": [],
    }
    snapshot: dict[str, Any] = {}
    caught: BaseException | None = None
    cache_before = compiler_cache_inventory(cache)

    def write() -> None:
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    try:
        report["compiler_environment"] = configure_compiler(args.compiler_root, cache)
        import torch

        from operational.train40_runtime.measurement import snapshot_files, verify_snapshot
        from scripts.benchmark_train40_producer_variants import _preserve_source_bytes
        from scripts.diagnose_train40_transport_summaries import (
            INPUT_NAMES,
            _copy_to_layout,
            _save_tensors,
            compare_transport_vectors,
        )
        from scripts.diagnose_train40_weighted_mean import (
            load_bound_tensor_archive,
            verify_scientific_sources,
        )

        source_path, source_binding = source / "summary.json", source / "original_snapshot.json"
        tensor_path, tensor_binding = (
            tensor_source / "summary.json",
            tensor_source / "original_snapshot.json",
        )
        previous = json.loads(source_path.read_text(encoding="utf-8"))
        bound = json.loads(source_binding.read_text(encoding="utf-8"))
        origin = json.loads(tensor_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") != "COMPLETE"
            or not previous.get("preservation", {}).get("passed")
            or not previous.get("causal_comparison_valid")
            or previous.get("scope")
            != "PRESERVED_PRE_ROUTER_TENSORS_NATIVE_WEIGHTED_MEAN_DIAGNOSTIC"
        ):
            raise ValueError("Weighted-mean source is not a complete preserved causal diagnostic")
        if (
            origin.get("status") != "COMPLETE"
            or not origin.get("preservation", {}).get("passed")
            or origin.get("scope")
            != "REAL_PRE_ROUTER_TRANSPORT_INPUTS_ONLY_NO_COMPLETE_UPDATE_CLOCK_OR_TTC"
        ):
            raise ValueError("Tensor source is not a complete preserved transport diagnostic")
        if previous["binding"].get("source_summary_sha256") != _sha(tensor_path) or previous[
            "binding"
        ].get("source_snapshot_sha256") != _sha(tensor_binding):
            raise ValueError("Tensor source differs from the weighted-mean evidence binding")
        records = validate_cohort(previous.get("records"))
        original_records = validate_cohort(origin.get("records"))
        original_by_id = {(r["index"], r["producer"], r["scale"]): r for r in original_records}
        scientific = {
            relative: ROOT / relative
            for relative in (
                "src/e_jepa_ttc/models/local_transport.py",
                "src/e_jepa_ttc/models/causal_scale_ttc.py",
                "operational/train40_runtime/compiled_transport.py",
                "operational/train40_runtime/compiled_pipeline.py",
                "scripts/diagnose_train40_transport_summaries.py",
                "scripts/diagnose_train40_weighted_mean.py",
            )
        }
        report["source_scientific_sha256"] = verify_scientific_sources(bound, scientific)
        protected_files = [source_path, source_binding, tensor_path, tensor_binding]
        for row in records:
            original_row = original_by_id[row["index"], row["producer"], row["scale"]]
            if any(
                row[key] != original_row[key]
                for key in (
                    "query_id",
                    "input_snapshot",
                    "output_snapshot",
                    "radius",
                    "temperature",
                    "deterministic_sampling",
                )
            ):
                raise ValueError(
                    "Record identity or tensor binding changed between source generations"
                )
            for directory, key in (
                (tensor_source, "input_snapshot"),
                (tensor_source, "output_snapshot"),
                (source, "new_output_snapshot"),
            ):
                load_bound_tensor_archive(directory, row[key])
                protected_files.append(directory / row[key]["path"])
        sources = sorted(
            {
                Path(__file__).resolve(),
                *scientific.values(),
                ROOT / "scripts/benchmark_train40_encoder.py",
                ROOT / "scripts/benchmark_train40_compiled_pipeline.py",
                ROOT / "scripts/benchmark_train40_producer_variants.py",
                ROOT / "operational/train40_runtime/measurement.py",
            }
        )
        snapshot = snapshot_files([*sources, *protected_files])
        (output / "original_snapshot.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )
        _preserve_source_bytes(sources, snapshot, output)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        options = {
            "use_fast_math": False,
            "force_same_precision": True,
            "triton.cudagraphs": False,
            "emulate_precision_casts": True,
            "eager_numerics.division_rounding": True,
        }
        if previous["binding"].get("compiler_options") != options:
            raise ValueError("Compiler options differ from the prior weighted-mean control")
        _, binding = make_native_linspace_physical()
        report["binding"] = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "seed": 7,
            "precision": "FP32_TF32_OFF",
            "compiler_options": options,
            "private_functions": binding,
            "source_summary_sha256": _sha(source_path),
            "expected_records": 12,
        }
        coordinate_records, coordinate_tensors = coordinate_probe(
            device="cuda", backend="inductor", options=options
        )
        report["standalone_coordinates"] = coordinate_records
        report["coordinate_snapshot"] = _save_tensors(
            output / "standalone_coordinates.pt", coordinate_tensors
        )
        probes = {}
        with torch.inference_mode():
            for row in records:
                item = {
                    key: row[key]
                    for key in (
                        "index",
                        "query_id",
                        "producer",
                        "scale",
                        "radius",
                        "temperature",
                        "deterministic_sampling",
                        "input_snapshot",
                        "new_output_snapshot",
                    )
                }
                item.update(status="FAILED", variants={}, control_reproduction={})
                report["records"].append(item)
                try:
                    cpu_inputs = load_bound_tensor_archive(tensor_source, row["input_snapshot"])
                    if set(cpu_inputs) != set(INPUT_NAMES):
                        raise ValueError("Expected exactly four frozen transport inputs")
                    inputs = tuple(
                        _copy_to_layout(cpu_inputs[name], "cuda") for name in INPUT_NAMES
                    )
                    archived = load_bound_tensor_archive(source, row["new_output_snapshot"])
                    initial = load_bound_tensor_archive(tensor_source, row["output_snapshot"])
                    expected = tuple(
                        initial[f"original_{direction}"] for direction in ("forward", "reverse")
                    )
                    key = (
                        row["radius"],
                        row["temperature"],
                        row["deterministic_sampling"],
                        tuple((tuple(value.shape), value.stride()) for value in inputs),
                    )
                    if key not in probes:
                        probes[key] = make_linspace_probe(
                            row["radius"],
                            row["temperature"],
                            deterministic_sampling=row["deterministic_sampling"],
                            backend="inductor",
                            options=options,
                        )
                    outputs = {}
                    for name, probe in probes[key].items():
                        actual = probe(*inputs)
                        torch.cuda.synchronize()
                        item["variants"][name] = compare_transport_vectors(actual, expected)
                        if name != CANDIDATE:
                            saved = tuple(
                                archived[f"{name}_{direction}"]
                                for direction in ("forward", "reverse")
                            )
                            item["control_reproduction"][name] = compare_transport_vectors(
                                actual, saved
                            )
                        outputs.update(
                            {
                                f"{name}_{direction}": value
                                for direction, value in zip(
                                    ("forward", "reverse"), actual, strict=True
                                )
                            }
                        )
                    stem = f"query{row['index']}_{row['producer']}_{row['scale']}"
                    item["output_snapshot"] = _save_tensors(output / f"{stem}_outputs.pt", outputs)
                    item["status"] = "COMPARED"
                except Exception as error:
                    item.update(
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
                    )
                write()
                print(
                    json.dumps(
                        {key: item[key] for key in ("index", "producer", "scale", "status")}
                    ),
                    flush=True,
                )
        report.update(diagnostic_gate(report["records"], coordinate_records))
        report["status"] = (
            "COMPLETE" if report["diagnostic_gate_passed"] else "COMPLETE_WITH_NUMERICAL_FAILURES"
        )
    except BaseException as error:
        caught = error
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if snapshot:
            report["preservation"] = verify_snapshot(snapshot)
            if not report["preservation"]["passed"]:
                report.update(status="FAILED", diagnostic_gate_passed=False)
        report["compiler_cache_before"] = cache_before
        report["compiler_cache_after"] = compiler_cache_inventory(cache)
        report["end_time"] = datetime.now(UTC).isoformat()
        write()
    if caught is not None:
        raise caught
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Require both generations of bound evidence and a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tensor-source", "compiler-root", "compiler-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
