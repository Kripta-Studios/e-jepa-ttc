"""Replay bound transport tensors with an opaque original weighted-mean boundary.

No model checkpoint, normalizer, RAW sensor or TTC head is loaded. Matching and
reverse sampling remain original ATen; the candidate also keeps the complete
weighted mean original. All other summary arithmetic is compiled. This isolates
the remaining summary drift and makes no latency or final-TTC admission claim.
"""

from __future__ import annotations

import argparse
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
_SAMPLER_OPERATOR: Any = None


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def native_weighted_mean_operator() -> Any:
    """Keep multiply, both sums, denominator clamp and division as original ATen."""
    import torch

    from e_jepa_ttc.models import local_transport as oracle

    @torch.library.custom_op(
        "train40_transport_diagnostics::native_weighted_mean",
        mutates_args=(),
        schema="(Tensor values, Tensor weights) -> Tensor",
    )
    def mean(values: Tensor, weights: Tensor) -> Tensor:
        if values.dtype != torch.float32 or weights.dtype != torch.float32:
            raise ValueError("Frozen weighted means require FP32 values and weights")
        if values.ndim != 3 or values.shape != weights.shape:
            raise ValueError("Frozen weighted means require matching [B,H,W] inputs")
        return oracle._weighted_mean(values, weights)

    @mean.register_fake
    def fake(values: Tensor, weights: Tensor) -> Tensor:
        if values.dtype != torch.float32 or weights.dtype != torch.float32:
            raise ValueError("Frozen weighted means require FP32 values and weights")
        if values.ndim != 3 or values.shape != weights.shape:
            raise ValueError("Frozen weighted means require matching [B,H,W] inputs")
        return values.new_empty((values.shape[0],))

    return mean


def _native_sample_bridge(
    reverse: Any, forward: Any, *, deterministic_sampling: bool = False
) -> Any:
    """Template cloned into the private registered module with a bound operator."""
    if _SAMPLER_OPERATOR is None:
        raise RuntimeError("The native sampler bridge requires its private bound module")
    return _SAMPLER_OPERATOR(
        forward.dx,
        forward.dy,
        forward.valid,
        reverse.dx,
        reverse.dy,
        reverse.valid,
        deterministic_sampling,
    )


@lru_cache(maxsize=1)
def make_native_mean_physical() -> tuple[Any, dict[str, Any]]:
    """Clone summary function globals in a registered module, leaving originals intact."""
    from e_jepa_ttc.models import local_transport as oracle
    from scripts.diagnose_train40_transport_summaries import native_reverse_sample_operator

    function_names = (
        "_weighted_mean",
        "_weighted_slope",
        "_summarise_one",
        "_sample_reverse_flow",
        "transport_physical_features",
    )
    functions = {name: getattr(oracle, name) for name in function_names}
    hashes = {
        name: hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()
        for name, function in functions.items()
    }
    hashes["native_sample_bridge"] = hashlib.sha256(
        textwrap.dedent(inspect.getsource(_native_sample_bridge)).encode()
    ).hexdigest()
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    module_name = f"_train40_native_weighted_mean_{identity[:16]}"
    private = types.ModuleType(module_name)
    private.__dict__.update(vars(oracle))
    private.__name__, private.__package__, private.__spec__ = module_name, "", None
    scope = vars(private)
    scope["_SAMPLER_OPERATOR"] = native_reverse_sample_operator()
    scope["_weighted_mean"] = native_weighted_mean_operator()

    def clone(function: FunctionType) -> FunctionType:
        result = FunctionType(
            function.__code__, scope, function.__name__, function.__defaults__, function.__closure__
        )
        result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
        result.__annotations__ = dict(function.__annotations__)
        return result

    scope["_sample_reverse_flow"] = clone(_native_sample_bridge)
    for name in ("_weighted_slope", "_summarise_one", "transport_physical_features"):
        function = functions[name]
        if not isinstance(function, FunctionType):
            raise TypeError(f"Scientific summary function is not a Python function: {name}")
        scope[name] = clone(function)
    # Dynamo resolves globals through __module__; retain this separate module
    # instead of installing replacement globals into the scientific module.
    sys.modules[module_name] = private
    return scope["transport_physical_features"], {
        "private_module": module_name,
        "function_source_sha256": hashes,
        "substitutions": {
            "mean": "ORIGINAL_ATEN_WEIGHTED_MEAN",
            "sampler": "ORIGINAL_ATEN_COMPLETE_REVERSE_SAMPLE",
        },
        "scientific_globals_modified": False,
    }


def make_weighted_mean_probe(
    radius: int,
    temperature: float,
    *,
    deterministic_sampling: bool,
    backend: str = "eager",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the saved diagnostic partition with one additional native boundary."""
    import torch
    from torch import nn

    from e_jepa_ttc.models import local_transport as oracle
    from scripts.diagnose_train40_transport_summaries import make_summary_probe

    physical, _ = make_native_mean_physical()

    class Summary(nn.Module):
        def forward(
            self,
            forward: Any,
            reverse: Any,
            foreground_previous: Tensor,
            foreground_current: Tensor,
        ) -> tuple[Tensor, Tensor]:
            return (
                physical(
                    forward,
                    reverse,
                    foreground_weight=foreground_previous,
                    radius=radius,
                    deterministic_sampling=deterministic_sampling,
                    reuse_reverse_sample=True,
                ),
                physical(
                    reverse,
                    forward,
                    foreground_weight=foreground_current,
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

    controls = make_summary_probe(
        radius,
        temperature,
        deterministic_sampling=deterministic_sampling,
        backend=backend,
        options=options,
    )
    return {
        "original": controls["original"],
        "compiled_summary_native_sampler": controls["compiled_summary_native_sampler"],
        "native_weighted_mean": candidate,
    }


def load_bound_tensor_archive(directory: Path, binding: dict[str, Any]) -> dict[str, Tensor]:
    """Verify archive bytes, tensor schema, strides and logical bytes before replay."""
    import torch

    name = binding.get("path")
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError("Tensor archive path must be a local basename")
    path = (directory / name).resolve(strict=True)
    if not path.is_relative_to(directory.resolve(strict=True)):
        raise ValueError("Tensor archive path escapes its evidence directory")
    if _sha(path) != binding.get("sha256") or path.stat().st_size != binding.get("bytes"):
        raise ValueError(f"Tensor archive hash/size differs: {name}")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    expected = binding.get("tensors")
    if (
        not isinstance(tensors, dict)
        or not isinstance(expected, dict)
        or set(tensors) != set(expected)
    ):
        raise ValueError("Tensor archive keys differ from their binding")
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Archive entries must contain only tensors")
        metadata = expected[key]
        if (
            list(value.shape) != metadata.get("shape")
            or list(value.stride()) != metadata.get("stride")
            or str(value.dtype) != metadata.get("dtype")
        ):
            raise ValueError(f"Tensor shape/stride/dtype differs: {key}")
        digest = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        if digest != metadata.get("logical_bytes_sha256"):
            raise ValueError(f"Tensor logical bytes differ: {key}")
    return tensors


def verify_scientific_sources(
    snapshot: dict[str, Any], current_sources: dict[str, Path]
) -> dict[str, str]:
    """Resolve bound source suffixes unambiguously and require identical current bytes."""
    result = {}
    for relative, current in current_sources.items():
        suffix = "/" + relative.replace("\\", "/")
        matches = [
            value for name, value in snapshot.items() if name.replace("\\", "/").endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"Scientific source binding is missing or ambiguous: {relative}")
        digest = _sha(current)
        if digest != matches[0].get("sha256"):
            raise ValueError(f"Scientific source bytes differ: {relative}")
        result[relative] = digest
    return result


def run(args: argparse.Namespace) -> None:
    """Replay preserved tensor evidence without loading the original model weights."""
    from scripts.benchmark_train40_compiled_pipeline import compiler_cache_inventory
    from scripts.benchmark_train40_encoder import configure_compiler

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("New output must be separate from source evidence")
    cache = args.compiler_cache.resolve()
    if cache == source or cache.is_relative_to(source):
        raise ValueError("Compiler cache must not change the source evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    args.compiler_cache.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    report: dict[str, Any] = {
        "status": "FAILED",
        "scope": "PRESERVED_PRE_ROUTER_TENSORS_NATIVE_WEIGHTED_MEAN_DIAGNOSTIC",
        "frozen_model_checkpoints_loaded": False,
        "normalizers_loaded": False,
        "raw_sensors_loaded": False,
        "final_ttc_admitted": False,
        "complete_update_admitted": False,
        "latency_measured": False,
        "config": configuration,
        "config_hash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "start_time": datetime.now(UTC).isoformat(),
        "records": [],
    }
    snapshot: dict[str, Any] = {}
    caught: BaseException | None = None
    cache_before = compiler_cache_inventory(args.compiler_cache)
    try:
        report["compiler_environment"] = configure_compiler(args.compiler_root, args.compiler_cache)
        import torch

        from operational.train40_runtime.measurement import snapshot_files
        from scripts.benchmark_train40_producer_variants import _preserve_source_bytes
        from scripts.benchmark_train40_runtime import _write
        from scripts.diagnose_train40_transport_summaries import (
            INPUT_NAMES,
            _copy_to_layout,
            _save_tensors,
            compare_transport_vectors,
        )

        source_summary_path = source / "summary.json"
        source_snapshot_path = source / "original_snapshot.json"
        summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        source_snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE" or not summary.get("preservation", {}).get("passed"):
            raise ValueError("Source diagnostic was not complete with preserved evidence")
        if (
            summary.get("scope")
            != "REAL_PRE_ROUTER_TRANSPORT_INPUTS_ONLY_NO_COMPLETE_UPDATE_CLOCK_OR_TTC"
        ):
            raise ValueError("Unexpected source evidence scope")
        scientific_sources = {
            relative: ROOT / relative
            for relative in (
                "src/e_jepa_ttc/models/local_transport.py",
                "src/e_jepa_ttc/models/causal_scale_ttc.py",
                "operational/train40_runtime/compiled_transport.py",
                "operational/train40_runtime/compiled_pipeline.py",
                "scripts/diagnose_train40_transport_summaries.py",
            )
        }
        report["source_scientific_sha256"] = verify_scientific_sources(
            source_snapshot, scientific_sources
        )
        records = summary.get("records")
        if (
            not isinstance(records, list)
            or not records
            or any(item.get("status") != "COMPARED" for item in records)
        ):
            raise ValueError("Source records must be nonempty completed tensor comparisons")
        identities = [(item["index"], item["producer"], item["scale"]) for item in records]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate source tensor identities")
        protected = [source_summary_path, source_snapshot_path]
        for record in records:
            for key in ("input_snapshot", "output_snapshot"):
                # Verify all files before CUDA compilation, retaining only one
                # archive at a time. The actual replay rechecks the same binding.
                validated = load_bound_tensor_archive(source, record[key])
                del validated
                protected.append(source / record[key]["path"])
        sources = [
            Path(__file__),
            *scientific_sources.values(),
            ROOT / "scripts/benchmark_train40_compiled_pipeline.py",
            ROOT / "scripts/benchmark_train40_encoder.py",
            ROOT / "scripts/benchmark_train40_producer_variants.py",
            ROOT / "scripts/benchmark_train40_runtime.py",
            ROOT / "operational/train40_runtime/measurement.py",
        ]
        sources = sorted({path.resolve() for path in sources})
        snapshot = snapshot_files([*sources, *protected])
        _write(args.output / "original_snapshot.json", snapshot)
        _preserve_source_bytes(sources, snapshot, args.output)
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
        _, function_binding = make_native_mean_physical()
        report["binding"] = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "seed": 7,
            "precision": "FP32_TF32_OFF",
            "source_summary_sha256": _sha(source_summary_path),
            "source_snapshot_sha256": _sha(source_snapshot_path),
            "expected_records": len(records),
            "private_functions": function_binding,
            "compiler_options": options,
        }
        probes_by_configuration: dict[tuple[Any, ...], dict[str, Any]] = {}
        with torch.inference_mode():
            for original in records:
                item: dict[str, Any] = {
                    key: original[key]
                    for key in (
                        "index",
                        "query_id",
                        "producer",
                        "scale",
                        "radius",
                        "temperature",
                        "deterministic_sampling",
                        "capture_binding",
                        "input_snapshot",
                        "output_snapshot",
                    )
                }
                item.update(status="FAILED", variants={})
                report["records"].append(item)
                try:
                    cpu_inputs = load_bound_tensor_archive(source, original["input_snapshot"])
                    if set(cpu_inputs) != set(INPUT_NAMES):
                        raise ValueError("Expected exactly the four captured transport inputs")
                    inputs = tuple(
                        _copy_to_layout(cpu_inputs[name], "cuda") for name in INPUT_NAMES
                    )
                    archived = load_bound_tensor_archive(source, original["output_snapshot"])
                    expected = tuple(
                        archived[f"original_{direction}"] for direction in ("forward", "reverse")
                    )
                    key = (
                        original["radius"],
                        original["temperature"],
                        original["deterministic_sampling"],
                        tuple((tuple(value.shape), value.stride()) for value in inputs),
                    )
                    if key not in probes_by_configuration:
                        probes_by_configuration[key] = make_weighted_mean_probe(
                            original["radius"],
                            original["temperature"],
                            deterministic_sampling=original["deterministic_sampling"],
                            backend="inductor",
                            options=options,
                        )
                    probes = probes_by_configuration[key]
                    output_values: dict[str, Tensor] = {}
                    for name, probe in probes.items():
                        try:
                            actual = probe(*inputs)
                            torch.cuda.synchronize()
                            comparison = compare_transport_vectors(actual, expected)
                            item["variants"][name] = {"status": "COMPARED", **comparison}
                            output_values.update(
                                {
                                    f"{name}_{direction}": value
                                    for direction, value in zip(
                                        ("forward", "reverse"), actual, strict=True
                                    )
                                }
                            )
                            if name == "original" and not comparison["exact"]:
                                raise ValueError(
                                    "Original transport did not reproduce archived outputs exactly"
                                )
                            if name == "compiled_summary_native_sampler":
                                previous = tuple(
                                    archived[f"{name}_{direction}"]
                                    for direction in ("forward", "reverse")
                                )
                                item["control_reproduction"] = compare_transport_vectors(
                                    actual, previous
                                )
                        except Exception as error:
                            item["variants"][name] = {
                                "status": "FAILED",
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(),
                            }
                            if name == "original":
                                break
                        _write(args.output / "summary.json", report)
                    stem = f"query{item['index']}_{item['producer']}_{item['scale']}"
                    item["new_output_snapshot"] = _save_tensors(
                        args.output / f"{stem}_outputs.pt", output_values
                    )
                    item["status"] = (
                        "COMPARED"
                        if len(item["variants"]) == 3
                        and all(
                            value["status"] == "COMPARED" for value in item["variants"].values()
                        )
                        else "FAILED"
                    )
                    del inputs, cpu_inputs, archived, expected, output_values
                except Exception as error:
                    item.update(
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
                    )
                _write(args.output / "summary.json", report)
                print(
                    json.dumps(
                        {key: item[key] for key in ("index", "producer", "scale", "status")}
                    ),
                    flush=True,
                )
        report["compiled_probe_configurations"] = len(probes_by_configuration)
        report["all_native_mean_exact"] = all(
            item["status"] == "COMPARED" and item["variants"]["native_weighted_mean"]["exact"]
            for item in report["records"]
        )
        report["all_native_mean_passed_1e5"] = all(
            item["status"] == "COMPARED" and item["variants"]["native_weighted_mean"]["passed_1e5"]
            for item in report["records"]
        )
        report["all_controls_reproduced_exactly"] = all(
            item.get("control_reproduction", {}).get("exact") for item in report["records"]
        )
        report["status"] = (
            "COMPLETE"
            if len(report["records"]) == len(records)
            and all(item["status"] == "COMPARED" for item in report["records"])
            else "FAILED"
        )
        report["causal_comparison_valid"] = (
            report["status"] == "COMPLETE" and report["all_controls_reproduced_exactly"]
        )
        report["interpretation"] = (
            "COMPLETE records execution, including negative numerical results; "
            "no final TTC or complete-update admission"
        )
    except BaseException as error:
        caught = error
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if snapshot:
            from operational.train40_runtime.measurement import verify_snapshot

            report["preservation"] = verify_snapshot(snapshot)
            if not report["preservation"]["passed"]:
                report["status"] = "FAILED"
                report["causal_comparison_valid"] = False
        report["compiler_cache_before"] = cache_before
        report["compiler_cache_after"] = compiler_cache_inventory(args.compiler_cache)
        report["end_time"] = datetime.now(UTC).isoformat()
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    if caught is not None:
        raise caught
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Select bound tensor evidence and isolated compiler paths; never overwrite output."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "compiler-root", "compiler-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
