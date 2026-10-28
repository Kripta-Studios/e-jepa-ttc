"""Attribute frozen TRAIN40 transport drift before the fine/coarse router.

Real endpoint features are captured once per producer with original encoder
batching. Four tensor-core counterfactuals isolate matching, summaries and reverse
sampling. This diagnostic has no complete-update clock and admits no final TTC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
INPUT_NAMES = ("previous", "current", "foreground_previous", "foreground_current")


def _copy_to_layout(value: Tensor, device: Any) -> Tensor:
    import torch

    span = (
        1
        + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride(), strict=True))
        if value.numel()
        else 0
    )
    # Initialize the entire storage, including any stride gaps. Saved snapshots
    # must not contain uninitialized padding from unrelated allocator contents.
    storage = torch.zeros(span, dtype=value.dtype, device=device)
    return storage.as_strided(value.shape, value.stride()).copy_(value.detach())


def clone_preserving_strides(value: Tensor) -> Tensor:
    """Own tensor bytes without changing the original layout or numerical values."""
    return _copy_to_layout(value, value.device)


@lru_cache(maxsize=1)
def native_reverse_sample_operator() -> Any:
    """Register the complete ATen sampler, including grid and validity arithmetic."""
    import torch

    from e_jepa_ttc.models import local_transport as oracle

    @torch.library.custom_op(
        "train40_transport_diagnostics::native_reverse_sample",
        mutates_args=(),
        schema=(
            "(Tensor forward_dx, Tensor forward_dy, Tensor forward_valid, "
            "Tensor reverse_dx, Tensor reverse_dy, Tensor reverse_valid, "
            "bool deterministic_sampling) -> (Tensor, Tensor, Tensor)"
        ),
    )
    def sample(
        forward_dx: Tensor,
        forward_dy: Tensor,
        forward_valid: Tensor,
        reverse_dx: Tensor,
        reverse_dy: Tensor,
        reverse_valid: Tensor,
        deterministic_sampling: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        forward = oracle.LocalTransportMatch(
            forward_dx, forward_dy, forward_dx, forward_dx, forward_valid
        )
        reverse = oracle.LocalTransportMatch(
            reverse_dx, reverse_dy, reverse_dx, reverse_dx, reverse_valid
        )
        values = oracle._sample_reverse_flow(
            reverse, forward, deterministic_sampling=deterministic_sampling
        )
        # The original dx/dy are views into one sampled tensor. Custom-op outputs
        # must not alias each other; these copies preserve their values exactly.
        return values[0].clone(), values[1].clone(), values[2].clone()

    @sample.register_fake
    def fake(
        forward_dx: Tensor,
        forward_dy: Tensor,
        forward_valid: Tensor,
        reverse_dx: Tensor,
        reverse_dy: Tensor,
        reverse_valid: Tensor,
        deterministic_sampling: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del forward_dy, forward_valid, reverse_dx, reverse_dy, reverse_valid
        del deterministic_sampling
        return (
            forward_dx.new_empty(forward_dx.shape),
            forward_dx.new_empty(forward_dx.shape),
            forward_dx.new_empty(forward_dx.shape, dtype=torch.bool),
        )

    return sample


def make_summary_probe(
    radius: int,
    temperature: float,
    *,
    deterministic_sampling: bool,
    backend: str = "eager",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build four explicit partitions without patching scientific functions.

    Only the named remainder is compiled in each counterfactual. All variants
    receive the same endpoint/foreground tensors. Summaries use the original
    ``_summarise_one`` with an explicit reverse sample, before any C2F router.
    """
    import torch
    from torch import nn

    from e_jepa_ttc.models import local_transport as oracle
    from operational.train40_runtime.compiled_pipeline import NativeMatchTransportPairTensorCore
    from operational.train40_runtime.compiled_transport import TransportPairTensorCore

    native_sampler = native_reverse_sample_operator()

    def matches(previous: Tensor, current: Tensor) -> tuple[Any, Any]:
        return (
            oracle.local_correlation_match(
                previous, current, radius=radius, temperature=temperature
            ),
            oracle.local_correlation_match(
                current, previous, radius=radius, temperature=temperature
            ),
        )

    def physical(forward: Any, reverse: Any, foreground: Tensor, sample: Any) -> Tensor:
        # Same glue as transport_physical_features; the original summaries are
        # called unchanged. Only the supplied reverse_sample crosses a boundary.
        batch, height, width = forward.dx.shape
        global_weight = torch.ones(
            (batch, height, width), device=forward.dx.device, dtype=forward.dx.dtype
        )
        if foreground.ndim == 4 and foreground.shape[1] == 1:
            foreground = foreground[:, 0]
        if foreground.shape != (batch, height, width):
            raise ValueError("foreground_weight must have shape [B,H,W] or [B,1,H,W]")
        values = [
            oracle._summarise_one(
                forward,
                reverse,
                spatial_weight=weight,
                radius=radius,
                deterministic_sampling=deterministic_sampling,
                reverse_sample=sample,
            )
            for weight in (global_weight, foreground)
        ]
        return torch.cat(values, dim=-1)

    class Sampler(nn.Module):
        def forward(self, forward: Any, reverse: Any) -> Any:
            return oracle._sample_reverse_flow(
                reverse, forward, deterministic_sampling=deterministic_sampling
            )

    class SummaryWithNativeSampler(nn.Module):
        def forward(
            self,
            forward: Any,
            reverse: Any,
            foreground_previous: Tensor,
            foreground_current: Tensor,
        ) -> tuple[Tensor, Tensor]:
            def sample(a: Any, b: Any) -> Any:
                return native_sampler(
                    a.dx, a.dy, a.valid, b.dx, b.dy, b.valid, deterministic_sampling
                )

            return (
                physical(forward, reverse, foreground_previous, sample(forward, reverse)),
                physical(reverse, forward, foreground_current, sample(reverse, forward)),
            )

    def compile_module(module: nn.Module) -> Any:
        return torch.compile(
            module.eval(), backend=backend, fullgraph=True, dynamic=False, options=options
        )

    original = TransportPairTensorCore(
        radius, temperature, deterministic_sampling=deterministic_sampling, reuse_samples=False
    ).eval()
    production_core = TransportPairTensorCore(
        radius, temperature, deterministic_sampling=deterministic_sampling, reuse_samples=True
    ).eval()
    native_match = compile_module(NativeMatchTransportPairTensorCore(production_core).eval())
    compiled_sampler = compile_module(Sampler())
    compiled_summary = compile_module(SummaryWithNativeSampler())

    def summary_variant(
        previous: Tensor, current: Tensor, foreground_previous: Tensor, foreground_current: Tensor
    ) -> tuple[Tensor, Tensor]:
        forward, reverse = matches(previous, current)
        return compiled_summary(forward, reverse, foreground_previous, foreground_current)

    def sampler_variant(
        previous: Tensor, current: Tensor, foreground_previous: Tensor, foreground_current: Tensor
    ) -> tuple[Tensor, Tensor]:
        forward, reverse = matches(previous, current)
        return (
            physical(forward, reverse, foreground_previous, compiled_sampler(forward, reverse)),
            physical(reverse, forward, foreground_current, compiled_sampler(reverse, forward)),
        )

    return {
        "original": original,
        "native_match": native_match,
        "compiled_summary_native_sampler": summary_variant,
        "native_summary_compiled_sampler": sampler_variant,
    }


def capture_transport_inputs(
    model: CausalScaleTTC, sensor: Tensor, delta: Tensor
) -> list[dict[str, Any]]:
    """Capture at most fine/coarse inputs using private wrappers on a model view."""
    from torch import nn

    from operational.train40_runtime.compiled_transport import (
        TransportCoreBinding,
        attach_transport_pair_core,
    )

    records: list[dict[str, Any]] = []
    view = attach_transport_pair_core(model)
    binding = view._train40_transport_binding
    if not isinstance(binding, TransportCoreBinding):
        raise TypeError("Capture requires an audited frozen transport binding")
    capture_binding = asdict(binding)

    class Recorder(nn.Module):
        def __init__(self, scale: str, core: Any) -> None:
            super().__init__()
            self.scale, self.core = scale, core

        def forward(self, *values: Tensor) -> tuple[Tensor, Tensor]:
            if any(item["scale"] == self.scale for item in records) or len(records) >= 2:
                raise RuntimeError("A producer capture must call each fine/coarse core once")
            records.append(
                {
                    "scale": self.scale,
                    "radius": self.core.radius,
                    "temperature": self.core.temperature,
                    "deterministic_sampling": self.core.deterministic_sampling,
                    "capture_binding": capture_binding,
                    "inputs": tuple(clone_preserving_strides(value) for value in values),
                }
            )
            return self.core(*values)

    for scale in ("fine", "coarse"):
        key = f"_train40_transport_{scale}"
        view._modules[key] = Recorder(scale, view._modules[key])
    try:
        view(sensor, delta, return_dense_features=False)
    finally:
        # The private AST adapter stores a bound method on its own view. Break
        # that ownership cycle so completed captures do not wait for cyclic GC.
        object.__delattr__(view, "_forward_impl")
    expected = ["fine", "coarse"] if model.config.transport_mode == "adaptive_pyramid" else ["fine"]
    if [record["scale"] for record in records] != expected:
        raise RuntimeError("Transport capture did not cover the expected scales in order")
    return records


def compare_transport_vectors(actual: Any, expected: Any) -> dict[str, Any]:
    """Compare every pre-router coordinate, retaining negative and nonfinite cases."""
    import torch

    from e_jepa_ttc.models.local_transport import TRANSPORT_FEATURE_NAMES

    outputs = []
    if len(actual) != 2 or len(expected) != 2:
        raise ValueError("Expected forward and reverse vectors")
    for direction, a, b in zip(("forward", "reverse"), actual, expected, strict=True):
        if a.shape != b.shape or a.ndim != 2 or a.shape[-1] != 18:
            outputs.append(
                {"direction": direction, "passed_1e5": False, "exact": False, "reason": "shape"}
            )
            continue
        left, right = a.detach().double().cpu(), b.detach().double().cpu()
        coordinates = []
        for index, name in enumerate(TRANSPORT_FEATURE_NAMES):
            lhs, rhs = left[:, index], right[:, index]
            finite = torch.isfinite(lhs) & torch.isfinite(rhs)
            error = (lhs - rhs).abs()
            allowed = 1e-5 + 1e-5 * rhs.abs()
            valid = finite & (error <= allowed)
            maximum = int(error.masked_fill(~finite, -1).argmax()) if len(error) else None
            coordinates.append(
                {
                    "name": name,
                    "index": index,
                    "passed_1e5": bool(valid.all()) and bool(len(valid)),
                    "exact": bool(finite.all()) and torch.equal(lhs, rhs),
                    "different_elements": int((lhs != rhs).sum()),
                    "outside_tolerance": int((~valid).sum()),
                    "nonfinite_pairs": int((~finite).sum()),
                    "max_abs": float(error[finite].max()) if bool(finite.any()) else None,
                    "max_abs_pair_index": maximum,
                    "signed_error_at_max": float((lhs - rhs)[maximum])
                    if maximum is not None and bool(finite[maximum])
                    else None,
                    "depends_on_reverse_sampler": index in (8, 17),
                }
            )
        outputs.append(
            {
                "direction": direction,
                "passed_1e5": all(item["passed_1e5"] for item in coordinates),
                "exact": all(item["exact"] for item in coordinates),
                "coordinates": coordinates,
            }
        )
    return {
        "passed_1e5": all(item["passed_1e5"] for item in outputs),
        "exact": all(item["exact"] for item in outputs),
        "outputs": outputs,
        "gate_role": "DIAGNOSTIC_ONLY_NOT_FINAL_TTC_ADMISSION",
        "atol": 1e-5,
        "rtol": 1e-5,
    }


def _save_tensors(path: Path, values: dict[str, Tensor]) -> dict[str, Any]:
    import torch

    tensors, binding = {}, {}
    for name, value in values.items():
        cpu = _copy_to_layout(value, "cpu")
        tensors[name] = cpu
        binding[name] = {
            "shape": list(cpu.shape),
            "stride": list(cpu.stride()),
            "dtype": str(cpu.dtype),
            "logical_bytes_sha256": hashlib.sha256(cpu.contiguous().numpy().tobytes()).hexdigest(),
        }
    torch.save(tensors, path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.name, "sha256": digest, "bytes": path.stat().st_size, "tensors": binding}


def run(args: argparse.Namespace) -> None:
    """Bind real sources and frozen weights; save complete and failed diagnostics."""
    from scripts.benchmark_train40_compiled_pipeline import (
        compiler_cache_inventory,
        snapshot_roi_helpers,
    )
    from scripts.benchmark_train40_encoder import configure_compiler

    for protected in (args.candidate, args.reference, args.manifest.parent):
        parent = protected.resolve()
        for path in (args.output, args.compiler_cache):
            proposed = path.resolve()
            if (
                proposed == parent
                or proposed.is_relative_to(parent)
                or parent.is_relative_to(proposed)
            ):
                raise ValueError("Output/cache must be separate from frozen inputs")
    if not args.indices or len(set(args.indices)) != len(args.indices):
        raise ValueError("Select nonempty distinct query indices")
    args.output.mkdir(parents=True, exist_ok=False)
    args.compiler_cache.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    summary: dict[str, Any] = {
        "status": "FAILED",
        "scope": "REAL_PRE_ROUTER_TRANSPORT_INPUTS_ONLY_NO_COMPLETE_UPDATE_CLOCK_OR_TTC",
        "complete_update_admitted": False,
        "final_ttc_admitted": False,
        "experiment_id": args.output.name,
        "config": configuration,
        "config_hash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "start_time": datetime.now(UTC).isoformat(),
        "records": [],
    }
    original: dict[str, Any] = {}
    captured_error: BaseException | None = None
    cache_before = compiler_cache_inventory(args.compiler_cache)
    try:
        summary["compiler_environment"] = configure_compiler(
            args.compiler_root, args.compiler_cache
        )
        import numpy as np
        import torch

        from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
        from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
        from operational.rgb_port import transfer as engine
        from operational.train40_runtime.inputs import PersistentSensorInputs
        from operational.train40_runtime.measurement import snapshot_files
        from operational.train40_runtime.producers import (
            overlapping_endpoints,
            reuse_endpoint_encoders,
        )
        from scripts.benchmark_train40_producer_variants import (
            _preserve_source_bytes,
            _weight_values,
        )
        from scripts.benchmark_train40_runtime import (
            _candidate_inventory,
            _source_inventory,
            _validate_reference,
            _write,
        )
        from scripts.final_train40_transfer import (
            load_models,
            read,
            reject_targets,
            verify_candidate,
            verify_frozen_runtime,
        )

        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        if any(not 0 <= index < len(rows) for index in args.indices):
            raise ValueError("Query index outside frozen manifest")
        _, reference_files, reference_binding = _validate_reference(
            args.reference, args.manifest, rows, binding
        )
        helpers, helper_path, helper_binding = snapshot_roi_helpers(args.output)
        summary["roi_helper_snapshot"] = helper_binding
        runtime_files = {
            "compiled_pipeline.py",
            "compiled_transport.py",
            "transport.py",
            "producers.py",
            "measurement.py",
            "inputs.py",
        }
        sources = [
            path
            for path in _source_inventory(frozen, reference_binding)
            if path.parent.name != "train40_runtime" or path.name in runtime_files
        ]
        sources.extend(
            [
                Path(__file__),
                ROOT / "scripts/benchmark_train40_compiled_pipeline.py",
                ROOT / "scripts/benchmark_train40_encoder.py",
                ROOT / "scripts/benchmark_train40_producer_variants.py",
                helper_path,
            ]
        )
        sources = sorted(set(path.resolve() for path in sources))
        original = snapshot_files(
            [*_candidate_inventory(args.candidate), args.manifest, *reference_files, *sources]
        )
        _write(args.output / "original_snapshot.json", original)
        _preserve_source_bytes(sources, original, args.output)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        summary["binding"] = {
            **binding,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "seed": 7,
            "precision": "FP32_TF32_OFF",
            "objects": 1,
            "event_endpoint_reuse": False,
            "rgb_endpoint_reuse": True,
            "rgb_object_chunk_size": 1,
            "dataset_manifest_hash": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "capture": "PRIVATE_MODEL_VIEW_CORE_WRAPPERS; ORIGINAL_MODULES_UNCHANGED",
            "tensor_layout": "ORIGINAL_STRIDES_AND_VALUES_PRESERVED",
            "original_reverse_sample_reuse": False,
            "native_match_production_reverse_sample_reuse": True,
        }
        options = {
            "use_fast_math": False,
            "force_same_precision": True,
            "triton.cudagraphs": False,
            "emulate_precision_casts": True,
            "eager_numerics.division_rounding": True,
        }
        summary["compiler_options"] = options
        models, _ = load_models(paths, "cuda")
        weights_before = _weight_values(models)
        with torch.inference_mode(), PersistentSensorInputs() as sensors:
            for index in args.indices:
                row = rows[index]
                boxes = [sensors._box(source) for source in list(row["bbox_sources"])[-2:]]
                square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
                region = helpers.artificial_regions(row["square_xyxy"], square, 1)[0]
                prepared = engine.prepare_event({**row, "square_xyxy": list(region.event_square)})
                event = torch.from_numpy(prepared["own_events"]).cuda()
                event_delta = torch.full((len(event), 2), 0.1, device="cuda")
                raw_rgb = helpers.read_raw_rgb_history(sensors, row)
                rgb = helpers.reference_rgb(raw_rgb, region)
                if not rgb:
                    raise ValueError(
                        f"Query {index} lacks RGB; four-producer diagnosis requires it"
                    )
                rgb_sensor = torch.from_numpy(np.stack([pixels for pixels, _ in rgb])).cuda()
                rgb_delta = torch.tensor(
                    [metadata["delta_t_s"] for _, metadata in rgb],
                    dtype=torch.float32,
                    device="cuda",
                )
                if not overlapping_endpoints(rgb_sensor, len(rgb_sensor)):
                    raise ValueError("Real RGB histories do not satisfy endpoint reuse")
                for name in ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"):
                    model = models[name]
                    if not isinstance(model, CausalScaleTTC):
                        raise TypeError(
                            "Summary diagnostics require frozen CausalScaleTTC producers"
                        )
                    sensor, delta = (
                        (event, event_delta) if name.startswith("E_") else (rgb_sensor, rgb_delta)
                    )
                    if name.startswith("R_") and len(sensor) > 1:
                        model = reuse_endpoint_encoders(
                            model, len(sensor), int(sensor.shape[1]), history_group_size=len(sensor)
                        )
                    captures = capture_transport_inputs(model, sensor, delta)
                    for captured in captures:
                        record: dict[str, Any] = {
                            "index": index,
                            "query_id": row["query_id"],
                            "producer": name,
                            "status": "FAILED",
                            "scale": captured["scale"],
                            "radius": captured["radius"],
                            "temperature": captured["temperature"],
                            "deterministic_sampling": captured["deterministic_sampling"],
                            "capture_binding": captured["capture_binding"],
                            "producer_input_shape": list(sensor.shape),
                            "variants": {},
                        }
                        summary["records"].append(record)
                        stem = f"query{index}_{name}_{captured['scale']}"
                        values = captured["inputs"]
                        record["input_snapshot"] = _save_tensors(
                            args.output / f"{stem}_inputs.pt",
                            dict(zip(INPUT_NAMES, values, strict=True)),
                        )
                        try:
                            probes = make_summary_probe(
                                captured["radius"],
                                captured["temperature"],
                                deterministic_sampling=captured["deterministic_sampling"],
                                backend="inductor",
                                options=options,
                            )
                            expected = tuple(
                                clone_preserving_strides(value)
                                for value in probes["original"](*values)
                            )
                            output_tensors = {
                                f"original_{direction}": value
                                for direction, value in zip(
                                    ("forward", "reverse"), expected, strict=True
                                )
                            }
                            for variant, probe in probes.items():
                                try:
                                    actual = tuple(
                                        clone_preserving_strides(value) for value in probe(*values)
                                    )
                                    torch.cuda.synchronize()
                                    record["variants"][variant] = {
                                        "status": "COMPARED",
                                        **compare_transport_vectors(actual, expected),
                                    }
                                    output_tensors.update(
                                        {
                                            f"{variant}_{direction}": value
                                            for direction, value in zip(
                                                ("forward", "reverse"), actual, strict=True
                                            )
                                        }
                                    )
                                except Exception as error:
                                    record["variants"][variant] = {
                                        "status": "FAILED",
                                        "error": f"{type(error).__name__}: {error}",
                                        "traceback": traceback.format_exc(),
                                    }
                                _write(args.output / "summary.json", summary)
                            record["output_snapshot"] = _save_tensors(
                                args.output / f"{stem}_outputs.pt", output_tensors
                            )
                            record["status"] = (
                                "COMPARED"
                                if all(
                                    result["status"] == "COMPARED"
                                    for result in record["variants"].values()
                                )
                                else "FAILED"
                            )
                            del probes, expected, output_tensors
                        except Exception as error:
                            record.update(
                                error=f"{type(error).__name__}: {error}",
                                traceback=traceback.format_exc(),
                            )
                        _write(args.output / "summary.json", summary)
                        print(
                            json.dumps(
                                {
                                    key: record[key]
                                    for key in ("index", "producer", "scale", "status")
                                }
                            ),
                            flush=True,
                        )
                    del captured, values, captures
        summary["weights_unchanged"] = weights_before == _weight_values(models)
        summary["all_diagnostic_comparisons_passed_1e5"] = all(
            record["status"] == "COMPARED"
            and all(result["passed_1e5"] for result in record["variants"].values())
            for record in summary["records"]
        )
        summary["all_diagnostic_comparisons_exact"] = all(
            record["status"] == "COMPARED"
            and all(result["exact"] for result in record["variants"].values())
            for record in summary["records"]
        )
        summary["status"] = (
            "COMPLETE"
            if summary["weights_unchanged"]
            and all(record["status"] == "COMPARED" for record in summary["records"])
            else "FAILED"
        )
        summary["interpretation"] = (
            "COMPLETE means all diagnostics ran, including negative numerical results; "
            "no runtime admission. Coordinates 8/17 alone consume the reverse sampler "
            "before the router; downstream mixed coordinates cannot isolate this cause."
        )
    except BaseException as error:
        captured_error = error
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if original:
            from operational.train40_runtime.measurement import verify_snapshot

            summary["preservation"] = verify_snapshot(original)
            if not summary["preservation"]["passed"]:
                summary["status"] = "FAILED"
        summary["compiler_cache_before"] = cache_before
        summary["compiler_cache_after"] = compiler_cache_inventory(args.compiler_cache)
        summary["end_time"] = datetime.now(UTC).isoformat()
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    if captured_error is not None:
        raise captured_error
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Require frozen paths, an existing compiler, and a fresh evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "manifest", "reference", "compiler-root", "compiler-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, default=[488, 505])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
