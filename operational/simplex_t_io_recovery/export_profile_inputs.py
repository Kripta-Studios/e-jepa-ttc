"""Export target-free registered profiling inputs without repeating measurements."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import continue_campaign

if TYPE_CHECKING:
    from torch import Tensor, nn

    from e_jepa_ttc.simplex_t.model import TemporalRefiner
    from operational.simplex_t_cost_context.engine import Resources


def main() -> None:
    """Reuse the admitted profiler's gathers and endpoints; execute no forward/update."""
    authority = continue_campaign.policy()
    continue_campaign.apply_policy(authority)
    import numpy as np

    from operational.simplex_t_cost_context import analyze, engine, profile

    night = continue_campaign.NIGHT
    if engine._owner(night / "DRIVER.lock") or engine._owner(engine.EXEC / "WRITER.lock"):
        raise RuntimeError("profiling input export requires the previous heavy owner to exit")
    resource = engine.Resources()
    resource.reserve_artifacts(20_000_000, "TARGET_FREE_PROFILE_INPUT_AND_CONTROL_WEIGHT_EXPORT")
    index = profile.record(engine.EXEC / "ANALYSIS_EXPORT_INDEX.json")
    previous = profile.record(night / "profiling/FULL_ROUTE_ADMISSION.json")
    rows: dict[str, Any] = {}

    def capture(
        model: nn.Module,
        inputs: tuple[Tensor, ...],
        label: str,
        guard: Resources,
        endpoint_sha: str,
    ) -> dict:
        guard.check()
        receipt = profile.record(night / "profiling" / (label + ".json"))
        if receipt["status"] != "MEASURED" or receipt["endpoint_sha256"] != endpoint_sha:
            raise ValueError("only previously measured admitted endpoints can be exported")
        arrays = {
            key: tensor.detach().cpu().numpy()
            for key, tensor in zip(("features", "times", "valid", "experts"), inputs, strict=True)
        }
        path, input_sha = analyze.cached_inputs(arrays)
        state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        if label.startswith("H"):
            weight = night / "profiling/weights" / (label + ".npz")
            weight.parent.mkdir(exist_ok=True)
            weight_sha = analyze.immutable_npz(weight, state)
            constructor = dict(
                kind="historical_temporal", config=asdict(cast("TemporalRefiner", model).cfg)
            )
        else:
            row = index["fits"][f"COST_CONTEXT_20261003/{label}/fold0/seed7"]
            weight = engine.EXEC / row["weights_path"]
            weight_sha = row["weights_sha256"]
            if profile.digest(weight) != weight_sha:
                raise ValueError("published head weights changed")
            with np.load(weight, allow_pickle=False) as archive:
                if set(archive.files) != set(state) or any(
                    not np.array_equal(archive[key], value) for key, value in state.items()
                ):
                    raise ValueError("profile model differs from the published numerical weights")
            constructor = dict(kind="cost_context", arm=label)
        rows[label] = dict(
            constructor=constructor,
            weights_path=weight.relative_to(night).as_posix(),
            weights_sha256=weight_sha,
            inputs_path=path.relative_to(night).as_posix(),
            inputs_sha256=input_sha,
            endpoint_sha256=endpoint_sha,
            selection_sha256=receipt["selection_sha256"],
            role="TRAIN_FOLD0_FIXED_HASH_SELECTION",
            queries=64,
            target_fields_included=False,
        )
        profile.atomic_json(
            night / "PROFILE_INPUT_EXPORT.json",
            dict(
                status="PROFILE_INPUT_EXPORT_FRAGMENT_COMMITTED",
                models=rows,
                optimizer_updates=0,
                additional_measurements=0,
                raw_reconstruction=False,
            ),
        )
        return receipt

    # Only the callback changes: source gathers and numerical endpoints are the
    # authoritative profiler's. No timing, model forward or optimizer is executed.
    profile.measure = capture
    profile.full_route_admission = lambda _: previous
    sys.argv = ["registered_profile_input_export"]
    profile.main()
    if len(rows) != 9:
        raise ValueError("all nine admitted profiling models must have cached inputs")
    profile.atomic_json(
        night / "PROFILE_INPUT_EXPORT.json",
        dict(
            status="NINE_PREPARED_HEAD_PROFILE_INPUTS_EXPORTED",
            models=rows,
            optimizer_updates=0,
            additional_measurements=0,
            raw_reconstruction=False,
            source_sha256=profile.digest(Path(__file__)),
            canonical_profile_source_sha256=profile.digest(Path(profile.__file__)),
            profiler_design=profile.DESIGN,
            historical_cache_validity_may_depend_on_full_producer_availability=True,
        ),
    )


if __name__ == "__main__":
    main()
