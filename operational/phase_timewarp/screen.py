"""Compact fixed-clock PAIR replay; no sensor tensors or targets are needed."""

import numpy as np
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.features import event_phase17
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.phase_timewarp.canonical_pair import CanonicalPhaseDomainError


def replay_pair_batches(models: dict, calls: list, arrays: dict, device: str) -> dict:
    """Rebuild every PAIR-dependent PHASE17 column in the original full batch.

    Both modalities share the PHASE17 layout. Columns 0:8 retain sensor statistics
    and A5/C2F diagnostics; the shared builder recomputes experts and differences.
    Rejected original batches remain rejected, never recovered by selecting rows.
    """
    observations: dict[str, list] = {"event": [], "rgb": []}
    outputs, records = {}, []
    with torch.inference_mode():
        for call in calls:
            key = f"call{call['call_id']}"
            modality = "event" if call["prefix"] == "E" else "rgb"
            pair = "PAIR_E_MATCHED" if modality == "event" else "PAIR_R"
            record = {"call_id": call["call_id"], "status": call["status"]}
            if call["status"] != "accepted":
                record["reason"] = "original_producer_batch_rejected"
                records.append(record)
                continue
            features = arrays[f"{key}.{pair}.input133"]
            raw, experts = arrays[key + ".raw17"], arrays[key + ".expert3"]
            count = len(call["observations"])
            if (
                features.shape != (count, 133)
                or raw.shape != (count, 17)
                or experts.shape != (count, 3)
                or any(a.dtype != np.float32 for a in (features, raw, experts))
                or [i["batch_index"] for i in call["observations"]] != list(range(count))
                or not all(i["source_accepted"] for i in call["observations"])
            ):
                raise ValueError("Compact batch identity/shape/dtype mismatch")
            np.testing.assert_array_equal(features[:, 128], arrays[key + ".delta"][:, -1])
            np.testing.assert_array_equal(raw[:, 8:11], experts)
            model = models[pair]
            try:
                phase = model(PairFeatureBatch(torch.from_numpy(features).to(device)))
                expert = torch.from_numpy(experts).to(device).clone()
                expert[:, 2] = emitted_phase(phase)
                previous = torch.from_numpy(raw).to(device)
                rebuilt = event_phase17(previous[:, :2], previous[:, 2:5], previous[:, 5:8], expert)
            except CanonicalPhaseDomainError as error:
                record.update(status="rejected", reason=str(error), whole_batch=True)
            else:
                raw_out, expert_out = rebuilt.cpu().numpy(), expert.cpu().numpy()
                outputs[key + ".raw_phase"] = phase.cpu().numpy().copy()
                outputs[key + ".raw17"] = raw_out.copy()
                outputs[key + ".expert3"] = expert_out.copy()
                for i, item in enumerate(call["observations"]):
                    observations[modality].append(
                        {**item, "raw17": raw_out[i].copy(), "expert3": expert_out[i].copy()}
                    )
            if hasattr(model, "last_trace"):
                for field, tensor in model.last_trace.items():
                    outputs[key + ".trace." + field] = tensor.cpu().numpy().copy()
            records.append(record)
    return {"observations": observations, "arrays": outputs, "calls": records}
