"""Extract actual frozen refiner states; H supervision remains explicitly in-sample."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import RUN, read, sha, write
from operational.rgb_port.train_heads import load_head_endpoint

from .common import OUT, guard, setup, source_hashes
from .model import RGBAnchor, frozen_state


def extract(role: str) -> None:
    if role not in {"H", "V"}:
        raise ValueError("Only H/V roles admitted")
    guard()
    target = OUT / "cache" / role
    target.mkdir(parents=True, exist_ok=True)
    inputs = {}
    source = {}
    for modality in ("E", "R"):
        path = RUN / f"features/{role}/{modality}_HISTORY.npz"
        expected = read(path.with_suffix(".json"))
        assert expected["role"] == role and sha(path) == expected["output_sha256"]
        source[modality] = dict(path=str(path), sha256=sha(path))
        with np.load(path, allow_pickle=False) as z:
            inputs[modality] = {k: z[k] for k in z.files}
    e, r = inputs["E"], inputs["R"]
    for name in ("sample_token", "group_id", "target_phase", "mass", "role"):
        assert np.array_equal(e[name], r[name]), name
    assert set(map(str, e["role"])) == {role}
    n = len(e["sample_token"])
    plan = dict(
        role=role,
        population=n,
        inputs=source,
        sources=source_hashes(),
        parents={
            name: sha(RUN / f"fits/{name}/checkpoint_last.pt")
            for name in ("E_CTX_MATCHED", "R_CTX")
        },
        feature_schema="[R160,rawR,phaseR,expertsR3,timeR4,countR/8; E_same]",
        states_are_OOF=False,
        new_supervised_stage_on_H=True,
        device="cuda_FP32_noTF32",
    )
    if (target / "BINDING.json").exists():
        assert read(target / "BINDING.json") == plan
    write(target / "BINDING.json", plan)
    if (target / "SEALED.json").exists():
        seal = read(target / "SEALED.json")
        assert sha(target / "states.npz") == seal["states_sha256"]
        return
    with historical_source_binding():
        models = {
            m: load_head_endpoint(RUN / f"fits/{name}", name, torch.device("cuda"))
            .eval()
            .requires_grad_(False)
            for m, name in (("E", "E_CTX_MATCHED"), ("R", "R_CTX"))
        }
    anchor = RGBAnchor("EVENT").cuda().eval()
    parts = {k: [] for k in ("features", "rgb_width_logits", "rgb_phase", "event_phase")}
    for start in range(0, n, 128):
        if start % 4096 == 0:
            guard()
        stop = min(start + 128, n)
        blocks, outputs, logits = {}, {}, {}
        for m in ("E", "R"):
            data = inputs[m]
            args = [
                torch.from_numpy(data[k][start:stop]).cuda()
                for k in ("features", "timing", "valid", "expert_phase")
            ]
            blocks[m], logits[m], outputs[m] = frozen_state(models[m], *args)
        x = torch.cat([blocks["R"], blocks["E"]], -1)
        with torch.no_grad():
            initial = anchor(x, logits["R"])
        for key in ("point_phase", "raw_location", "q10", "q90"):
            assert torch.equal(initial[key], outputs["R"][key]), (start, key)
        for key, value in dict(
            features=x,
            rgb_width_logits=logits["R"],
            rgb_phase=outputs["R"]["point_phase"],
            event_phase=outputs["E"]["point_phase"],
        ).items():
            parts[key].append(value.cpu().numpy())
        if start % 2048 == 0:
            print(role, start, n, flush=True)
    values: dict = {k: np.concatenate(v) for k, v in parts.items()}
    values.update({k: e[k] for k in ("sample_token", "group_id", "target_phase", "mass", "role")})
    temporary = target / "states.pending.npz"
    np.savez_compressed(temporary, **values)
    temporary.replace(target / "states.npz")
    write(
        target / "SEALED.json",
        dict(
            role=role,
            population=n,
            binding_sha256=sha(target / "BINDING.json"),
            states_sha256=sha(target / "states.npz"),
            initial_prediction_and_quantile_bitwise_parity=True,
            optimizer_updates=0,
        ),
    )
    print(role, "sealed", n, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("H", "V"))
    args = parser.parse_args()
    setup()
    extract(args.role)


if __name__ == "__main__":
    main()
