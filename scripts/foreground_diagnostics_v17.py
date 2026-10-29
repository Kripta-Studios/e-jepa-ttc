"""CPU P-only pilot of native foreground sensitivity, without optimizer updates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.causal_scale_ttc import target_log_ratio_from_ttc
from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import OLD, OWNER, RUN
from operational.representation_audit.spatial_cache import native_parent, source_for
from operational.rgb_port.infer_experts import delta_matrix


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = read(OWNER / "P_MANIFEST.json")
    table = pd.read_parquet(manifest["rows_path"]).set_index("sample_token")
    source = source_for("P")
    # This diagnostic holds one query at a time; cache capacity changes no inputs.
    source._inputs.limit = 16 * 1024**2
    ordered = table.loc[source._tokens]
    target = ordered.target_ttc.to_numpy(np.float64)
    masks = {
        "0-1": (target > 0) & (target <= 1),
        "1-3": (target > 1) & (target <= 3),
        "3-6": (target > 3) & (target <= 6),
        "6plus": target > 6,
        "negative": target < 0,
    }
    rng = np.random.default_rng(20261012)
    samples = {
        band: rng.choice(np.flatnonzero(mask), 8, replace=False) for band, mask in masks.items()
    }
    model = native_parent()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.encoder.foreground.requires_grad_(True)
    parameters = list(model.encoder.foreground.parameters())
    rows = []
    checked = {}
    try:
        for band, indices in samples.items():
            for index in indices:
                path = OLD / f"vector_cache/P/part_{index // 128:06d}.npz"
                if path.name not in checked:
                    checked[path.name] = sha(path)
                    if checked[path.name] != read(path.with_suffix(".json"))["sha256"]:
                        raise ValueError("P target shard changed")
                with np.load(path, allow_pickle=False) as archive:
                    offset = int(index % 128)
                    if str(archive["sample_token"][offset]) != str(source._tokens[index]):
                        raise ValueError("P target/input identity mismatch")
                    heights = torch.from_numpy(
                        archive["target_height_normalized"][offset : offset + 1].copy()
                    )
                    valid = torch.from_numpy(
                        archive["target_valid"][offset : offset + 1].copy()
                    ).bool()
                batch = source.batch([int(index)], "event")
                delta = delta_matrix(batch.delta_t_s, batch.events.shape[1])
                output = model(batch.events.float(), delta)
                required, physical = target_log_ratio_from_ttc(batch.target_ttc_s, delta[:, -1])
                if not physical.all() or not valid[:, -2:].all():
                    raise ValueError("Unexpected P target support")
                log_height = output.visible_height_normalized.clamp_min(1e-6).log()
                truth_height = heights.clamp_min(1e-6).log()
                height_loss = (log_height[valid] - truth_height[valid]).square().mean()
                temporal_loss = (
                    (
                        (log_height[:, -1] - log_height[:, -2])
                        - (truth_height[:, -1] - truth_height[:, -2])
                    )
                    .square()
                    .mean()
                )
                logvar = output.log_ratio_log_variance[:, -1]
                nll = (
                    0.5
                    * (-logvar).exp()
                    * (output.pair_log_height_ratio[:, -1] - required).square()
                    + 0.5 * logvar
                ).mean()
                vectors = {}
                for name, loss in (
                    ("height", height_loss),
                    ("temporal", temporal_loss),
                    ("nll", nll),
                ):
                    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                    vectors[name] = torch.cat([g.flatten() for g in gradients]).double()
                probabilities = (
                    output.foreground_logits / model.config.foreground_temperature
                ).sigmoid()
                row = {
                    "sample_token": str(source._tokens[index]),
                    "group_id": str(ordered.iloc[index].group_id),
                    "band": band,
                    "target_ttc": float(target[index]),
                    "prediction_ttc": float(output.ttc_mean_seconds.detach()[0]),
                    "height_last": float(output.visible_height_normalized.detach()[0, -1]),
                    "target_height_last": float(heights[0, -1]),
                    "probability_mean": float(probabilities.detach().mean()),
                    "probability_std_spatial": float(
                        probabilities.detach().flatten(-2).std(-1).mean()
                    ),
                    "sigmoid_saturation_fraction": float(
                        ((probabilities < 0.01) | (probabilities > 0.99)).float().mean()
                    ),
                    "height_loss": float(height_loss.detach()),
                    "temporal_loss": float(temporal_loss.detach()),
                    "nll": float(nll.detach()),
                }
                row.update(
                    {name + "_gradient_norm": float(v.norm()) for name, v in vectors.items()}
                )
                for name in ("height", "temporal"):
                    a, b = vectors[name], vectors["nll"]
                    denominator = a.norm() * b.norm()
                    row[name + "_nll_cosine"] = (
                        float(a @ b / denominator) if denominator > 0 else None
                    )
                rows.append(row)
            print(band, len(rows), flush=True)
        if any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()):
            raise AssertionError("Diagnostic changed native parameters or buffers")
    finally:
        source.close()
    folder = RUN / "foreground_diagnostic"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "P40.csv", index=False)
    write(
        folder / "SCOPE.json",
        {
            "role": "P",
            "queries": len(rows),
            "optimizer_updates": 0,
            "device": "cpu",
            "dtype": "float32",
            "dropout": "eval",
            "gradient_parameters": "original native encoder.foreground only",
            "parameters_and_buffers_unchanged": True,
            "components": "unweighted log-height MSE, log-height-change MSE and pair NLL only",
            "excludes": "top-k, risk and all other native producer loss terms",
            "sampling": "8 without replacement per TTC band, seed20261012; descriptive pilot only",
            "generalization_claim": False,
            "parent_sha256": read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"],
            "manifest_sha256": sha(OWNER / "P_MANIFEST.json"),
            "source_sha256": sha(Path(__file__)),
            "target_shards": checked,
            "foreground_decoder": model.config.foreground_decoder,
        },
    )


if __name__ == "__main__":
    main()
