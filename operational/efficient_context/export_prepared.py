"""Copy only hash-bound small TRAIN inputs and weights for independent output replay."""

from .common import Campaign, atomic_bytes, atomic_json, digest, npz, read


def export(c: Campaign) -> None:
    """Reconstruct exactly the inputs already measured; perform no new profiling."""
    import numpy as np

    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS

    freeze = read(c.out / "prepared_heads/PROTOCOL.json")
    night = c.historical / "artifacts/simplex_t/nocturnal_20261003"
    source = read(night / "PROFILE_INPUT_EXPORT.json")
    route = read(c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json")
    target = c.out / "prepared_heads/replay"
    models = {}
    for label in ("H1", "H8", "H16"):
        record = source["models"][label + "_SEED7"]
        models[label] = {"config": record["constructor"]["config"]}
        for kind in ("weights", "inputs"):
            path = night / record[kind + "_path"]
            if digest(path) != freeze["source_pins"][label + "_" + kind]:
                raise ValueError("prepared TRAIN replay binding changed")
            atomic_bytes(target / f"{label}_{kind}.npz", path.read_bytes())
    with np.load(target / "H16_inputs.npz", allow_pickle=False) as z:
        wide = {k: z[k][:, WIDE_SLOTS].copy() for k in ("features", "times", "valid")}
        wide["experts"] = z["experts"].copy()
    for i, query in enumerate(route["queries"]):
        directory = c.historical / route["index_dirs"][query["pool"]]
        manifest = read(directory / "INDEX_MANIFEST.json")
        index_path = directory / "query_context_index.npz"
        if digest(index_path) != manifest["index_sha256"]:
            raise ValueError("prepared query timing binding changed")
        with np.load(index_path, allow_pickle=False) as z:
            if str(z["tokens"][query["index_row"]]) != query["sample_token"]:
                raise ValueError("prepared replay TRAIN query mapping differs")
            gap = np.diff(-z["lag_us"][list(WIDE_SLOTS)]) / 1e6
        valid = wide["valid"][i]
        wide["times"][i, :, 2] = 0
        wide["times"][i, 1:, 2] = gap.astype(np.float32) * valid[:-1]
        wide["times"][i, ~valid] = 0
    npz(target / "WIDE_inputs.npz", **wide)
    checkpoint = c.out / "fits/seed7/fold0/checkpoint_last.pt"
    if digest(checkpoint) != freeze["source_pins"]["WIDE_checkpoint"]:
        raise ValueError("measured WIDE TRAIN head endpoint changed")
    atomic_bytes(target / "WIDE_weights.npz", checkpoint.with_name("WEIGHTS.npz").read_bytes())
    models["WIDE"] = {"config": c.freeze()["sources"]["0"]["model"]}
    atomic_json(
        target / "MODELS.json",
        {
            "models": models,
            "query_tokens": [q["sample_token"] for q in route["queries"]],
            "scope": "included TRAIN head outputs only; no new latency measurements",
            "optimizer_updates": 0,
        },
    )
