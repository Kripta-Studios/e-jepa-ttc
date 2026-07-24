import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _init_stats() -> dict:
    return {
        "windows": 0,
        "unique_tracks": set(),
        "categories": defaultdict(int),
        "finite_ttc": 0,
        "approaching": 0,
        "receding": 0,
        "ttc_values": [],
        "risk_0_5_s_pos": 0,
        "risk_0_5_s_neg": 0,
        "risk_1_0_s_pos": 0,
        "risk_1_0_s_neg": 0,
        "risk_2_0_s_pos": 0,
        "risk_2_0_s_neg": 0,
        "risk_4_0_s_pos": 0,
        "risk_4_0_s_neg": 0,
        "missing_future_targets_per_horizon": defaultdict(int),
        "total_horizons": defaultdict(int),
        "shards": 0,
        "size_bytes": 0,
    }


def _finalize_stats(stats: dict) -> dict:
    ttc_arr = np.array(stats["ttc_values"])
    if len(ttc_arr) > 0:
        mean_ttc = float(np.mean(ttc_arr))
        median_ttc = float(np.median(ttc_arr))
        p25 = float(np.percentile(ttc_arr, 25))
        p75 = float(np.percentile(ttc_arr, 75))
    else:
        mean_ttc = median_ttc = p25 = p75 = 0.0

    missing_perc = {}
    for h in stats["missing_future_targets_per_horizon"]:
        tot = stats["total_horizons"][h]
        missing_perc[h] = (
            float(stats["missing_future_targets_per_horizon"][h] / tot) if tot > 0 else 0.0
        )

    return {
        "windows": stats["windows"],
        "unique_tracks": len(stats["unique_tracks"]),
        "categories": dict(stats["categories"]),
        "finite_ttc": stats["finite_ttc"],
        "approaching": stats["approaching"],
        "receding": stats["receding"],
        "ttc_mean": mean_ttc,
        "ttc_median": median_ttc,
        "ttc_p25": p25,
        "ttc_p75": p75,
        "risk_0_5_s_pos": stats["risk_0_5_s_pos"],
        "risk_0_5_s_neg": stats["risk_0_5_s_neg"],
        "risk_1_0_s_pos": stats["risk_1_0_s_pos"],
        "risk_1_0_s_neg": stats["risk_1_0_s_neg"],
        "risk_2_0_s_pos": stats["risk_2_0_s_pos"],
        "risk_2_0_s_neg": stats["risk_2_0_s_neg"],
        "risk_4_0_s_pos": stats["risk_4_0_s_pos"],
        "risk_4_0_s_neg": stats["risk_4_0_s_neg"],
        "missing_future_targets_perc": missing_perc,
        "shards": stats["shards"],
        "size_bytes": stats["size_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed split statistics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest_dir = args.manifest.parent

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    sequence_splits = manifest.get("sequence_splits", {})
    shards = manifest.get("shards", [])

    seen_shards = set()
    for shard in shards:
        if shard["path"] in seen_shards:
            raise ValueError(f"Validation failed: Duplicate shard path detected: {shard['path']}")
        seen_shards.add(shard["path"])

    seq_stats = defaultdict(_init_stats)
    split_stats = defaultdict(_init_stats)
    split_tracks = defaultdict(set)

    for shard in shards:
        seq_id = shard["sequence_id"]
        split = sequence_splits.get(seq_id, "unknown")
        if split == "unknown":
            raise ValueError(f"Validation failed: Sequence {seq_id} not assigned to any split.")

        path = manifest_dir / shard["path"]
        if not path.exists():
            import sys

            logging.error(f"Validation failed: Shard not found: {path}")
            sys.exit(1)

        size = path.stat().st_size

        seq_stats[seq_id]["shards"] += 1
        seq_stats[seq_id]["size_bytes"] += size
        split_stats[split]["shards"] += 1
        split_stats[split]["size_bytes"] += size

        try:
            data = np.load(path, allow_pickle=True)
            ttc_s = data["ttc_s"].squeeze(-1) if data["ttc_s"].ndim > 1 else data["ttc_s"]
            if np.any(np.isnan(ttc_s)):
                raise ValueError(f"Validation failed: NaN TTC values detected in shard {path}")
            track_id = data["track_id"]
            category = data["category"]
            n_samples = ttc_s.shape[0]

            seq_stats[seq_id]["windows"] += n_samples
            split_stats[split]["windows"] += n_samples

            for i in range(n_samples):
                global_track_id = f"{seq_id}_{track_id[i]}"
                seq_stats[seq_id]["unique_tracks"].add(global_track_id)
                split_stats[split]["unique_tracks"].add(global_track_id)
                split_tracks[split].add(global_track_id)
                seq_stats[seq_id]["categories"][category[i]] += 1
                split_stats[split]["categories"][category[i]] += 1

                t = ttc_s[i]
                if np.isfinite(t):
                    seq_stats[seq_id]["finite_ttc"] += 1
                    split_stats[split]["finite_ttc"] += 1

                    if t > 0:
                        seq_stats[seq_id]["approaching"] += 1
                        split_stats[split]["approaching"] += 1
                        seq_stats[seq_id]["ttc_values"].append(t)
                        split_stats[split]["ttc_values"].append(t)
                    else:
                        seq_stats[seq_id]["receding"] += 1
                        split_stats[split]["receding"] += 1
                else:
                    seq_stats[seq_id]["receding"] += 1
                    split_stats[split]["receding"] += 1

                for risk in [0.5, 1.0, 2.0, 4.0]:
                    is_pos = (t > 0) and (t <= risk)
                    if is_pos:
                        seq_stats[seq_id][f"risk_{str(risk).replace('.', '_')}_s_pos"] += 1
                        split_stats[split][f"risk_{str(risk).replace('.', '_')}_s_pos"] += 1
                    else:
                        seq_stats[seq_id][f"risk_{str(risk).replace('.', '_')}_s_neg"] += 1
                        split_stats[split][f"risk_{str(risk).replace('.', '_')}_s_neg"] += 1

            if "future_window_start_us" in data and "prediction_horizons_s" in data:
                fws = data["future_window_start_us"]  # [B, H]
                horizons = data["prediction_horizons_s"]
                for h_idx, h_s in enumerate(horizons):
                    h_ms = int(h_s * 1000)
                    missing = np.sum(fws[:, h_idx] == -1)
                    seq_stats[seq_id]["missing_future_targets_per_horizon"][h_ms] += int(missing)
                    split_stats[split]["missing_future_targets_per_horizon"][h_ms] += int(missing)
                    seq_stats[seq_id]["total_horizons"][h_ms] += n_samples
                    split_stats[split]["total_horizons"][h_ms] += n_samples

        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            raise

    # Data Leakage checks
    all_splits = list(split_tracks.keys())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            s1, s2 = all_splits[i], all_splits[j]
            intersection = split_tracks[s1].intersection(split_tracks[s2])
            if intersection:
                msg = (
                    f"Validation failed: Data leakage detected! Track IDs "
                    f"{intersection} appear in both '{s1}' and '{s2}'."
                )
                raise ValueError(msg)

    final_seq_stats = {k: _finalize_stats(v) for k, v in seq_stats.items()}
    final_split_stats = {k: _finalize_stats(v) for k, v in split_stats.items()}

    # Categorical completeness
    for required_split in ["validation", "test", "calibration"]:
        if (
            required_split in final_split_stats
            and final_split_stats[required_split]["windows"] == 0
        ):
            raise ValueError(f"Validation failed: Split '{required_split}' is empty.")

    cal_stats = final_split_stats.get("calibration", None)
    if cal_stats is not None:
        for risk in [0.5, 1.0, 2.0, 4.0]:
            k_pos = f"risk_{str(risk).replace('.', '_')}_s_pos"
            k_neg = f"risk_{str(risk).replace('.', '_')}_s_neg"
            if cal_stats[k_pos] == 0 or cal_stats[k_neg] == 0:
                msg = (
                    "Validation failed: Calibration split lacks positive/negative "
                    f"examples for risk threshold {risk}s."
                )
                raise ValueError(msg)

    output_data = {
        "sequences": final_seq_stats,
        "splits": final_split_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Markdown export
    md_output = args.output.with_suffix(".md")
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("# EAP Dataset Statistics\n\n")
        f.write("## Splits\n")
        for split, s in final_split_stats.items():
            f.write(f"\n### {split}\n")
            for k, v in s.items():
                f.write(f"- **{k}**: {v}\n")
        f.write("\n## Sequences\n")
        for seq, s in final_seq_stats.items():
            f.write(f"\n### {seq}\n")
            for k, v in s.items():
                f.write(f"- **{k}**: {v}\n")

    logging.info(f"Statistics written to {args.output} and {md_output}")


if __name__ == "__main__":
    main()
