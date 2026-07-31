import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.data.eap import EAPEventReader, build_eap_temporal_windows
from e_jepa_ttc.data.garlttc_eap import (
    load_garlttc_train_index,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig


def audit_clock_alignment() -> None:
    eap_root = Path("E:/eAP_dataset")
    garlttc_root = Path("E:/GarlTTC_dataset")
    
    # Extract all sequences from train.parquet
    data_df = pd.read_parquet(garlttc_root / "data" / "train.parquet")
    split_seqs = data_df["sequence_id"].unique().tolist()

    # Load dataset
    index = load_garlttc_train_index(
        garlttc_root, split_seqs
    )
    df = index.merged.dropna(subset=["event_windows_us"])
    
    config = EAPJEPATrainerConfig()

    results_by_sequence = {}
    reference_out_of_bounds_count = 0
    context_out_of_bounds_count = 0
    contexts_without_valid_future = 0
    valid_future_count_by_horizon = [0] * len(config.horizons_ms)
    invalid_future_count_by_horizon = [0] * len(config.horizons_ms)
    
    examples = []
    
    # Process by sequence
    for seq_id, seq_df in df.groupby("sequence_id"):
        seq_df = seq_df.sort_values("timestamp_us")
        
        # We need to process each context (unique timestamp_us)
        context_rows = seq_df.drop_duplicates(subset=["timestamp_us"]).copy()
        
        event_references = []
        clock_offsets = []
        
        reader = None
        current_events_path = None
        
        for _idx, row in context_rows.iterrows():
            parsed_windows = normalize_event_windows_us(row["event_windows_us"])
            event_reference_end_us = int(parsed_windows[-1][1])
            clock_offset = int(row["timestamp_us"]) - event_reference_end_us
            
            event_references.append(event_reference_end_us)
            clock_offsets.append(clock_offset)
            
            events_path = row["events_path"]
            if events_path != current_events_path:
                if reader is not None:
                    reader.close()
                resolved = resolve_eap_events_path(eap_root, events_path)
                reader = EAPEventReader(resolved)
                reader.open()
                current_events_path = events_path
            
            # Check bounds
            if not (reader.t_start_us <= event_reference_end_us <= reader.t_end_us):
                reference_out_of_bounds_count += 1
            
            windows = build_eap_temporal_windows(
                reference_end_us=event_reference_end_us,
                event_window_ms=config.event_window_ms,
                horizons_ms=config.horizons_ms,
            )
            
            if (
                windows.context_start_us < reader.t_start_us
                or windows.context_end_us > reader.t_end_us
            ):
                context_out_of_bounds_count += 1
                
            future_valid = []
            for _h_start, h_end in windows.future_windows_us:
                if h_end > reader.t_end_us:
                    future_valid.append(False)
                else:
                    future_valid.append(True)
                    
            if not any(future_valid):
                contexts_without_valid_future += 1
                
            for h_idx, valid in enumerate(future_valid):
                if valid:
                    valid_future_count_by_horizon[h_idx] += 1
                else:
                    invalid_future_count_by_horizon[h_idx] += 1
            
            if len(examples) < 10:
                examples.append({
                    "sequence_id": str(seq_id),
                    "timestamp_us": int(row["timestamp_us"]),
                    "event_windows_us": str(row["event_windows_us"]),
                    "event_reference_end_us": int(event_reference_end_us),
                    "clock_offset_us": int(clock_offset),
                    "reader.t_start_us": int(reader.t_start_us),
                    "reader.t_end_us": int(reader.t_end_us),
                    "context_window": [windows.context_start_us, windows.context_end_us],
                    "future_windows": windows.future_windows_us,
                    "future_valid_mask": future_valid
                })
        
        if reader is not None:
            reader.close()
            
        context_rows["event_reference_end_us"] = event_references
        context_rows["clock_offset_us"] = clock_offsets
        
        timestamp_deltas = np.diff(context_rows["timestamp_us"].values)
        event_reference_deltas = np.diff(context_rows["event_reference_end_us"].values)
        
        delta_mismatch_count = int(np.sum(timestamp_deltas != event_reference_deltas))
        max_delta_err = 0
        if len(timestamp_deltas) > 0:
            max_delta_err = float(np.max(np.abs(timestamp_deltas - event_reference_deltas)))
            
        timestamps_monotonic = bool(np.all(timestamp_deltas >= 0))
        event_reference_monotonic = bool(np.all(event_reference_deltas >= 0))
        
        results_by_sequence[str(seq_id)] = {
            "row_count": int(len(seq_df)),
            "context_count": int(len(context_rows)),
            "timestamp_us_min": int(seq_df["timestamp_us"].min()),
            "timestamp_us_max": int(seq_df["timestamp_us"].max()),
            "event_reference_end_us_min": int(context_rows["event_reference_end_us"].min()),
            "event_reference_end_us_max": int(context_rows["event_reference_end_us"].max()),
            "clock_offset_us_min": int(context_rows["clock_offset_us"].min()),
            "clock_offset_us_max": int(context_rows["clock_offset_us"].max()),
            "clock_offset_us_mean": float(context_rows["clock_offset_us"].mean()),
            "clock_offset_us_std": float(context_rows["clock_offset_us"].std())
            if len(context_rows) > 1
            else 0.0,
            "unique_clock_offset_count": int(context_rows["clock_offset_us"].nunique()),
            "delta_mismatch_count": delta_mismatch_count,
            "maximum_delta_error_us": max_delta_err,
            "timestamps_monotonic": timestamps_monotonic,
            "event_reference_monotonic": event_reference_monotonic
        }
        
    output = {
        "sequences": results_by_sequence,
        "global_summary": {
            "reference_out_of_bounds_count": reference_out_of_bounds_count,
            "context_out_of_bounds_count": context_out_of_bounds_count,
            "contexts_without_valid_future": contexts_without_valid_future,
            "valid_future_count_by_horizon": valid_future_count_by_horizon,
            "invalid_future_count_by_horizon": invalid_future_count_by_horizon
        },
        "examples": examples
    }
    
    out_dir = Path("artifacts/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "garlttc_eap_clock_alignment_v1.json"
    
    out_data = json.dumps(output, indent=2)
    out_path.write_text(out_data)
    
    out_hash = hashlib.sha256(out_data.encode()).hexdigest()
    
    print(f"Saved: {out_path}")
    print(f"SHA256: {out_hash}")
    
    # Just print the global summary for visibility
    print("GLOBAL SUMMARY:")
    print(json.dumps(output["global_summary"], indent=2))

if __name__ == "__main__":
    audit_clock_alignment()
