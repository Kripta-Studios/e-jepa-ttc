Stable Screen v3 — corrected flat overlay
=========================================

This archive is intentionally flat: its top-level entries are scripts/ and
configs/, so extracting it into the repository root places every file in the
correct location.

The runner has a new name and does not overwrite the historical runner:
  scripts/run_e_jepa_garl_final_stable_v3.py

Without --pretrained it selects:
  configs/experiment/e_jepa_garl_event_dense_level_dynamics_stable_scratch_screen_v3.yaml

With --pretrained it selects:
  configs/experiment/e_jepa_garl_event_dense_level_dynamics_stable_transfer_screen_v3.yaml
