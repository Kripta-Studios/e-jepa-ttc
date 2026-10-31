# Correlated native-expert failure augmentation

User's continuing research instruction authorizes this fixed development test.
Prior sign-safe and geometric-channel augmentations were rejected. No new
candidate has been selected or scored on the competition sequences.

## Mechanism

The frozen adapter supplies F0, RGB and event estimates to a688parameter gate.
F0 depends on the other producers, so their failures are not independent.
Slider750's previously exposed startup diagnosis includes simultaneous F0/RGB
sign/magnitude failures. Slider1000 has mutually consistent but jointly biased
positive native estimates. Existing QA corruption changes just ONE native expert
in25%of training rows; it does not simulate either correlated failure class.

This one additional control changes only which experts are perturbed. It is an
expert-output failure simulation, not a claim of raw-sensor augmented training.
No additional input, model capacity, label, encoder update or benchmark fit.

## Fixed recipe before execution

- Same2608eligible queries,25groups,eightfamily-held-out folds,seeds7/13/21.
- Original inverse ExpertRouter QA architecture688parameters, normalization,
  family-balanced MAE/RAE loss,300updates AdamW lr.01 wd.01,clip1.
- In25%of rows, draw one of four patterns uniformly: one random native expert;
  F0+RGB; F0+event; all three. Multiply all selected members of that row by the
  SAME factor from{-3,-1,.25,4}, exactly the original factor support. Geometry,
  quality and target remain unchanged. Keep50%clean+50%corrupted objective.
- Compare to frozen matchedIQ OOF from campaign_v1b, same arithmetic seed average
  for metrics. Require full coverage and no increase in road AND Slider500 macro
  MAE/RAE. No relaxed gates if rejected. Preserve per-seed OOF predictions.
- If the single control passes, refit three seeds on2608, export and freeze,
  then evaluate active eAPTRAIN40 preservation against22.1956387%/1.17704749s.
  Only after that gate may the frozen candidate be evaluated on434 exposed
  competition queries using corrected versioned targets. Never selectbythese.
- All variants, failures, input/source hashes, loss traces and timings retained.
  CPUcachedfeatures; no GPU workload competes with current CUDA objectiveprofile.

## Implementation and checks

New script scripts/train_phase_correlated_fusion.py and tests/unit/
test_phase_correlated_fusion.py. Do not modify originalmodel or frozenrun.
Tests must demonstrate deterministic paired/triple masks, identicalwithin-row
factor, unchangedgeometry/targets (notaccepted by augmentationfunction), and
explicit noaliasmutation ofcleaninputs. Normalization andtargets onlyfitmask;
assert groupdisjointness andexistingbenchmarkexclusions.
