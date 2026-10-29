# GarlTTC Dataset

This repository contains the public GarlTTC structured annotations and benchmark
inputs. RGB frames and event streams are not duplicated here; they are referenced
from `NAIL-HNU/eAP-dataset`.

## Required Companion Dataset

Download `NAIL-HNU/eAP-dataset` first, then download this repository. Pass the
local eAP public root as `--data-root` and this repository root as
`--garlttc-annotation-root` when running the release code.

## Files

- `data/train.parquet`: train sample index and eAP media references. Samples
  whose frames are outside the published eAP train40 media are filtered out.
- `annotations/train.parquet`: train TTC supervision keyed by `sample_token`.
- `data/test_inputs.parquet`: public benchmark inputs with no TTC ground truth.
- `splits/train.txt`: the published train40 sequence list.
- `splits/test.txt`: the benchmark test12 sequence list.
- `dataset_info.json`: published sequence metadata.
- `sample_submission.json`: valid JSON submission template.

The benchmark accepts JSON only. Test labels are private and are not included in
this public dataset repository.
