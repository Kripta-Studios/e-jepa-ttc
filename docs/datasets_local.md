# Local Dataset Study

Date: 2026-06-30.

The handoff contains an EvTTC mini subset at `datasets/evttc`.

Observed structure:

```text
datasets/evttc/
  CCRs-1/
    low-100/overlap-100/
    medium-100/overlap-100/
    high-100/overlap-100/
```

Each sequence folder contains:

- one large event HDF5 file;
- one `gt.hdf5`;
- one `ttc.csv`;
- one `leftlabel/` folder with ISAT JSON labels;
- one `.bag` and one `.mp4`, ignored for the initial MVP.

Observed aggregate file counts:

```text
.hdf5: 6 files, about 15.4 GiB
.csv:  3 files
.json: 365 label files
.bag:  3 files
.mp4:  3 files
```

The `ttc.csv` files are whitespace-separated without headers. The current parser treats columns as:

```text
frame_id, timestamp_seconds, distance_like, relative_speed_like, ttc_seconds
```

The final column is used as the supervised TTC target. This mapping must be verified against the
official EvTTC schema before publishing experimental results.

Initial sequence counts from `ttc.csv`:

```text
low-100:    750 rows
medium-100: 900 rows
high-100:   1150 rows
```

The local subset is useful for loader validation and smoke experiments only. It is not enough for
claims about cross-scenario generalization because all three sequences come from `CCRs-1`.
