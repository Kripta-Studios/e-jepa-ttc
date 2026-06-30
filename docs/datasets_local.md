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

Validated event HDF5 layout:

```text
prophesee/event_cam_left/x            uint16 [N]
prophesee/event_cam_left/y            uint16 [N]
prophesee/event_cam_left/t            int64  [N], microseconds from sequence start
prophesee/event_cam_left/p            int8   [N]
prophesee/event_cam_left/ms_map_idx   uint64 [milliseconds]
prophesee/event_cam_left/calib/resolution = [1280, 720]
```

Observed event counts:

```text
low-100:       44,005,484 left events, ms_map_idx length 16,886
medium-100:    49,470,495 left events, ms_map_idx length 18,475
high-100:     266,944,580 left events, ms_map_idx length 26,339
```

`gt.hdf5` contains stereo Prophesee depth and pose:

```text
prophesee/{left,right}/depth/depth
prophesee/{left,right}/depth/ts
prophesee/{left,right}/pose/pose
prophesee/{left,right}/pose/ts
```

The MVP records `gt.hdf5` in the manifest but does not use depth or pose in the primary event-only
TTC path.

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

Default local split:

```text
train:      CCRs-1-low-100-overlap-100
validation: CCRs-1-medium-100-overlap-100
test:       CCRs-1-high-100-overlap-100
```

With `context_ms=100`, `stride_ms=20`, horizons `[25, 50, 100, 250, 500]`, and TTC clipping
`[0.1, 12.0]`, the generated local index contains 1,230 windows. The index is stored under
`data/cache/` and is not committed.

The local subset is useful for loader validation and smoke experiments only. It is not enough for
claims about cross-scenario generalization because all three sequences come from `CCRs-1`.
