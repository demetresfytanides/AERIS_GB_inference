# Changes vs upstream argonne-lcf/AERIS-GB

Upstream snapshot: `argonne-lcf/AERIS-GB` (private repo, cloned June 2025).
All changes were validated on Aurora XPU (ALCF) with checkpoint
`p_1Bd66c_1100k_lrrd_base` and confirmed in paper-grade K=50 ensemble runs.

---

## Bug fix 1 — swin_shift hardcoded window_size=60

**File**: `aeris_wp_inference/model.py`
**Patch script**: `patches/aeris_swin_shift_window.py`
**Status**: Applied to `aeris_wp_inference/model.py` in this repo.

### Problem

`swin_shift()` (the shift-window attention exchange operation) contained four
hardcoded references to `window_size == 60`:

```python
# Line 22-23 in upstream model.py
assert ws_y == 60
assert ws_x == 60

# Lines 116 and 118
out = rearrange(..., ws_y=60, ..., ws_x=60)  # appears twice
```

The function already extracted `ws_y, ws_x = window_size[0], window_size[1]`
at line 21, making the assertions and the two rearranges redundant — but they
hard-fail for any checkpoint that uses window_size != (60, 60).

The production checkpoint `p_1Bd66c_1100k_lrrd_base` uses **window_size=[30,30]**
(6 h interval family). Running the upstream code with this checkpoint raises:
```
AssertionError  (at model.py:22)
```

### Fix

Replace the two assertions with comments and change the two rearranges to use
`ws_y=ws_y, ws_x=ws_x` (the variables already in scope):

```python
# assert ws_y == 60  # PATCHED by aeris_swin_shift_window.py
# assert ws_x == 60  # PATCHED
out = rearrange(..., ws_y=ws_y, ..., ws_x=ws_x)  # PATCHED
out = rearrange(..., ws_y=ws_y, ..., ws_x=ws_x)  # PATCHED
```

The patch script (`patches/aeris_swin_shift_window.py`) is idempotent and
detects a PATCH_MARKER comment so it won't double-apply.

---

## Bug fix 2 — swin_shift uses global ranks, breaks DP > 1

**File**: `aeris_wp_inference/model.py`
**Patch script**: `patches/aeris_swin_shift_DP.py`
**Status**: Applied to `aeris_wp_inference/model.py` in this repo.

### Problem

`swin_shift()` computes peer ranks via the WP grid (group-local indices)
and passes them to `torch.distributed.send(dst=...)` / `recv(src=...)`.
The `dst`/`src` kwargs expect **global** ranks.

With DP=1, global and group-local ranks coincide (group = [0..SP-1]),
so the bug is invisible. With DP > 1, DP group `g` owns global ranks
`[g*SP .. (g+1)*SP - 1]`. Passing local rank 8 as `dst` means "global
rank 8" — which belongs to DP group 0, not the current group:

```
ValueError: Global rank 8 is not part of group [16..31]
```

This crash was first seen in the K=20 run on 6 nodes (DP=4). All
`DP > 1` configurations are affected (DP=2, DP=4, DP=8, …).

### Fix

Switch from `dst=` / `src=` (global rank) to `group_dst=` / `group_src=`
(group-local rank, available in PyTorch 2.x):

```python
# Before (6 call sites, 3 send/recv blocks):
torch.distributed.send(corner, dst=corner_dst, group=sp_group)
torch.distributed.recv(recv_corner, src=corner_src, group=sp_group)

# After:
torch.distributed.send(corner, group_dst=int(corner_dst), group=sp_group)
torch.distributed.recv(recv_corner, group_src=int(corner_src), group=sp_group)
```

Six call sites in total (corner, vertical, horizontal exchanges, two
orderings each). The patch script covers all six.

---

## Bug fix 3 — ARCO v3 returns NaN for static fields (IC builder only)

**Not affecting this repo** (the IC builder is not included here).

For reference: `geopotential_at_surface` and `land_sea_mask` return NaN
from the ARCO ERA5 v3 zarr store despite being listed in the dataset.
The fix is to fall back to the WB2 HDF5 at
`/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/test/2020_0000.h5`
for those two fields only.

---

## Design change — custom rollout replacing vendor inference.py

**File**: `inference/aeris_inference.py` (new, replaces vendor `inference.py`)

The vendor's `inference.py` hardcodes an output path inside a WB2 HDF5 file
and is coupled to the Aurora training filesystem layout. `aeris_inference.py`
replaces it with:

1. **`ARCOInferenceDataset`** — reads our pre-fetched ARCO ERA5 NetCDF instead
   of the WB2 HDF5 file. Exposes the same interface `AERIS_SP_ERA5_Data` expects.

2. **`rollout_and_collect()`** — same DPMSolver++ Heun diffusion math as the
   vendor, but collects each step's output into numpy arrays instead of writing
   to a hardcoded HDF5. Supports multi-DP member distribution (round-robin by
   global member id).

3. **Parallel distributed writer** — each DP group's SP-rank-0 writes its own
   member files in parallel after the rollout. Eliminates a 43 GB Gloo gather
   that caused hangs at DP ≥ 12 (job 8527209). Output: `member_000.nc …
   member_049.nc` (same format as GenCast member files).

4. **APE on CPU** — the Absolute Positional Encoding (~600 MB tensor) is
   computed on CPU before moving the per-rank slice (~38 MB) to XPU. Avoids
   a Level Zero allocator deadlock that stalled 128 ranks for 18+ minutes
   at DP=8/12 (job 8529025). The CPU computation is mathematically identical
   (pure trigonometry, no learned weights).

---

## Topology history — what was tried and what works

| DP | Nodes | PPN | K | Result |
|----|-------|-----|---|--------|
| 1  | 2     | 8   | 1 | PASS — smoke + 240h K=1 |
| 2  | 4     | 8   | 2 | PASS — 24h validation (job 8526653) |
| 4  | 6     | 12  | 20| PASS — full 240h K=20 in 3h13m (job 8526669) |
| 12 | 16    | 12  | 50| FAIL — Gloo init hung, first SP subgroup only connected |
| 8  | 16    | 8   | 50| PASS — full 240h K=50 in ~5h (job 8527xxx) |

DP=8 (16 nodes × 8 ppn) is the **validated production topology** for K=50.
DP=12 (16 nodes × 12 ppn) reliably hangs at Gloo init — do not use.

---

## Performance

Measured on Aurora XPU with `p_1Bd66c_1100k_lrrd_base`, WP=4×4=16 tiles,
2 Aurora nodes (24 tiles, 16 used):

- Per diffusion step: ~4.3 s
- Per rollout step (10 diffusion sub-steps): ~43 s
- 240 h / 6 h = 40 rollout steps per member: ~29 min per member
- K=20 on DP=4 (5 members sequential per DP group): ~3h 15min wall
- K=50 on DP=8 (7 members sequential per DP group 0-1, 6 for groups 2-7): ~5h wall
