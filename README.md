# AERIS-GB Inference — Aurora XPU

Self-contained inference package for running **AERIS-GB** ensemble forecasts
on **Aurora (Intel XPU)** at ALCF.

Derived from the private `agentic-wxbench` research repo (Argonne CROCUS group).
Everything here has been validated in production for the WFIP3 and AWAKEN
benchmark campaigns (September 2024 and August 2023 ICs, K=50 ensemble members,
240 h horizons).

---

## What is included

```
AERIS_GB_inference/
├── README.md                   ← this file
├── CHANGES.md                  ← detailed changelog vs upstream argonne-lcf/AERIS-GB
│
├── inference/
│   ├── aeris_inference.py      ← main inference script (clean, self-contained)
│   ├── run_aeris.pbs           ← PBS job script (K=50, 16 nodes, DP=8)
│   └── setup_venv.sh           ← one-time venv bootstrap (run once per filesystem)
│
├── patches/
│   ├── aeris_swin_shift_window.py  ← fix #1: hardcoded window_size=60 assertion
│   └── aeris_swin_shift_DP.py      ← fix #2: swin_shift uses global ranks, breaks DP>1
│
├── aeris_wp_inference/         ← vendored model source (patches already applied)
│   ├── model.py                ← both patches applied inline
│   ├── era5.py                 ← unmodified
│   └── inference.py            ← unmodified (not used by aeris_inference.py)
│
└── checkpoints/
    └── p_1Bd66c_1100k_lrrd_base/   ← 10-stage PP checkpoint used in production
        ├── checkpoint_PP0.pth       ← ~94 MB (embedding stage)
        ├── checkpoint_PP1..PP8.pth  ← ~2.7 GB each (transformer stages)
        └── checkpoint_PP9.pth       ← ~137 MB (output stage)
```

**Not included:**
- IC preparation script (you need a pre-built ARCO ERA5 NetCDF; see notes below)
- WB2 normalisation constants (read from `/flare/datasets/wb2/...` at runtime)

---

## Quick-start

### 1. Bootstrap the venv (once per filesystem)

```bash
module load frameworks
bash inference/setup_venv.sh
```

### 2. Point to an initial condition NetCDF

A sample IC for **2024-09-20T00:00:00Z** is provided at:

```
samples/aeris_20240920T00.nc
```

(On Flare at `/flare/Chicago_AIDT/dfytanidis/AERIS_GB_inference/samples/`; not tracked by git due to file size.)

ERA5 data for building your own ICs is also available via
[WeatherBench2](https://weatherbench2.readthedocs.io/) (Google Cloud bucket
`gs://weatherbench2/datasets/era5/`) or locally on Flare at
`/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/`.

### 3. Submit the job

Edit the variables block at the top of `inference/run_aeris.pbs`:

```bash
IC_NC="samples/aeris_20240920T00.nc"    # sample IC (or your own)
CYCLE="2024-09-20T00:00:00Z"            # forecast cycle (must match IC)
OUT_DIR="/path/to/output"               # where member_XXX.nc files go
MEMBERS=50                              # ensemble size
LEAD_HOURS=240                          # forecast horizon
```

Then submit:

```bash
qsub inference/run_aeris.pbs
```

Output: `${OUT_DIR}/member_000.nc … member_049.nc`
Each file: ~7.2 GB (WFIP3/AWAKEN production; zlib complevel=1).

---

## Checkpoint

`checkpoints/p_1Bd66c_1100k_lrrd_base` — configuration summary:

| Parameter | Value |
|-----------|-------|
| Architecture | AERIS-GB (Swin Transformer + diffusion) |
| Dim | 1536 |
| Heads | 12 |
| Head dim | 128 |
| PP stages | 10 |
| Window size | 30 × 30 (6 h interval) |
| Input variables | 69 (no SST) |
| Diffusion | DPMSolver++ Heun, 10 sub-steps default |
| Training step | ~1.1 M (lr-reduced run 2) |

The checkpoint ships as 10 pipeline-parallel (PP) shards:
`checkpoint_PP0.pth .. checkpoint_PP9.pth` plus `.hydra/config.yaml`.
They are re-assembled at load time via `convert_inference_checkpoint()`.

---

## Topology guide

AERIS uses **window parallelism (WP)**: the 720×1440 spatial domain is
partitioned across WP_X × WP_Y tiles. For `p_1Bd66c` (window_size=30):
- WP = 4×4 = **16 tiles per model instance** (1 XPU tile per rank)
- 1 Aurora node has 12 tiles → minimum 2 nodes per instance
- DP groups replicate the model across independent ensembles

Recommended topology for K=50:

| Nodes | PPN | Total ranks | SP | DP | Members per DP group |
|-------|-----|------------|----|----|----------------------|
| 16    | 8   | 128        | 16 | 8  | 7 (groups 0-1), 6 (groups 2-7) |
| 6     | 12  | 72 (→ DP=4)| 16 | 4  | Use for K=20 only    |
| 2     | 8   | 16 (→ DP=1)| 16 | 1  | Use for K=1 / smoke  |

---

## System requirements

- **Cluster**: Aurora (ALCF) or any Intel XPU system with the Aurora frameworks module
- **Module**: `module load frameworks` (provides PyTorch XPU 2.10, mpi4py, oneCCL)
- **WB2 norms**: `/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/` (read-only)
- **Allocation**: `AI4SRM` (or your project allocation)

---

## Changes vs upstream

See `CHANGES.md` for the full changelog with line-level references.
The two critical bugs are summarised in `patches/`.
