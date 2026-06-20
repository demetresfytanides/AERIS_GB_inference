"""
AERIS-GB ensemble inference — Aurora XPU (ALCF)

Runs inside a PBS multi-rank job. Reads a pre-built ARCO ERA5 initial-condition
NetCDF, runs the AERIS diffusion rollout (DPMSolver++ Heun, 10 sub-steps per
rollout step), and writes one NetCDF per ensemble member in parallel.

Usage (from inside the PBS job):
  mpiexec -n 128 -ppn 8 python aeris_inference.py \\
      --cycle 2024-09-20T00:00:00Z \\
      --ic-source /path/to/aeris_20240920T00.nc \\
      --lead-hours 240 \\
      --members 50 \\
      --diffusion-steps 10 \\
      --wp-x 4 --wp-y 4 \\
      --checkpoint-dir /path/to/checkpoints/p_1Bd66c_1100k_lrrd_base \\
      --output-dir /path/to/output

Output: member_000.nc … member_049.nc, each ~7 GB (zlib complevel=1).

Requirements:
  - module load frameworks  (PyTorch XPU 2.10 + mpi4py + oneCCL)
  - WB2 normalisation constants at /flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/
  - aeris_wp_inference/ directory (patched vendor source) on PYTHONPATH
    → automatically resolved relative to this script's location

See CHANGES.md for a description of what was changed vs upstream argonne-lcf/AERIS-GB.

Exit codes: 0 ok, 1 bad args, 2 missing data, 3 inference failure, 5 setup error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── vendor source path ──────────────────────────────────────────────────────
# aeris_wp_inference/ lives one directory up from this script (same repo root).
VENDOR_DIR = Path(__file__).resolve().parents[1] / "aeris_wp_inference"
if not VENDOR_DIR.exists():
    raise RuntimeError(
        f"aeris_wp_inference/ not found at {VENDOR_DIR}.\n"
        "Either run from AERIS_GB_inference/ or set PYTHONPATH manually."
    )
sys.path.insert(0, str(VENDOR_DIR))

import h5py
import torch
import torch.accelerator as acc
from torch.distributed.device_mesh import DeviceMesh
from einops import rearrange
from omegaconf import OmegaConf
import xarray as xr

from model import LocalAERIS, convert_inference_checkpoint
from era5 import AERIS_SP_ERA5_Data

logger = logging.getLogger("aeris")

# ── WB2 normalisation constants ─────────────────────────────────────────────
WB2_ROOT = Path("/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug")


def _load_wb2_norms(variables: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
    """Load (x_mean, x_std, t_mean, t_std), each shape (C,)."""
    def stack(npz_path):
        with np.load(npz_path) as d:
            try:
                return np.stack([d[v] for v in variables], axis=0).astype(np.float32)
            except KeyError as e:
                raise RuntimeError(f"variable {e} not in {npz_path}") from e

    x_mean = stack(WB2_ROOT / "normalize_mean.npz")
    x_std  = stack(WB2_ROOT / "normalize_std.npz")
    t_mean = stack(WB2_ROOT / "normalize_diff_mean_6.npz")
    t_std  = stack(WB2_ROOT / "normalize_diff_std_6.npz")
    return x_mean, x_std, t_mean, t_std


# ── Dataset (ARCO ERA5 NetCDF → vendor AERIS_SP_ERA5_Data interface) ────────

class ARCOInferenceDataset:
    """
    Drop-in replacement for the vendor's ERA5HDF5InferenceDataset.

    Reads an ARCO ERA5 NetCDF containing:
      - 69 prognostic variables (no SST) at the forecast cycle timestamp
      - toa_incident_solar_radiation  (time dim, rollout_count+1 steps)
      - geopotential_at_surface       (static, no time dim)
      - land_sea_mask                 (static, no time dim)
    """
    def __init__(self, arco_nc: Path, prognostic_vars: list[str],
                 rollout_count: int):
        self.path = arco_nc
        self.rollout_count = rollout_count
        self.interval = 6  # p_1Bd66c family: 6 h step

        ds = xr.open_dataset(arco_nc)

        # Variable order must exactly match the vendor's inference.py:
        #   [prognostic..., toa, geopotential_at_surface, land_sea_mask]
        self.variables = list(prognostic_vars) + [
            "toa_incident_solar_radiation",
            "geopotential_at_surface",
            "land_sea_mask",
        ]
        self.channels = len(self.variables)

        self.lat_n = int(ds.sizes.get("lat", 721))
        self.lon_n = int(ds.sizes.get("lon", 1440))
        assert self.lat_n == 721,  f"expected lat=721, got {self.lat_n}"
        assert self.lon_n == 1440, f"expected lon=1440, got {self.lon_n}"

        # Build the (C, 721, 1440) IC array.
        ic_layers = []
        for v in self.variables:
            if v == "toa_incident_solar_radiation":
                ic_layers.append(ds[v].isel(time=0).values.astype(np.float32))
            else:
                ic_layers.append(ds[v].values.astype(np.float32))
        self._ic = np.stack(ic_layers, axis=0)  # (C, 721, 1440)

        # TOA trajectory for the full rollout: shape (rollout_count+1, 721, 1440)
        toa_full = ds["toa_incident_solar_radiation"].values.astype(np.float32)
        n_needed = rollout_count + 1
        if toa_full.shape[0] < n_needed:
            raise RuntimeError(
                f"TOA in {arco_nc} has {toa_full.shape[0]} steps but "
                f"rollout needs {n_needed} (rollout_count={rollout_count})"
            )
        self._toa_trajectory = toa_full[:n_needed]
        ds.close()

        self.lat = np.linspace(90.0,  -90.0, self.lat_n,  dtype=np.float32)
        self.lon = np.linspace(0.0,  359.75, self.lon_n, dtype=np.float32)

        self.x_means, self.x_stds, self.t_means, self.t_stds = \
            _load_wb2_norms(self.variables)

        # Attrs probed by AERIS_SP_ERA5_Data
        self.sst_mask = None
        self.read     = 0
        self.shape    = [self.channels, 721, 1440]
        self.samples  = 1

    def standardize_x(self, x, start_ch=0, end_ch=None, channel_dim=0):
        sh = [1, 1, 1]; sh[channel_dim] = -1
        return (x - self.x_means[start_ch:end_ch].reshape(sh)) / \
                    self.x_stds[start_ch:end_ch].reshape(sh)

    def unstandardize_x(self, x, start_ch=0, end_ch=None, channel_dim=0):
        sh = [1, 1, 1]; sh[channel_dim] = -1
        return x * self.x_stds[start_ch:end_ch].reshape(sh) + \
                   self.x_means[start_ch:end_ch].reshape(sh)

    def standardize_t(self, t, start_ch=0, end_ch=None, channel_dim=0):
        sh = [1, 1, 1]; sh[channel_dim] = -1
        return (t - self.t_means[start_ch:end_ch].reshape(sh)) / \
                    self.t_stds[start_ch:end_ch].reshape(sh)

    def unstandardize_t(self, t, start_ch=0, end_ch=None, channel_dim=0):
        sh = [1, 1, 1]; sh[channel_dim] = -1
        return t * self.t_stds[start_ch:end_ch].reshape(sh) + \
                   self.t_means[start_ch:end_ch].reshape(sh)

    def __len__(self):  return self.samples

    def __getitem__(self, idx):
        x_std = torch.from_numpy(self.standardize_x(self._ic)).float()
        labels = np.zeros(
            (self.rollout_count + 1, self.channels, self.lat_n, self.lon_n),
            dtype=np.float32,
        )
        toa_idx = self.variables.index("toa_incident_solar_radiation")
        labels[:, toa_idx] = self._toa_trajectory
        return x_std, torch.from_numpy(labels)


# ── Output helpers ───────────────────────────────────────────────────────────

PLEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

VAR_ALIASES = {
    "2m_temperature":           "2m_temperature",
    "10m_u_component_of_wind":  "10m_u_component_of_wind",
    "10m_v_component_of_wind":  "10m_v_component_of_wind",
    "mean_sea_level_pressure":  "mean_sea_level_pressure",
}


def _build_member_dataset(per_step: dict[str, np.ndarray],
                          lat, lon, lead_hours, cycle, ckpt_name):
    """Pack a single member's per-step arrays into an xr.Dataset."""
    cycle_dt = np.datetime64(cycle.rstrip("Z"))
    lead_sec  = np.array(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")

    data_vars = {}
    by_base: dict[str, list] = {}

    for v, arr in per_step.items():
        matched = False
        for lv in PLEVELS:
            if v.endswith(f"_{lv}"):
                base = v[: -len(f"_{lv}")]
                by_base.setdefault(base, []).append((lv, arr))
                matched = True
                break
        if not matched:
            data_vars[v] = (("lead_time", "lat", "lon"), arr)

    for base, lv_arrs in by_base.items():
        lv_arrs.sort(key=lambda t: t[0])
        data_vars[base] = (
            ("lead_time", "level", "lat", "lon"),
            np.stack([t[1] for t in lv_arrs], axis=1),
        )

    coords = {
        "lead_time": ("lead_time", lead_sec),
        "lat":       ("lat", lat),
        "lon":       ("lon", lon),
        "time":      cycle_dt,
    }
    if by_base:
        coords["level"] = ("level", np.array(PLEVELS, dtype=np.int32))

    return xr.Dataset(data_vars=data_vars, coords=coords,
                      attrs={"model": "AERIS-GB", "checkpoint": ckpt_name,
                             "cycle": cycle})


# ── Rollout ──────────────────────────────────────────────────────────────────

def rollout_and_collect(model, device, mesh, cfg, aeris_sp_data,
                        local_member_ids: list[int],
                        total_members: int,
                        rollout_steps: int,
                        diffusion_steps: int,
                        sigma_min: float,
                        sigma_max: float,
                        out_channels: int,
                        member_logs: dict[int, np.ndarray]):
    """
    DPMSolver++ Heun diffusion rollout for the local subset of ensemble members.

    Each DP group runs its own slice of the K members (round-robin assignment).
    Members are seeded by their global member ID so results are reproducible
    regardless of the DP topology used.

    Outputs are collected into `member_logs` (dict: global_member_id → array
    of shape (rollout_steps, out_channels, 721, 1440)) on the SP-rank-0 of
    each DP group. Non-SP-rank-0 ranks pass member_logs={}.
    """
    sp_group  = mesh.get_group(mesh_dim=1)
    rank      = mesh.get_rank()
    dp_rank   = mesh.get_local_rank(mesh_dim=0)
    sp_rank   = mesh.get_local_rank(mesh_dim=1)

    sigma_data = cfg.model.sigma_data
    sigma_min  = cfg.model.sigma_min if sigma_min == -1 else sigma_min
    sigma_max  = cfg.model.sigma_max if sigma_max == -1 else sigma_max

    # DPMSolver++ Heun noise schedule (vendor recipe)
    ramp = torch.linspace(0, 1, diffusion_steps, device=device)
    rho  = 10
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas  = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    t_steps = torch.atan(sigmas / sigma_data)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    gp  = aeris_sp_data.get_gp().to(device)
    lsm = aeris_sp_data.get_lsm().to(device)
    X, rad_T, X_orig = aeris_sp_data.get()
    X     = X.clone()
    rad_T = rad_T.to(device)

    if rank == 0:
        logger.info(f"IC ready; K={total_members} total, DP group {dp_rank} "
                    f"running {len(local_member_ids)} members: {local_member_ids}")

    for li, gmid in enumerate(local_member_ids):
        generator = torch.Generator(device=device)
        generator.manual_seed(gmid)   # seed by global ID — topology-stable

        if sp_rank == 0:
            logger.info(f"DP{dp_rank} member {li+1}/{len(local_member_ids)} "
                        f"(global id={gmid})")

        X_un    = aeris_sp_data.unstandardize_x(X.numpy(force=True), 0,
                                                 out_channels, channel_dim=2)
        condition = X.clone().to(device)
        torch.xpu.synchronize() if hasattr(torch, "xpu") else None

        for step in range(rollout_steps):
            t0 = time.time()

            condition_concat = torch.cat([
                condition,
                rad_T[step:step + 1],
                gp,
                lsm,
                rad_T[step + 1:step + 2],
            ], dim=2)

            latents = torch.randn(X.shape, generator=generator, device=device)
            x_t     = latents * sigma_data

            # ---- DPMSolver++ Heun loop ----
            for ds in range(diffusion_steps):
                s, t  = t_steps[ds], t_steps[ds + 1]
                delta = t - s

                model_in = torch.cat([x_t / sigma_data, condition_concat], dim=2).to(device)
                with torch.no_grad():
                    F_s = model(model_in, s.view(1))
                x_euler = x_t + delta * sigma_data * F_s

                if ds < diffusion_steps - 1:
                    model_in = torch.cat([x_euler / sigma_data, condition_concat], dim=2)
                    with torch.no_grad():
                        F_t = model(model_in.to(device), t.view(1))
                    x_t = x_t + delta * sigma_data * 0.5 * (F_s + F_t)
                else:
                    x_t = x_euler

            # ---- Update state ----
            Y_un = aeris_sp_data.unstandardize_t(x_t.numpy(force=True), 0,
                                                   out_channels, channel_dim=2)
            X_un = X_un + Y_un
            condition = torch.tensor(
                aeris_sp_data.standardize_x(X_un, 0, out_channels, channel_dim=2),
                device=condition.device, dtype=condition.dtype,
            )

            # ---- Gather tiled field → full (720, 1440) on SP-rank-0 ----
            sharded  = torch.from_numpy(X_un)
            gathered = aeris_sp_data.gather_tensor(sharded)
            if sp_rank == 0:
                full = rearrange(gathered, "b (h w) c -> b c h w", h=720, w=1440).numpy()
                pole_row  = full[:, :, 0:1, :]              # pad N pole (row 0 dup)
                full_721  = np.concatenate([pole_row, full], axis=2)
                member_logs[gmid][step] = full_721[0]
                dt = time.time() - t0
                logger.info(f"  step {step+1}/{rollout_steps}  [{dt:.1f}s]")

        torch.distributed.barrier(group=sp_group)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(prog="aeris_inference", description=__doc__)
    p.add_argument("--cycle",           required=True,
                   help="ISO 8601 UTC cycle, e.g. 2024-09-20T00:00:00Z")
    p.add_argument("--ic-source",       required=True, type=Path,
                   help="Path to the ARCO ERA5 IC NetCDF for this cycle")
    p.add_argument("--lead-hours",      type=int, default=240,
                   help="Forecast horizon in hours (default 240 = 10 days)")
    p.add_argument("--members",         type=int, default=50,
                   help="Ensemble size K (default 50)")
    p.add_argument("--member-offset",   type=int, default=0,
                   help="Global member ID offset for split-job runs "
                        "(e.g. --members 25 --member-offset 25 → IDs 25-49)")
    p.add_argument("--diffusion-steps", type=int, default=10,
                   help="DPMSolver++ Heun sub-steps per rollout step (default 10)")
    p.add_argument("--wp-x",            type=int, default=4,
                   help="Window-parallel tiles in X (default 4)")
    p.add_argument("--wp-y",            type=int, default=4,
                   help="Window-parallel tiles in Y (default 4)")
    p.add_argument("--sigma-min",       type=float, default=-1,
                   help="Override sigma_min (default: read from checkpoint config)")
    p.add_argument("--sigma-max",       type=float, default=-1,
                   help="Override sigma_max (default: read from checkpoint config)")
    p.add_argument("--checkpoint-dir",  required=True, type=Path,
                   help="Directory with checkpoint_PP0..PP9.pth + .hydra/config.yaml")
    p.add_argument("--output-dir",      required=True, type=Path,
                   help="Where to write member_000.nc … member_{K-1}.nc")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  rank=%(rank)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # mpi4py must be imported before torch.distributed.init_process_group
    from mpi4py import MPI
    rank       = int(MPI.COMM_WORLD.Get_rank())
    world_size = int(MPI.COMM_WORLD.Get_size())

    old_factory = logging.getLogRecordFactory()
    def _add_rank(*a, **k):
        rec = old_factory(*a, **k)
        rec.rank = rank
        return rec
    logging.setLogRecordFactory(_add_rank)

    if not args.ic_source.exists():
        if rank == 0: logger.error(f"IC not found: {args.ic_source}")
        return 2
    if not args.checkpoint_dir.exists():
        if rank == 0: logger.error(f"checkpoint dir not found: {args.checkpoint_dir}")
        return 2

    # ---- Distributed init ----
    device_count = acc.device_count()
    local_rank   = rank % device_count
    acc.set_device_index(local_rank)
    device  = torch.device(f"{acc.current_accelerator()}:{local_rank}")
    backend = "xccl" if str(acc.current_accelerator()) == "xpu" else "nccl"

    print(f"[init r={rank}/{world_size}] backend={backend}  device={device}", flush=True)
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank)
    print(f"[init r={rank}] process group ready", flush=True)

    SP = args.wp_x * args.wp_y   # WP-only mode: SP == WP
    if world_size % SP != 0:
        if rank == 0: logger.error(f"world_size {world_size} not divisible by SP={SP}")
        return 1
    DP = world_size // SP

    # DeviceMesh: rows=DP groups, cols=SP ranks within each group
    mesh = DeviceMesh(
        str(acc.current_accelerator()),
        [[i + j * SP for i in range(SP)] for j in range(DP)],
    )
    sp_rank         = mesh.get_local_rank(mesh_dim=1)
    sp_group        = mesh.get_group(mesh_dim=1)
    sp_group_ranks  = torch.distributed.get_process_group_ranks(sp_group)

    print(f"[init r={rank}] DP={DP} SP={SP} sp_rank={sp_rank}", flush=True)

    # Gloo group for intra-SP host-based collectives (vendor requirement)
    sp_group_gloo = torch.distributed.new_group(
        sp_group_ranks, backend="gloo", use_local_synchronization=True,
    )
    torch.distributed.barrier()
    torch.distributed.all_reduce(torch.tensor(1).to(device), group=sp_group)
    print(f"[init r={rank}] barriers done — proceeding to load checkpoint", flush=True)

    # ---- Checkpoint config ----
    hydra_cfg_path = args.checkpoint_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        if rank == 0: logger.error(f"missing {hydra_cfg_path}")
        return 5
    cfg = OmegaConf.load(hydra_cfg_path)

    train_vars   = list(cfg.data.dataset.variables)
    out_channels = len(train_vars)
    if rank == 0:
        logger.info(f"checkpoint: {args.checkpoint_dir.name}  "
                    f"vars={out_channels}  interval={cfg.model.interval}h  "
                    f"PP_stages={cfg.model.PP_stages}  "
                    f"WP={cfg.model.WP_X}x{cfg.model.WP_Y}")

    # ---- Dataset ----
    rollout_count = args.lead_hours // cfg.model.interval
    dataset = ARCOInferenceDataset(args.ic_source, train_vars,
                                   rollout_count=rollout_count)
    print(f"[init r={rank}] dataset ready  channels={dataset.channels}  "
          f"rollout_count={rollout_count}", flush=True)

    # ---- Model ----
    wp_dims = (args.wp_y, args.wp_x)
    assert SP == wp_dims[0] * wp_dims[1], \
        f"SP ({SP}) must equal WP_Y*WP_X ({wp_dims[0]*wp_dims[1]})"

    print(f"[init r={rank}] building LocalAERIS", flush=True)
    model = LocalAERIS(
        device_mesh=mesh,
        heads=cfg.model.heads,
        dim=cfg.model.dim,
        head_dim=cfg.model.head_dim,
        mlp_dim=cfg.model.mlp_dim,
        window_size=cfg.model.window_size,
        image_shape=(720, 1440),
        rope_base=10_000,
        sublayers=cfg.model.sublayers,
        sinusoidal_emb_max_period=cfg.model.sinusoidal_emb_max_period,
        n_layers=cfg.model.PP_stages,
        model_in_channels=out_channels * 2 + 4,
        model_out_channels=out_channels,
        SP=SP, sp_rank=sp_rank, wp_dims=wp_dims,
    ).to(device)
    print(f"[init r={rank}] model on device", flush=True)

    print(f"[init r={rank}] loading checkpoint shards", flush=True)
    convert_inference_checkpoint(str(args.checkpoint_dir), cfg.model.PP_stages,
                                 model, map_location=device)
    print(f"[init r={rank}] checkpoint loaded", flush=True)

    # ---- APE on CPU (avoids XPU allocator deadlock at DP≥8) ----
    # Compute the ~600 MB APE tensor on CPU, then move only the ~38 MB
    # per-rank slice to XPU. See CHANGES.md for details.
    print(f"[init r={rank}] computing APE on CPU", flush=True)
    ape_cpu_mod = model.input_stage.ape.to("cpu")
    ape_full    = ape_cpu_mod(
        torch.zeros((1, model.input_stage.model_in_channels, 721, 1440),
                    dtype=model.input_stage.data_dtype, device="cpu")
    )[:, :, 1:, :]   # strip N-pole duplicate
    model.input_stage.ape = model.input_stage.ape.to(device)

    WP_grid = np.arange(SP).reshape(wp_dims)
    my_y, my_x = tuple(i.item() for i in np.where(WP_grid == sp_rank))
    wp_y, ws_y, wp_x, ws_x = (wp_dims[0], model.window_size[0],
                               wp_dims[1], model.window_size[1])
    prev_ape = rearrange(
        ape_full,
        "b c (wp_y wcl_y ws_y) (wp_x wcl_x ws_x) -> b wp_y wp_x (wcl_y ws_y wcl_x ws_x) c",
        wp_y=wp_y, ws_y=ws_y, wp_x=wp_x, ws_x=ws_x,
    )
    model.input_stage.ape_generated = prev_ape[:, my_y, my_x].contiguous().to(device)
    del prev_ape, ape_full, ape_cpu_mod
    print(f"[init r={rank}] APE ready (per-rank shape="
          f"{tuple(model.input_stage.ape_generated.shape)})", flush=True)

    # ---- DataLoader + AERIS_SP_ERA5_Data wrapper ----
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            pin_memory=False, num_workers=0) if sp_rank == 0 else None
    aeris_sp_data = AERIS_SP_ERA5_Data(
        dataset, dataloader, out_channels, rollout_count, mesh, sp_group_gloo,
        wp_dims=wp_dims, image_shape=model.image_shape,
        window_size=model.window_size, device=device,
    )

    torch.distributed.barrier()
    if rank == 0:
        logger.info(f"ready; K={args.members} members, {rollout_count} rollout steps, "
                    f"{args.diffusion_steps} diffusion sub-steps")

    # ---- Member assignment ----
    dp_rank = mesh.get_local_rank(mesh_dim=0)
    offset  = args.member_offset
    local_member_ids = [offset + i for i in range(dp_rank, args.members, DP)]

    member_logs: dict[int, np.ndarray] = {}
    if sp_rank == 0:
        for gmid in local_member_ids:
            member_logs[gmid] = np.zeros(
                (rollout_count, out_channels, 721, 1440), dtype=np.float32)

    # ---- Rollout ----
    t_start = time.time()
    with torch.no_grad():
        rollout_and_collect(
            model, device, mesh, cfg, aeris_sp_data,
            local_member_ids=local_member_ids,
            total_members=args.members,
            rollout_steps=rollout_count,
            diffusion_steps=args.diffusion_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            out_channels=out_channels,
            member_logs=member_logs,
        )
    if rank == 0:
        logger.info(f"rollout complete in {time.time()-t_start:.1f}s")

    # ---- Parallel write: each DP-root rank writes its members ----
    if sp_rank != 0:
        # Non-root ranks within each DP group have nothing to write.
        logger.debug(f"rank {rank} (sp_rank={sp_rank}) exiting after rollout")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name    = args.checkpoint_dir.name
    ic_block     = dataset._ic[None, :out_channels, :, :].astype(np.float32)
    lead_hrs     = list(range(0, args.lead_hours + 1, cfg.model.interval))

    print(f"[write r={rank}] dp_rank={dp_rank} writing members {local_member_ids}",
          flush=True)

    for k in local_member_ids:
        full_k     = np.concatenate([ic_block, member_logs[k]], axis=0)  # (T+1, C, H, W)
        per_step_k = {VAR_ALIASES.get(vn, vn): full_k[:, ch].astype(np.float32)
                      for ch, vn in enumerate(dataset.variables[:out_channels])}
        ds_k = _build_member_dataset(per_step_k, dataset.lat, dataset.lon,
                                     lead_hrs, args.cycle, ckpt_name)
        out_path = args.output_dir / f"member_{k:03d}.nc"
        enc = {v: {"zlib": True, "complevel": 1} for v in ds_k.data_vars}
        ds_k.to_netcdf(out_path, engine="netcdf4", encoding=enc)
        sz = out_path.stat().st_size / 1e6
        print(f"[write r={rank}] wrote {out_path.name} ({sz:.0f} MB)", flush=True)

    logger.info(f"rank {rank} done ({len(local_member_ids)} members written)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:
        logger.exception("aeris_inference failed")
        sys.exit(3)
