#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate polar residual volumes for JRDB so MotionBEV can be trained.
The 'residual' parameter (default 8) means we accumulate the last N-1
range sweeps and store them as delta log-occupancy along the range axis,
identical to MotionBEV for SemanticKITTI.

Output tree (created automatically):

    <out_root>/polar_residual/<sequence>/<frame_idx>.npy
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import deque

# Optional GPU acceleration ────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

 # Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure project root (one level up) is on Python path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Remove JRDB import; add Open3D and glob imports
import open3d as o3d
import glob
# Minimal PCDSequence loader for .pcd files with pose support
class PCDSequence:
    """
    Lightweight loader for JRDB upper‑velodyne scans **with pose support**.
    * LiDAR: root/pointclouds/upper_velodyne/<seq>/000000.pcd
    * Poses: root/poses/<seq>_poses_kitti.txt  (one 3×4 row per frame)
    """
    def __init__(self, root: Path, sequence_id: str):
        self.seq_dir  = root / "pointclouds" / "upper_velodyne" / sequence_id
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(self.seq_dir)
        self.frames = sorted(self.seq_dir.glob("*.pcd"))
        if len(self.frames) == 0:
            raise RuntimeError(f"No .pcd in {self.seq_dir}")
        # ── JRDB stores *all* poses for the sequence in one file
        #     <JRDB root>/poses/<sequence>_poses_kitti.txt (4×4 row‑major per line, first row dropped)
        pose_file = root / "poses" / f"{sequence_id}_poses_kitti.txt"
        if not pose_file.is_file():
            raise FileNotFoundError(pose_file)

        raw = np.loadtxt(pose_file, dtype=np.float32).reshape(-1, 12)      # (N,12)
        if raw.shape[0] != len(self.frames):
            raise ValueError(f"#poses {raw.shape[0]} ≠ #scans {len(self.frames)} for {sequence_id}")

        self.poses = []
        for row in raw:
            T = np.eye(4, dtype=np.float32)
            T[:3, :4] = row.reshape(3, 4)
            self.poses.append(T)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # point cloud
        pcd = o3d.io.read_point_cloud(str(self.frames[idx]))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[1] == 3:
            pts = np.hstack((pts, np.zeros((pts.shape[0],1), dtype=np.float32)))
        return pts, self.poses[idx]
# from utils.lidar_projection   import cart2polar_grids, PolarGrid  # helper from MotionBEV

# ─────────────────────────────────────────────────────────────────────────
# Lightweight polar-grid utilities (drop-in replacement for utils.lidar_projection)

class PolarGrid:
    """Discretises XYZ points into (range, azimuth, height) voxels."""
    def __init__(self, n_r, n_a, n_z, r_max=80.0, z_min=-3.0, z_max=3.0):
        self.n_r, self.n_a, self.n_z = n_r, n_a, n_z
        self.r_max = r_max
        self.z_min, self.z_max = z_min, z_max
        self.dr = r_max / n_r
        self.dtheta = 2 * np.pi / n_a
        self.dz = (z_max - z_min) / n_z

def cart2polar_grids(points_xyz, grid: PolarGrid):
    """
    Rasterise an (N,3) point cloud into a boolean occupancy tensor shaped
    (n_r, n_a, n_z) using the provided PolarGrid description.
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r      = np.sqrt(x ** 2 + y ** 2)
    theta  = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)

    keep   = (r < grid.r_max) & (z >= grid.z_min) & (z < grid.z_max)
    if not np.any(keep):
        return np.zeros((grid.n_r, grid.n_a, grid.n_z), dtype=bool)

    r_idx  = np.floor(r[keep]      / grid.dr    ).astype(int)
    a_idx  = np.floor(theta[keep]  / grid.dtheta).astype(int)
    z_idx  = np.floor((z[keep] - grid.z_min) / grid.dz).astype(int)

    r_idx  = np.clip(r_idx, 0, grid.n_r - 1)
    a_idx  = np.clip(a_idx, 0, grid.n_a - 1)
    z_idx  = np.clip(z_idx, 0, grid.n_z - 1)

    occ    = np.zeros((grid.n_r, grid.n_a, grid.n_z), dtype=bool)
    occ[r_idx, a_idx, z_idx] = True
    return occ
# ─────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------

def build_residual(seq_reader, grid_size, window, out_dir, use_gpu=False):
    """
    Generate residual as height‑delta between
      * past  (frames [i-window, …, i-1])
      * recent(frames [i-window+1, …, i])
    All past scans are motion‑compensated into the current frame
    using provided poses.
    """
    os.makedirs(out_dir, exist_ok=True)
    grid = PolarGrid(*grid_size)
    xp   = cp if use_gpu else np
    print(f"[INFO] Using {'GPU/CuPy' if use_gpu and HAS_CUPY else 'CPU/Numpy'}")

    seq_name  = Path(out_dir).name
    past_q    = deque(maxlen=window)
    past_occ  = deque(maxlen=window)

    for fi in tqdm(range(len(seq_reader)), desc=seq_name, ncols=90):
        pts_i, pose_i = seq_reader[fi]
        T_i_inv       = xp.asarray(np.linalg.inv(pose_i))

        # compensate & voxelise each stored past cloud in queue
        for j, (pts_j, pose_j) in enumerate(past_q):
            pose_j_xp = pose_j if isinstance(pose_j, xp.ndarray) else xp.asarray(pose_j)
            T = T_i_inv @ pose_j_xp                       # 4×4
            pts_j_h = xp.concatenate((pts_j[:,:3], xp.ones((pts_j.shape[0],1))),1).T
            pts_j_tf= (T @ pts_j_h).T[:,:3]
            occ_j   = cart2polar_grids(xp.asnumpy(pts_j_tf), grid)
            past_occ[j] = xp.asarray(occ_j)

        # voxelise current frame
        occ_i = cart2polar_grids(pts_i[:,:3], grid)
        occ_i = xp.asarray(occ_i)

        if len(past_occ) == window:
            past_stack   = xp.any(xp.stack(list(past_occ), axis=0), axis=0)
            recent_stack = xp.any(xp.stack(list(past_occ)[1:] + [occ_i], axis=0), axis=0)
            delta        = recent_stack.astype(xp.int8) - past_stack.astype(xp.int8)
            res_vol      = xp.clip(delta, -1, 1).astype(xp.int8)
            np.save(out_dir / f"{fi:06d}.npy",
                    xp.asnumpy(res_vol) if use_gpu else res_vol)

        # enqueue current scan (convert both pts and pose to xp backend if GPU mode)
        pts_i_xp  = xp.asarray(pts_i)  if use_gpu else pts_i
        pose_i_xp = xp.asarray(pose_i) if use_gpu else pose_i
        past_q.append((pts_i_xp, pose_i_xp))
        past_occ.append(occ_i)

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jrdb_root",   default="/scratch/aadith_warrier/JRDB", help="JRDB root directory")
    ap.add_argument("--split_file",  default="config/jrdb_split.txt")
    ap.add_argument("--out_root",    default="/scratch/aadith_warrier/JRDB_residual")
    ap.add_argument("--grid_size",   default="480,360,32",
                    help="comma-sep ‘W,H,Z’ (must match MotionBEV config)")
    ap.add_argument("--residual", type=int, default=4,
                    help="sliding‑window length (N frames)")
    ap.add_argument("--use_gpu", action="store_true",
                    help="If set and CuPy is available, compute residual volumes on GPU.")
    ap.add_argument("--data_cfg", default="config/jrdb_motion.yaml",
                    help="Path to the JRDB data loader YAML (used by JRDB)")
    args = ap.parse_args()

    data_root = Path(args.jrdb_root)  # pass the real JRDB root

    grid_size = [int(x) for x in args.grid_size.split(",")]
    seqs = []
    with open(args.split_file) as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            split_type, seq = line.split()
            seqs.append((split_type, seq))

    for split_type, seq in seqs:
        # Use minimal PCDSequence loader
        reader = PCDSequence(data_root, seq)
        out_dir = Path(args.out_root, "polar_residual", seq)
        if out_dir.exists() and len(reader) > 0 and len(list(out_dir.glob("*.npy"))) == len(reader):
            print(f"[✓] Residuals already exist for {seq}")
            continue
        build_residual(reader, grid_size, args.residual, out_dir, args.use_gpu and HAS_CUPY)

if __name__ == "__main__":
    main()