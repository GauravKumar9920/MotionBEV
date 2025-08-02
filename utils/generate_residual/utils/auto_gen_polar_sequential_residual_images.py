from pathlib import Path
import argparse
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import deque

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Assume PolarGrid and cart2polar_grids are defined elsewhere or imported

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
            T = T_i_inv @ pose_j                          # 4×4
            pts_j_h = xp.concatenate((pts_j[:,:3], xp.ones((pts_j.shape[0],1))),1).T
            pts_j_tf= (T @ pts_j_h).T[:,:3]
            occ_j   = cart2polar_grids(xp.asnumpy(pts_j_tf), grid)
            past_occ[j] = xp.asarray(occ_j)

        # voxelise current frame
        occ_i = cart2polar_grids(pts_i[:,:3], grid)
        occ_i = xp.asarray(occ_i)

        if len(past_occ) == window:
            past_stack   = xp.logical_or.reduce(past_occ)
            recent_stack = xp.logical_or.reduce(list(past_occ)[1:]+[occ_i])
            delta        = recent_stack.astype(xp.int8) - past_stack.astype(xp.int8)
            res_vol      = xp.clip(delta, -1, 1).astype(xp.int8)
            np.save(out_dir / f"{fi:06d}.npy",
                    xp.asnumpy(res_vol) if use_gpu else res_vol)

        # enqueue current scan
        past_q.append((pts_i, pose_i))
        past_occ.append(occ_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("sequence")
    ap.add_argument("--residual", type=int, default=4,
                    help="sliding‑window length (N frames)")
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("residual"))
    args = ap.parse_args()

    reader = PCDSequence(args.root, args.sequence)
    grid_size = (64, 512)  # example grid size, define accordingly
    out_dir = args.out / args.sequence
    build_residual(reader, grid_size, args.residual, out_dir, args.use_gpu and HAS_CUPY)


if __name__ == "__main__":
    main()
