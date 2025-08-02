#!/usr/bin/env python3
import os, copy
import multiprocessing as mp          #  NEW
from multiprocessing import cpu_count
from numba import cuda                 #  NEW
from auto_gen_polar_sequential_residual_images import load_yaml, process_one_seq

# ---- helper that runs one sequence then closes the CUDA context ----
def _worker(cfg):
    try:
        process_one_seq(cfg)
    finally:                           # Always run, even on crash
        try:
            cuda.close()               # <- releases GPU context cleanly
        except Exception:
            pass

def main():
    mp.set_start_method("spawn", force=True)      #  NEW  (one-time)

    # ---------- build the per-sequence config list ----------
    cfg_path = os.path.join(os.path.dirname(__file__),
                            "../config/data_preparing_polar_sequential.yaml")
    base = load_yaml(cfg_path)
    scan_root, out_root = base["scan_folder"], base["residual_image_folder"]

    seq_cfgs = []
    for seq in range(22):
        s = f"{seq:02d}"
        cfg = copy.deepcopy(base)
        cfg["scan_folder"]           = os.path.join(scan_root, "sequences", s, "velodyne")
        cfg["pose_file"]             = os.path.join(scan_root, "sequences", s, "poses.txt")
        cfg["calib_file"]            = os.path.join(scan_root, "sequences", s, "calib.txt")
        cfg["residual_image_folder"] = os.path.join(out_root,  s, "residual_images")
        seq_cfgs.append(cfg)

    n_workers = min(cpu_count(), len(seq_cfgs))
    print(f"[MotionBEV] generating residual images with {n_workers} workers…")

    # ---- use spawn context so every worker is a fresh interpreter ----
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        pool.map(_worker, seq_cfgs)

    print("[MotionBEV] all sequences finished.")

if __name__ == "__main__":
    main()