import yaml
from torch.utils.data import ConcatDataset
import os
import numpy as np
import open3d as o3d
from pathlib import Path
import json
from utils.log_util import get_logger

# Dataset helpers from core dataset module
from dataloader.dataset import (
    voxel_dataset,
    spherical_dataset,
    collate_fn_BEV,
    get_JRDB_label_name,
)

LOG = get_logger("JRDB_dataset")

class PCDSequence:
    """
    Lightweight loader for JRDB upper-velodyne scans with per-frame 3D bounding box labels.
    * LiDAR: root/pointclouds/upper_velodyne/<seq>/*.pcd
    * Labels: root/labels/labels_3d/<seq>.json
    """
    def __init__(self, root: Path, sequence_id: str):
        self.seq_dir = root / "pointclouds" / "upper_velodyne" / sequence_id
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {self.seq_dir}")
        # Gather point cloud frames
        self.frames = sorted(self.seq_dir.glob("*.pcd"))
        if not self.frames:
            raise RuntimeError(f"No .pcd files in {self.seq_dir}")
        # Load per-frame bounding-box labels from JSON
        label_file = root / "labels" / "labels_3d" / f"{sequence_id}.json"
        if not label_file.is_file():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        with open(label_file, 'r', encoding='utf-8') as lf:
            label_json = json.load(lf)
        self.frame_labels = label_json['labels']

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(self.frames[idx]))
        pts = np.asarray(pcd.points, dtype=np.float32)
        # Ensure intensity channel is present (dummy if missing)
        if pts.shape[1] == 3:
            pts = np.hstack((pts, np.zeros((pts.shape[0], 1), dtype=np.float32)))

        # Initialize all points as static (label 0)
        labels = np.zeros((pts.shape[0],), dtype=np.uint8)

        # Assign moving label (1) for points inside any bounding box
        file_name = self.frames[idx].name  # e.g. "000000.pcd"
        for box_info in self.frame_labels.get(file_name, []):
            # Skip boxes marked no_eval
            if box_info.get('attributes', {}).get('no_eval', False):
                continue
            # Box parameters
            b = box_info['box']
            cx, cy, cz = b['cx'], b['cy'], b['cz']
            l, w, h = b['l'], b['w'], b['h']
            rot = b['rot_z']
            # Transform points into box coordinate frame
            local = pts[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
            cos_r, sin_r = np.cos(-rot), np.sin(-rot)
            xp = local[:, 0] * cos_r - local[:, 1] * sin_r
            yp = local[:, 0] * sin_r + local[:, 1] * cos_r
            zp = local[:, 2]
            # Check box inclusion
            mask = (np.abs(xp) <= l / 2) & (np.abs(yp) <= w / 2) & (np.abs(zp) <= h / 2)
            labels[mask] = 1

        return pts, labels.reshape(-1, 1)


class JRDB(ConcatDataset):
    """
    JRDB loader that aggregates multiple PCDSequence instances based on a split file.
    Inherits from torch.utils.data.ConcatDataset.
    """
    def __init__(self, data_cfg_path, data_root, split,
                 return_ref, residual, residual_root, drop_few_static_frames):
        # Load data config to get split_file path
        with open(data_cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        split_file = cfg['data_loader']['split_file']
        # Read sequences for this split
        seqs = []
        with open(split_file, 'r') as sf:
            for line in sf:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tag, seq_id = line.split()
                if tag == split:
                    seqs.append(seq_id)
        # Build list of PCDSequence datasets
        datasets = [PCDSequence(Path(data_root), seq_id) for seq_id in seqs]
        # Initialize ConcatDataset
        super().__init__(datasets)
