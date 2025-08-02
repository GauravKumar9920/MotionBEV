# Wrapper for JRDB dataset - adapting SemKITTI for JRDB
# this is a  test from the terminal
from .dataset import (
    SemKITTI,
    voxel_dataset,
    spherical_dataset,
    collate_fn_BEV,
)
from pathlib import Path

class JRDB(SemKITTI):
    """Wrapper to adapt SemKITTI dataset class for JRDB"""
    def __init__(self, data_cfg_path, data_root, split, return_ref, residual, residual_root, drop_few_static_frames):
        # Convert JRDB parameters to SemKITTI parameters
        super().__init__(
            data_config_path=data_cfg_path,
            data_path=str(data_root),
            imageset=split,
            return_ref=return_ref,
            residual=residual,
            residual_path=str(residual_root),
            drop_few_static_frames=drop_few_static_frames,
        )
