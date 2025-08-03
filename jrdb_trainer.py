#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotionBEV training entry-point for the JRDB LiDAR-only semantic / motion
segmentation task.

It follows the overall structure of `train_SemanticKITTI.py` but swaps in
the JRDB dataloader and default paths so you can launch training with:

    python jrdb_trainer.py \
        --arch_cfg config/MotionBEV-jrdb.yaml \
        --data_cfg config/jrdb-motion.yaml

Both YAMLs can mirror the corresponding SemanticKITTI ones; only the label
mappings and JRDB folder locations differ.
"""
from dataloader.jrdb_dataset import (
    JRDB,
    voxel_dataset,
    spherical_dataset,
    collate_fn_BEV,
)
from dataloader.dataset import get_JRDB_label_name

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from icecream import ic
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Network + utils
from network.CA_BEV_Unet import CA_Unet
from network.A_BEV_Unet import BEV_Unet
from network.ptBEVnet import ptBEVnet
from utils.lovasz_losses import lovasz_softmax
from utils.log_util import get_logger, make_log_dir
from config.config import load_config_data
from utils.warmupLR import warmupLR
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
mp.set_start_method('spawn', force=True)

# ────────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=UserWarning)


# ────────────────────────────────────────────────────────────────────────────────
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bincount = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bincount[: n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8  # avoid 0-div
    )


def fast_hist_crop(output, target, valid_ids):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(valid_ids) + 1)
    hist = hist[valid_ids, :][:, valid_ids]
    return hist


# ────────────────────────────────────────────────────────────────────────────────
def train(arch_cfg_path: str, data_cfg_path: str):
    cfg = load_config_data(arch_cfg_path)
    ic(cfg)

    data_cfg = cfg["data_loader"]
    model_cfg = cfg["model_params"]
    train_cfg = cfg["train_params"]
    fea_compre = model_cfg["grid_size"][2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on", device)

    # ─────────── Data & labels ───────────
    unique_label, unique_label_str, _ = get_JRDB_label_name(data_cfg_path)

    if data_cfg["dataset_type"] == "polar":
        fea_dim, circular_padding = 8, True
    elif data_cfg["dataset_type"] == "traditional":
        fea_dim, circular_padding = 7, False
    else:
        raise NotImplementedError

    # Datasets
    train_pt_dataset = JRDB(
        data_cfg_path=data_cfg_path,
        data_root=Path(data_cfg["data_path"]),
        split="train",
        return_ref=data_cfg["return_ref"],
        residual=data_cfg["residual"],
        residual_root=data_cfg["residual_path"],
        drop_few_static_frames=data_cfg["drop_few_static_frames"],
    )
    val_pt_dataset = JRDB(
        data_cfg_path=data_cfg_path,
        data_root=Path(data_cfg["data_path"]),
        split="val",
        return_ref=data_cfg["return_ref"],
        residual=data_cfg["residual"],
        residual_root=data_cfg["residual_path"],
        drop_few_static_frames=False,
    )

    ds_cls = spherical_dataset if data_cfg["dataset_type"] == "polar" else voxel_dataset
    train_dataset = ds_cls(
        train_pt_dataset,
        grid_size=model_cfg["grid_size"],
        rotate_aug=data_cfg["rotate_aug"],
        flip_aug=data_cfg["flip_aug"],
        transform_aug=data_cfg.get("transform_aug", False),
        fixed_volume_space=data_cfg["fixed_volume_space"],
    )
    val_dataset = ds_cls(
        val_pt_dataset,
        grid_size=model_cfg["grid_size"],
        fixed_volume_space=data_cfg["fixed_volume_space"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        collate_fn=collate_fn_BEV,
        shuffle=data_cfg["shuffle"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        collate_fn=collate_fn_BEV,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # ─────────── Model ───────────
    bev_backbone = (
        CA_Unet
        if model_cfg["use_co_attention"]
        else BEV_Unet
    )(
        n_class=len(unique_label),
        n_height=fea_compre,
        residual=data_cfg["residual"],
        input_batch_norm=model_cfg["use_norm"],
        dropout=model_cfg["dropout"],
        circular_padding=circular_padding,
    )
    model = ptBEVnet(
        bev_backbone,
        grid_size=model_cfg["grid_size"],
        fea_dim=fea_dim,
        ppmodel_init_dim=model_cfg["ppmodel_init_dim"],
        kernal_size=1,
        fea_compre=fea_compre,
        residual_ch=data_cfg["residual"],
    ).to(device)

    # load pretrained weights if specified
    pretrain = train_cfg["model_load_path"]
    if pretrain and os.path.exists(pretrain):
        print("Loading pretrained weights from", pretrain)
        model.load_state_dict(torch.load(pretrain, map_location=device))

    # ─────────── Optimizer / LR ───────────
    opt_type = train_cfg["optimizer"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]

    if opt_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=train_cfg["momentum"],
            weight_decay=wd,
        )
    else:
        raise NotImplementedError

    steps_per_epoch = len(train_loader)
    scheduler = warmupLR(
        optimizer=optimizer,
        lr=lr,
        warmup_steps=int(train_cfg["wup_epochs"] * steps_per_epoch),
        momentum=train_cfg.get("momentum", 0.9),
        decay=train_cfg["lr_decay"] ** (1 / steps_per_epoch),
    )

    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    save_root = make_log_dir(arch_cfg_path, data_cfg_path, train_cfg["name"])
    logger = get_logger(Path(save_root, "train.log"))

    # ─────────── Train loop ───────────
    best_miou, best_loss = 0.0, float("inf")
    global_iter = 0
    model.train()

    for epoch in range(train_cfg["max_num_epochs"]):
        epoch_loss = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            vox_label, grid, pt_lab, pt_fea = batch
            pt_fea = [x.to(device) for x in pt_fea]
            grid = [x.to(device) for x in grid]
            vox_label = vox_label.to(device)

            optimizer.zero_grad()
            vox_out, _ = model(pt_fea, grid, device)
            loss = (
                lovasz_softmax(torch.nn.functional.softmax(vox_out), vox_label, ignore=255)
                + ce_loss(vox_out, vox_label)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.item())
            global_iter += 1

            if global_iter % train_cfg["checkpoint_every_n_steps"] == 0:
                pbar.set_postfix(loss=np.mean(epoch_loss[-20:]))

            # ── Validation ──
            if (
                train_cfg["eval_every_n_steps"] > 0
                and global_iter % train_cfg["eval_every_n_steps"] == 0
            ):
                model.eval()
                hist, val_losses = [], []
                with torch.no_grad():
                    for v_vox_label, v_grid, v_pt_lab, v_pt_fea in val_loader:
                        v_pt_fea = [x.to(device) for x in v_pt_fea]
                        v_grid = [x.to(device) for x in v_grid]
                        v_vox_label = v_vox_label.to(device)

                        v_vox_out, _ = model(v_pt_fea, v_grid, device)
                        v_loss = (
                            lovasz_softmax(
                                torch.nn.functional.softmax(v_vox_out).detach(),
                                v_vox_label,
                                ignore=255,
                            )
                            + ce_loss(v_vox_out.detach(), v_vox_label)
                        )
                        val_losses.append(v_loss.item())

                        preds = torch.argmax(v_vox_out, dim=1).cpu().numpy()
                        for idx, g in enumerate(v_grid):
                            hist.append(
                                fast_hist_crop(
                                    preds[idx, g[:, 0], g[:, 1], g[:, 2]],
                                    v_pt_lab[idx],
                                    unique_label,
                                )
                            )

                miou = np.nanmean(per_class_iu(sum(hist))) * 100
                logger.info(f"Validation mIoU: {miou:.2f} - Loss: {np.mean(val_losses):.4f}")

                if miou > best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), Path(save_root, "best_miou.pt"))
                if np.mean(val_losses) < best_loss:
                    best_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), Path(save_root, "best_loss.pt"))

                model.train()

        # end epoch
        logger.info(f"Epoch {epoch} finished. Avg loss: {np.mean(epoch_loss):.4f}")

    logger.info("Training complete. Best mIoU %.2f, best loss %.4f", best_miou, best_loss)




# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MotionBEV JRDB trainer")
    parser.add_argument("--arch_cfg", required=True, help="Architecture YAML")
    parser.add_argument("--data_cfg", required=True, help="Data YAML with label map")
    args = parser.parse_args()
    train(args.arch_cfg, args.data_cfg)