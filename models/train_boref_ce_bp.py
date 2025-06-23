
"""
train_boref_ce_bp_patch.py


Patch-based training for BoRefAttnNet using cross-entropy and adaptive multi-class boundary penalty.
I used 4 Nvidia L4 GPUs for computation 
"""

import os
import csv
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# FreedSurfer dataset
from freedsurfer_data_loader import FreedSurferData
# BoRefAttnNet architecture
from boref_net import BoRefAttnNet
# CONFIG

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 1
EPOCHS = 150
LEARNING_RATE = 1e-4
PATIENCE = 10

CHECKPOINT_DIR = "boref_ce_bp_patch_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE = "boref_ce_bp_patch_training.log"
CSV_LOG_FILE = "boref_ce_bp_patch_training.csv"
DATA_SPLIT_FILE = "data_split.json"

PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)

BOUNDARY_WEIGHT = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open(CSV_LOG_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["Epoch", "SubPatchStep", "TrainLoss", "ValLoss"])

logging.info("=== Starting BoRefAttnNet training (CE + boundary penalty) ===")

def worker_init_fn(worker_id):
    s_ = SEED + worker_id
    np.random.seed(s_)
    random.seed(s_)

#  DataLoaders 
with open(DATA_SPLIT_FILE) as f:
    splits = json.load(f)

train_ds = FreedSurferData(
    data_root="/home/osasu/data_processed_remapped",
    subject_list=splits["train"],
    crop_size=None,
    augment=True
)

val_ds = FreedSurferData(
    data_root="/home/osasu/data_processed_remapped",
    subject_list=splits["val"],
    crop_size=None,
    augment=False
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#  Model & Loss
model = BoRefAttnNet(
    n_channels=1,
    n_classes=6,
    bilinear=True,
    gn_groups=8,
    use_checkpoint=True
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
ce_loss_func = nn.CrossEntropyLoss()

# Introducing my loss approach 
def boundary_loss_multi_class(logits, target, boundary_weight=2.0):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).float()

    boundary_map = torch.zeros_like(preds)
    boundary_map[:, 1:, :, :] += (preds[:, 1:, :, :] - preds[:, :-1, :, :]).abs()
    boundary_map[:, :, 1:, :] += (preds[:, :, 1:, :] - preds[:, :, :-1, :]).abs()
    boundary_map[:, :, :, 1:] += (preds[:, :, :, 1:] - preds[:, :, :, :-1]).abs()

    ce_per_voxel = F.cross_entropy(logits, target.long(), reduction='none')
    boundary_term = boundary_map * ce_per_voxel

    return boundary_weight * boundary_term.mean()

# Total loss combining both CE and adaptive boundary penalty
def total_loss(seg_logits, target, alpha=1.0, beta=BOUNDARY_WEIGHT):
    loss_ce = ce_loss_func(seg_logits, target.long())
    bound_loss = boundary_loss_multi_class(seg_logits, target)
    return alpha * loss_ce + beta * bound_loss

# Training Loop
def sliding_window_subvolumes(vol, window_size, stride):
    C, D, H, W = vol.shape
    wD, wH, wW = window_size
    sD, sH, sW = stride
    for d0 in range(0, D - wD + 1, sD):
        for h0 in range(0, H - wH + 1, sH):
            for w0 in range(0, W - wW + 1, sW):
                yield (d0,h0,w0), vol[:, d0:d0+wD, h0:h0+wH, w0:w0+wW]

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    epoch_loss_sum = 0.0
    subject_count  = 0
    subpatch_step  = 0

    for sample in loader:
        full_img = sample["t1"].squeeze(0).to(device)  # (1, D,H,W)
        full_lbl = sample["label"].squeeze(0).to(device)

        subpatch_loss_sum = 0.0
        subpatch_count    = 0

        for (d0,h0,w0), patch_img in sliding_window_subvolumes(full_img, PATCH_SIZE, STRIDE):
            patch_lbl = full_lbl[
                d0 : d0 + PATCH_SIZE[0],
                h0 : h0 + PATCH_SIZE[1],
                w0 : w0 + PATCH_SIZE[2]
            ]
            patch_img = patch_img.unsqueeze(0)  # (B=1,1,PD,PH,PW)
            patch_lbl = patch_lbl.unsqueeze(0)  # (1,PD,PH,PW)

            optimizer.zero_grad()
            with autocast():
                seg_logits, boundary_logits = model(patch_img)
                loss = total_loss(seg_logits, patch_lbl, alpha=1.0, beta=BOUNDARY_WEIGHT)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            subpatch_loss_sum += loss.item()
            subpatch_count    += 1
            subpatch_step     += 1

            if subpatch_step % 50 == 0:
                msg = f"Epoch[{epoch}] Subpatch[{subpatch_step}]: trainLoss={loss.item():.4f}"
                print(msg)
                logging.info(msg)
                with open(CSV_LOG_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, subpatch_step, loss.item(), None])

        if subpatch_count>0:
            subject_loss    = subpatch_loss_sum / subpatch_count
            epoch_loss_sum += subject_loss
            subject_count  += 1

    if subject_count == 0:
        return 0.0
    else:
        return epoch_loss_sum / subject_count

def validate_one_epoch(model, loader):
    model.eval()
    val_loss_sum = 0.0
    subj_count   = 0

    with torch.no_grad():
        for sample in loader:
            full_img = sample["t1"].squeeze(0).to(device)
            full_lbl = sample["label"].squeeze(0).to(device)

            sub_loss_sum = 0.0
            sub_count    = 0

            for (d0,h0,w0), patch_img in sliding_window_subvolumes(full_img, PATCH_SIZE, STRIDE):
                patch_lbl = full_lbl[
                    d0 : d0+PATCH_SIZE[0],
                    h0 : h0+PATCH_SIZE[1],
                    w0 : w0+PATCH_SIZE[2]
                ]
                patch_img = patch_img.unsqueeze(0)
                patch_lbl = patch_lbl.unsqueeze(0)

                with autocast():
                    seg_logits, boundary_logits = model(patch_img)
                    vloss = total_loss(seg_logits, patch_lbl, alpha=1.0, beta=BOUNDARY_WEIGHT)

                sub_loss_sum += vloss.item()
                sub_count    += 1

            if sub_count>0:
                val_loss_sum += (sub_loss_sum / sub_count)
                subj_count   += 1

    if subj_count == 0:
        return 0.0
    else:
        return val_loss_sum / subj_count

# Main Loop
best_val_loss = None
early_stop_counter = 0

for ep in range(1, EPOCHS+1):
    tr_loss = train_one_epoch(model, train_loader, optimizer, ep)
    val_loss= validate_one_epoch(model, val_loader)

    msg = f"Epoch[{ep}] trainLoss={tr_loss:.4f}, valLoss={val_loss:.4f}"
    print(msg)
    logging.info(msg)
    with open(CSV_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ep, None, tr_loss, val_loss])

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"boref_epoch_{ep}.pth")
    torch.save(model.state_dict(), ckpt_path)
    logging.info(f"Checkpoint saved => {ckpt_path}")

    if best_val_loss is None or val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"No improvement in val loss for {PATIENCE} epochs => stop at epoch {ep}")
            logging.info(f"Early stopping at epoch {ep}")
            break

print("=== BoRefAttnNet training complete (CE+Boundary)! ===")
logging.info("BoRefAttnNet (CE+BP) done.")
