
"""
train_boref_ce_patch.py

Patch-based training for BoRefAttnNet using cross-entropy (CE) only.
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
import torch.optim as optim

# FreedSurfer dataset
from freedsurfer_data_loader import FreedSurferData
# BoRefAttnNet architecture
from boref_net import BoRefAttnNet

# CONFIG 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE     = 1
EPOCHS         = 150
LEARNING_RATE  = 1e-4
PATIENCE       = 10

CHECKPOINT_DIR = "boref_ce_patch_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE       = "boref_ce_patch_training.log"
CSV_LOG_FILE   = "boref_ce_patch_training.csv"
DATA_SPLIT_FILE= "data_split.json"

PATCH_SIZE     = (128,128,128)
STRIDE         = (64,64,64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

with open(CSV_LOG_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["Epoch","SubPatchStep","TrainLoss","ValLoss"])

logging.info("=== Starting patch-based training for BoRefAttnNet (CE-only) ===")

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
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#  Model & Loss
model = BoRefAttnNet(
    n_channels=1,
    n_classes=6,
    bilinear=True,
    gn_groups=8,
    use_checkpoint=True
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
ce_loss   = nn.CrossEntropyLoss()

# Helper
def sliding_window_subvolumes(vol, window_size, stride):
    C, D, H, W = vol.shape
    wD, wH, wW = window_size
    sD, sH, sW = stride
    for d0 in range(0, D - wD + 1, sD):
        for h0 in range(0, H - wH + 1, sH):
            for w0 in range(0, W - wW + 1, sW):
                yield (d0, h0, w0), vol[:, d0:d0+wD, h0:h0+wH, w0:w0+wW]

#  Training Loop 
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    count = 0
    step = 0

    for sample in loader:
        img, lbl = sample["t1"].squeeze(0).to(device), sample["label"].squeeze(0).to(device)
        sub_loss = 0
        patches = 0

        for (d0, h0, w0), patch_img in sliding_window_subvolumes(img, PATCH_SIZE, STRIDE):
            patch_lbl = lbl[
                d0:d0+PATCH_SIZE[0],
                h0:h0+PATCH_SIZE[1],
                w0:w0+PATCH_SIZE[2]
            ].unsqueeze(0)
            patch_img = patch_img.unsqueeze(0)

            optimizer.zero_grad()
            with autocast():
                seg_logits, _ = model(patch_img)
                loss = ce_loss(seg_logits, patch_lbl.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sub_loss += loss.item()
            patches += 1
            step += 1

            if step % 50 == 0:
                logging.info(f"Epoch[{epoch}] Step[{step}]: Loss={loss.item():.4f}")

        total_loss += sub_loss / patches
        count += 1

    return total_loss / count

def validate_one_epoch(model, loader):
    model.eval()
    val_loss = 0
    count = 0

    with torch.no_grad():
        for sample in loader:
            img, lbl = sample["t1"].squeeze(0).to(device), sample["label"].squeeze(0).to(device)
            sub_loss = 0
            patches = 0

            for (d0, h0, w0), patch_img in sliding_window_subvolumes(img, PATCH_SIZE, STRIDE):
                patch_lbl = lbl[
                    d0:d0+PATCH_SIZE[0],
                    h0:h0+PATCH_SIZE[1],
                    w0:w0+PATCH_SIZE[2]
                ].unsqueeze(0)
                patch_img = patch_img.unsqueeze(0)

                with autocast():
                    seg_logits, _ = model(patch_img)
                    loss = ce_loss(seg_logits, patch_lbl.long())

                sub_loss += loss.item()
                patches += 1

            val_loss += sub_loss / patches
            count += 1

    return val_loss / count

# MAIN LOOP 
best_val_loss = None
early_stop = 0

for ep in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model, train_loader, optimizer, ep)
    val_loss = validate_one_epoch(model, val_loader)

    logging.info(f"Epoch[{ep}] trainLoss={train_loss:.4f}, valLoss={val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{ep}.pth"))

    if best_val_loss is None or val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop = 0
    else:
        early_stop += 1
        if early_stop >= PATIENCE:
            logging.info(f"Early stopping at epoch {ep}")
            break

logging.info("=== Training complete! ===")

