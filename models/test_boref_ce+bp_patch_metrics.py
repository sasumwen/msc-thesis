"""
test_train_boref_ce_bp_patch.py

Test script for patch-based training of BoRefAttnNet using cross-entropy and boundary penalty.
"""

import torch
from freedsurfer_data_loader import FreedSurferData
from boref_net import BoRefAttnNet
from torch.utils.data import DataLoader
import json

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and Configurations
DATA_ROOT = "/home/osasu/data_processed_remapped"
DATA_SPLIT_FILE = "data_split.json"
CHECKPOINT_PATH = "boref_ce_bp_patch_checkpoints/boref_epoch_147.pth"
PATCH_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)

# Load data splits
with open(DATA_SPLIT_FILE) as f:
    splits = json.load(f)

# Validation dataset and loader for quick test
val_ds = FreedSurferData(
    data_root=DATA_ROOT,
    subject_list=splits["val"],
    crop_size=None,
    augment=False
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Initialize model
model = BoRefAttnNet(
    n_channels=1,
    n_classes=6,
    bilinear=True,
    gn_groups=8,
    use_checkpoint=True
).to(device)

# Load trained weights
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Helper function
def sliding_window_subvolumes(vol, window_size, stride):
    C, D, H, W = vol.shape
    wD, wH, wW = window_size
    sD, sH, sW = stride
    for d0 in range(0, D - wD + 1, sD):
        for h0 in range(0, H - wH + 1, sH):
            for w0 in range(0, W - wW + 1, sW):
                yield (d0, h0, w0), vol[:, d0:d0+wD, h0:h0+wH, w0:w0+wW]

# Run test
def run_test():
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            img = sample["t1"].squeeze(0).to(device)

            patches_processed = 0
            for (d0, h0, w0), patch_img in sliding_window_subvolumes(img, PATCH_SIZE, STRIDE):
                patch_img = patch_img.unsqueeze(0)
                seg_logits, boundary_logits = model(patch_img)

                print(f"Sample[{idx}] Patch[{patches_processed}] seg_logits shape: {seg_logits.shape}, boundary_logits shape: {boundary_logits.shape}")

                patches_processed += 1

                # For quick test, limit to 2 patches per subject
                if patches_processed >= 2:
                    break

            # Limit to 2 subjects for quick testing
            if idx >= 1:
                break

    print("Boundary penalty test completed successfully.")

if __name__ == "__main__":
    run_test()
