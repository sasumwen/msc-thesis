#!/usr/bin/env python3

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import glob

#!/usr/bin/env python3

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

class FreedSurferData(Dataset):
    def __init__(self, data_root, subject_list, crop_size=(128,128,128), augment=False):
        self.data_root = data_root
        self.subject_list = subject_list
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        subject_id = self.subject_list[idx]

        subject_path = os.path.join(self.data_root, subject_id)
        
        t1_path = os.path.join(subject_path, f"{subject_id}_t1.nii")
        label_path = os.path.join(subject_path, f"{subject_id}_label.nii.gz")

        if not os.path.exists(t1_path):
            raise FileNotFoundError(f"T1 file not found: {t1_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load MRI and labels
        t1_img = nib.load(t1_path).get_fdata().astype(np.float32)
        label_img = nib.load(label_path).get_fdata().astype(np.int64)

        # Normalize T1
        t1_img = (t1_img - np.mean(t1_img)) / (np.std(t1_img) + 1e-5)

        # Optional cropping for training
        t1_img, label_img = self.random_crop_3d(t1_img, label_img)

        # Convert to tensors
        t1_tensor = torch.from_numpy(t1_img).unsqueeze(0)  # [1,D,H,W]
        label_tensor = torch.from_numpy(label_img)         # [D,H,W]

        return {"t1": t1_tensor, "label": label_tensor, "sub_id": subject_id}

    def random_crop_3d(self, image, label):
        if self.crop_size is None:
            return image, label

        D, H, W = image.shape
        cd, ch, cw = self.crop_size
        d0 = np.random.randint(0, D - cd + 1)
        h0 = np.random.randint(0, H - ch + 1)
        w0 = np.random.randint(0, W - cw + 1)

        img_crop = image[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        lbl_crop = label[d0:d0+cd, h0:h0+ch, w0:w0+cw]

        return img_crop, lbl_crop

# Load splits from JSON
if __name__ == "__main__":
    with open("data_split.json") as f:
        splits = json.load(f)

    train_ds = FreedSurferData(
        data_root="/home/osasu/data_processed_remapped",
        subject_list=splits["train"],
        crop_size=(128,128,128),
        augment=True
    )

    val_ds = FreedSurferData(
        data_root="/home/osasu/data_processed_remapped",
        subject_list=splits["val"],
        crop_size=(128,128,128),
        augment=False
    )

    test_ds = FreedSurferData(
        data_root="/home/osasu/data_processed_remapped",
        subject_list=splits["test"],
        crop_size=(128,128,128),
        augment=False
    )

    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    for sample in dataloader:
        print("Loaded sample:", sample["sub_id"])
        print("MRI shape:", sample["t1"].shape)
        print("Label shape:", sample["label"].shape)
        break
