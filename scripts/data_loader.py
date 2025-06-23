import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MRISegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with remapped .nii.gz files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Get list of all remapped files
        self.file_list = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith("_remapped.nii.gz")
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        nifti_path = self.file_list[idx]

        # Load NIfTI image
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.float32)

        # Normalization (mean=0, std=1)
        mean, std = data.mean(), data.std()
        data = (data - mean) / std

        # Convert to torch tensor and add channel dimension
        data_tensor = torch.from_numpy(data).unsqueeze(0)

        sample = {'image': data_tensor, 'filename': os.path.basename(nifti_path)}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Usage example
data_dir = '/home/osasu/data_processed_remapped'
dataset = MRISegmentationDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through data loader for testing
if __name__ == "__main__":
    for i, batch in enumerate(data_loader):
        images, filenames = batch['image'], batch['filename']
        print(f"Batch {i}: {images.shape}, Files: {filenames}")
        if i == 2:  # Just load first 3 batches for checking
            break
