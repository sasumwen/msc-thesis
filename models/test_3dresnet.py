import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from data_loader import MRISegmentationDataset

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA LOADER
test_dataset = MRISegmentationDataset(data_dir='/home/osasu/data_processed_remapped')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# MODEL SETUP
model = r3d_18(pretrained=False, progress=True, num_classes=2).to(device)
model.load_state_dict(torch.load('resnet3d_checkpoints/epoch_best.pth'))
model.eval()

# TESTING LOOP
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        print(f"Predictions: {predictions.cpu().numpy()}")
