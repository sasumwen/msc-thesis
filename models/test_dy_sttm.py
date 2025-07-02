import torch
from torch.utils.data import DataLoader
from dy_sttm import DynamicSTTM
from data_loader import MRISegmentationDataset

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA LOADER
test_dataset = MRISegmentationDataset(data_dir='/home/osasu/data_processed_remapped')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# MODEL SETUP
model = DynamicSTTM(input_dim=128, num_classes=2, heads=8, dropout=0.3).to(device)
model.load_state_dict(torch.load('dysttm_checkpoints/epoch_best.pth'))
model.eval()

# TESTING LOOP
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)

        # Reshape input as needed (batch_size, seq_len, spatial_dim)
        B, C, D, H, W = images.shape
        inputs = images.view(B, D, H * W)

        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)

        print(f"Predictions: {predictions.cpu().numpy()}")
