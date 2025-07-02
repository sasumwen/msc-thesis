import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models.video import r3d_18
from data_loader import MRISegmentationDataset

# CONFIGURATION
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 5e-5
PATIENCE = 10
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "resnet3d_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA LOADER
train_dataset = MRISegmentationDataset(data_dir='/home/osasu/data_processed_remapped')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL SETUP
model = r3d_18(pretrained=False, progress=True, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# TRAINING FUNCTION
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader):
        images = batch['image'].to(device)
        labels = torch.zeros(images.size(0), dtype=torch.long).to(device)  # Adjust labels appropriately

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# MAIN TRAINING LOOP
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {train_loss:.4f}")

    scheduler.step(train_loss)

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth"))

    # Early stopping logic
    if train_loss < best_loss:
        best_loss = train_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
