import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from data_loader import MRISegmentationDataset

# Load data
data_dir = '/home/osasu/data_processed_remapped'
test_dataset = MRISegmentationDataset(data_dir=data_dir)

# Extract features and labels
features, labels = [], []
for data in test_dataset:
    image = data['image'].numpy().flatten()  # Flatten 3D MRI volume into 1D feature vector
    features.append(image)
    labels.append(0)  # Replace with actual labels

features = np.array(features)
labels = np.array(labels)

# Load model
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

# Test predictions
test_preds = model.predict(features)
accuracy = accuracy_score(labels, test_preds)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Predictions: {test_preds}")
