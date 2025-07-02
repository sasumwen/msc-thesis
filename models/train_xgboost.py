import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_loader import MRISegmentationDataset

# CONFIGURATION
SEED = 42
np.random.seed(SEED)

# DATA LOADER
data_dir = '/home/osasu/data_processed_remapped'
dataset = MRISegmentationDataset(data_dir=data_dir)

# Extract features and labels
features, labels = [], []
for data in dataset:
    image = data['image'].numpy().flatten()  # Flatten 3D MRI volume into 1D feature vector
    features.append(image)
    labels.append(0)  # Replace with actual labels

features = np.array(features)
labels = np.array(labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=SEED)

# XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=6, objective='binary:logistic', random_state=SEED)

# Train model
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

# Save model
model.save_model("xgboost_model.json")

# Validation predictions
val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")
