import os
import json
import random

# Configuration
DATA_DIR = "/home/osasu/data_processed_remapped"
OUTPUT_SPLIT_FILE = "data_split.json"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Validate splits sum up to 1.0
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Ratios must sum to 1.0"

# List subject folders (excluding hidden files)
subjects = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]

# Shuffle subjects for reproducibility
random.seed(42)
random.shuffle(subjects)

# Calculate split sizes
num_total = len(subjects)
num_train = int(num_total * TRAIN_RATIO)
num_val = int(num_total * VAL_RATIO)

# Split subjects into train, validation, test
train_subjects = subjects[:num_train]
val_subjects = subjects[num_train:num_train + num_val]
test_subjects = subjects[num_train + num_val:]

# Sanity check to ensure no subject is lost
assert len(train_subjects) + len(val_subjects) + len(test_subjects) == num_total, "Mismatch in split sizes"

# Save splits as JSON
split_data = {
    "train": train_subjects,
    "val": val_subjects,
    "test": test_subjects
}

with open(OUTPUT_SPLIT_FILE, "w") as f:
    json.dump(split_data, f, indent=4)

# Print summary
print(f"Data split saved to '{OUTPUT_SPLIT_FILE}'")
print(f"Total subjects: {num_total}")
print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")
