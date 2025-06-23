#!/usr/bin/env bash

# Base data directory
BASE_DIR="/home/osasu/data"

# Raw data directories (control and schiz_positive)
DATA_DIRS=("control" "schiz_positive")

# Output processed data directory
PROCESSED_DIR="$BASE_DIR/data_processed"
mkdir -p "$PROCESSED_DIR"

# Template for SSwarper2
BASE_TEMPLATE="/home/osasu/abin/MNI152_2009_template_SSW.nii.gz"

# Loop through each group (control and schiz_positive)
for GROUP in "${DATA_DIRS[@]}"; do
    echo "Processing group: $GROUP"

    # Find each subject directory within the group
    SUB_DIRS=$(find "$BASE_DIR/$GROUP" -mindepth 1 -maxdepth 1 -type d)

    # Loop through each subject directory
    for SUB_DIR in $SUB_DIRS; do
        SUBJECT_ID=$(basename "$SUB_DIR")

        # Search for the specific T1 file (_run-01_T1w.nii.gz)
        T1_FILE=$(find "$SUB_DIR" -type f -name "*_run-01_T1w.nii.gz" | head -n 1)

        # Check if the required T1 file exists
        if [[ -z "$T1_FILE" ]]; then
            echo "T1 file '_run-01_T1w.nii.gz' not found for subject: $SUBJECT_ID. Skipping..."
            continue
        fi

        echo "Processing subject: $SUBJECT_ID"
        echo "Found T1 file: $T1_FILE"

        # Define output directory for SSwarper2 for this subject
        OUT_SUB_DIR="$PROCESSED_DIR/$GROUP/$SUBJECT_ID"
        mkdir -p "$OUT_SUB_DIR"

        # Run SSwarper2
        sswarper2 \
          -input "$T1_FILE" \
          -base "$BASE_TEMPLATE" \
          -subid "$SUBJECT_ID" \
          -odir "$OUT_SUB_DIR" \
          -cost_aff lpa+ZZ \
          -cost_nl_init lpa \
          -cost_nl_final pcl \
          -minp 11 \
          -verb \
          -nolite

        echo "Completed processing for subject: $SUBJECT_ID"
    done
done

echo "All processing done. Check processed data under $PROCESSED_DIR."
