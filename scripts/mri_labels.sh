#!/usr/bin/env bash

DATA_PROCESSED_REMAPPED="/home/osasu/data_processed_remapped"
DATA_PROCESSED="/home/osasu/data_processed"

# Loop over remapped label files
for label_file in "$DATA_PROCESSED_REMAPPED"/*_remapped.nii.gz; do
    # Extract subject ID from filename
    filename=$(basename "$label_file")
    subject=$(echo "$filename" | cut -d'_' -f1)
    session=$(echo "$filename" | cut -d'_' -f2)

    # Create new directory for subject
    subject_dir="$DATA_PROCESSED_REMAPPED/$subject"
    mkdir -p "$subject_dir"

    # Move and rename label file
    mv "$label_file" "$subject_dir/${subject}_label.nii.gz"

    # Copy the corresponding T1 MRI
    anatss_file="$DATA_PROCESSED/$subject/$session/anatSS.${subject}.nii"

    if [ -f "$anatss_file" ]; then
        cp "$anatss_file" "$subject_dir/${subject}_t1.nii"
    else
        echo "T1 MRI not found for subject: $subject, session: $session"
    fi
done


# Completion message
echo "All files reorganized successfully."
