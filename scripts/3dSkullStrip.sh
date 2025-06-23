#!/usr/bin/env bash

DATA_PROCESSED_DIR="/home/osasu/data_processed"

export PATH=$PATH:/home/osasu/abin

process_subject() {
    ses_dir="$1"
    T1_IMAGE=$(find "$ses_dir" -maxdepth 1 -type f -name "anatUAC*.nii" | head -n 1)

    if [ -f "$T1_IMAGE" ]; then
        SUBJECT_ID=$(basename $(dirname "$ses_dir"))
        OUTPUT_FILE="$ses_dir/anatSS.${SUBJECT_ID}.nii"

        if [ ! -f "$OUTPUT_FILE" ]; then
            echo "Generating anatSS for $SUBJECT_ID"
            3dSkullStrip -input "$T1_IMAGE" -prefix "$OUTPUT_FILE"
        else
            echo "anatSS already exists for $SUBJECT_ID, skipping."
        fi
    else
        echo "No anatUAC file found in $ses_dir, skipping."
    fi
}

export -f process_subject

find "$DATA_PROCESSED_DIR" -type d -name "ses-*" | parallel -j $(nproc) process_subject
