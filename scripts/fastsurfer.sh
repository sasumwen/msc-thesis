#!/usr/bin/env bash

DATA_PROCESSED_DIR="/home/osasu/data_processed"
FS_LICENSE_PATH="/home/osasu/license.txt"

export FS_CMD="docker run --rm -u root \
-v ${DATA_PROCESSED_DIR}:/data \
-v /home/osasu/license.txt:/fs_license/license.txt \
deepmi/fastsurfer:latest"

run_fastsurfer() {
    ses_dir="$1"
    SUBJECT_ID=$(basename $(dirname "$ses_dir"))
    SESSION_ID=$(basename "$ses_dir")
    anatSS="$ses_dir/anatSS.${SUBJECT_ID}.nii"
    OUTPUT_DIR="$ses_dir/fastsurfer_output"

    if [ -f "$anatSS" ]; then
        if [ ! -d "$OUTPUT_DIR" ]; then
            echo "Running FastSurfer for subject: $SUBJECT_ID"
            $FS_CMD \
              --t1 "/data/$SUBJECT_ID/$SESSION_ID/anatSS.${SUBJECT_ID}.nii" \
              --sid "${SUBJECT_ID}" \
              --sd "/data/$SUBJECT_ID/$SESSION_ID/fastsurfer_output" \
              --fs_license "$FS_LICENSE_PATH" \
              --parallel \
              --no_cuda
        else
            echo "FastSurfer already completed for subject: $SUBJECT_ID, skipping."
        fi
    else
        echo "anatSS file not found for $SUBJECT_ID, skipping."
    fi
}

export -f run_fastsurfer
find "$DATA_PROCESSED_DIR" -type d -name "ses-*" | parallel -j $(nproc) run_fastsurfer {}
