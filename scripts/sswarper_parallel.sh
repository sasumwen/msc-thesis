#!/usr/bin/env bash

BASE_TEMPLATE="/home/osasu/abin/MNI152_2009_template_SSW.nii.gz"
OUTPUT_DIR="/home/osasu/data_processed"

process_subject() {
    T1_FILE="$1"
    SUBJECT_ID=$(basename $(dirname $(dirname "$T1_FILE")))
    GROUP=$(basename $(dirname $(dirname $(dirname "$T1_FILE"))))
    OUT_SUB_DIR="$OUTPUT_DIR/$GROUP/$SUBJECT_ID"

    mkdir -p "$OUT_SUB_DIR"

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
}

export -f process_subject
export BASE_TEMPLATE OUTPUT_DIR

parallel -j $(nproc) process_subject :::: ~/subject_list.txt
