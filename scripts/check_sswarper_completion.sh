#!/bin/bash

OUTPUT_DIR="/home/osasu/data_processed"

for subdir in "$OUTPUT_DIR"/sub-*; do
    subject=$(basename "$subdir")
    
    if [[ -f "$subdir/anatQQ.${subject}.nii" && -f "$subdir/anatQQ.${subject}_WARP.nii" ]]; then
        echo "$subject: ✅ Completed"
    else
        echo "$subject: ❌ Incomplete or Ongoing"
    fi
done
