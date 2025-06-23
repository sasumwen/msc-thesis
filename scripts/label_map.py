import os
import nibabel as nib
import numpy as np

FASTSURFER_DIR = "/home/osasu/data_processed"
OUTPUT_DIR = "/home/osasu/data_processed_remapped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

label_map = {
    0: 0,    # Background
    17: 1, 53: 1,           # Hippocampus
    4: 2, 43: 2,            # Lateral Ventricles
    18: 3, 54: 3,           # Amygdala
    11: 4, 50: 4,           # Caudate (Basal Ganglia)
    12: 4, 51: 4,           # Putamen (Basal Ganglia)
    13: 4, 52: 4,           # Pallidum (Basal Ganglia)
    10: 5, 49: 5            # Thalamus
}

def remap_labels(input_nifti, output_nifti, label_mapping):
    img = nib.load(input_nifti)
    data = img.get_fdata().astype(np.int32)

    remapped_data = np.zeros_like(data, dtype=np.int32)
    for old_label, new_label in label_mapping.items():
        remapped_data[data == old_label] = new_label

    nib.save(nib.Nifti1Image(remapped_data, img.affine), output_nifti)
    print(f"Saved remapped file: {output_nifti}")

# Updated traversal for your directory structure
for sub_dir in os.listdir(FASTSURFER_DIR):
    sub_path = os.path.join(FASTSURFER_DIR, sub_dir)
    if os.path.isdir(sub_path):
        for ses_dir in os.listdir(sub_path):
            ses_path = os.path.join(sub_path, ses_dir)
            fastsurfer_output_path = os.path.join(ses_path, "fastsurfer_output", sub_dir, "mri")
            aparc_file = os.path.join(fastsurfer_output_path, "aparc.DKTatlas+aseg.deep.mgz")

            if os.path.isfile(aparc_file):
                subject_session_id = f"{sub_dir}_{ses_dir}"
                temp_nifti_path = os.path.join(OUTPUT_DIR, f"{subject_session_id}_temp.nii.gz")
                output_label_file = os.path.join(OUTPUT_DIR, f"{subject_session_id}_remapped.nii.gz")

                # Convert from mgz to nifti
                img = nib.load(aparc_file)
                nib.save(img, temp_nifti_path)

                # Remap labels
                remap_labels(temp_nifti_path, output_label_file, label_map)

                os.remove(temp_nifti_path)

print("All subjects processed successfully.")
