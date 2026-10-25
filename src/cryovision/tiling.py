import os
import tarfile

def create_manifest_entry(img_path, mask_path, split, water_frac):
    return {
        "img_source": str(img_path.name),
        "mask_source": str(mask_path.name),
        "split": split,
        "water_fraction": float(water_frac),
        "crs": "None",
        "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "epoch_date": "unknown"
    }

def package_dataset(patches_dir, manifest_path, archive_path):
    print(f"Creating archive at {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add manifest
        tar.add(manifest_path, arcname="manifest.json")
        # Add patches
        for root, _, files in os.walk(patches_dir):
            for file in files:
                if file.endswith('.npz'):
                    fpath = os.path.join(root, file)
                    arcname = os.path.relpath(fpath, patches_dir)
                    tar.add(fpath, arcname=arcname)
    print("Packaging complete.")
