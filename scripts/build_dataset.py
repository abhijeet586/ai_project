import argparse
import os
import sys
import json
import random
import yaml
import numpy as np
import rasterio
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cryovision.tiling import create_manifest_entry, package_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def discover_pairs(data_dir):
    data_dir = Path(data_dir)
    images = [p for p in data_dir.rglob('*') if p.is_file() and ('image' in str(p).lower() or 'img' in str(p).lower())]
    masks = {p.name.lower().replace('mask', 'image').replace('label', 'image'): p for p in data_dir.rglob('*') if p.is_file() and ('mask' in str(p).lower() or 'label' in str(p).lower())}
    
    pairs = []
    for img in images:
        key = img.name.lower()
        if key in masks:
            pairs.append((img, masks[key]))
        else:
            key2 = img.name.lower().replace('image', 'mask')
            if key2 in masks:
                pairs.append((img, masks[key2]))
    return list(set(pairs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)['data']
    raw_dir = config['raw_dir']
    out_dir = Path(config['patches_dir'])
    manifest_path = config['manifest_path']
    archive_path = config['archive_path']
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = discover_pairs(raw_dir)
    if not pairs:
        print(f"No pairs found in {raw_dir}")
        return
        
    # Standardize splits safely
    random.seed(42)
    pairs.sort(key=lambda x: str(x[0])) # ensure deterministic sort before shuffle
    random.shuffle(pairs)
    
    n = len(pairs)
    n_test = int(n * config['test_split'])
    n_val = int(n * config['val_split'])
    
    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test:n_test+n_val]
    train_pairs = pairs[n_test+n_val:]
    
    def get_split(pair):
        if pair in test_pairs: return "test"
        if pair in val_pairs: return "val"
        return "train"

    manifest = {}
    kept = 0
    filtered = 0
    
    print("Processing patches...")
    for idx, (img_path, mask_path) in enumerate(pairs):
        with rasterio.open(img_path) as src:
            img = src.read() # (3, H, W)
        with rasterio.open(mask_path) as src:
            mask = src.read(1) > 0 # (H, W) boolean
            
        water_frac = np.mean(mask)
        split = get_split((img_path, mask_path))
        
        # Filtering logic for train set only
        if split == "train" and water_frac == 0.0:
            if random.random() > config['keep_ratio']:
                filtered += 1
                continue
                
        # Calculate NDWI (ch0=NIR, ch2=Green)
        nir = img[0].astype(np.float32)
        green = img[2].astype(np.float32)
        denom = green + nir
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = np.where(denom == 0, 0, (green - nir) / denom)
            
        # Create 4-channel input (NIR, Red, Green, NDWI)
        x = np.stack([img[0], img[1], img[2], ndwi], axis=0).astype(np.float32)
        y = mask.astype(np.float32)
        
        patch_name = f"patch_{idx:04d}"
        np.savez_compressed(out_dir / f"{patch_name}.npz", x=x, y=y)
        
        manifest[patch_name] = create_manifest_entry(img_path, mask_path, split, water_frac)
        kept += 1
        
    print(f"Kept {kept} patches, filtered {filtered} empty patches.")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    package_dataset(out_dir, manifest_path, archive_path)

if __name__ == "__main__":
    main()
