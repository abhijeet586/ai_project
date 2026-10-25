import argparse
import os
import sys
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import random

def discover_dataset(data_dir):
    data_dir = Path(data_dir)
    print(f"Scanning {data_dir}...")
    
    # 1. Gather all image-like files
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    all_files = [p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in extensions]
    
    # 2. Heuristically separate images and masks
    images = []
    masks = []
    for f in all_files:
        lpath = str(f).lower()
        if 'mask' in lpath or 'label' in lpath:
            masks.append(f)
        elif 'image' in lpath or 'img' in lpath:
            images.append(f)
        else:
            pass # Uncategorized
            
    # Try pairing by filename
    img_dict = {f.name: f for f in images}
    mask_dict = {f.name: f for f in masks}
    
    pairs = []
    for name, img_path in img_dict.items():
        # strict match
        if name in mask_dict:
            pairs.append((img_path, mask_dict[name]))
        # fuzzy match (e.g. image_1.tif vs mask_1.tif)
        else:
            fuzzy1 = name.lower().replace('image', 'mask').replace('img', 'mask')
            for mname, mpath in mask_dict.items():
                if fuzzy1 == mname.lower() or name.lower() == mname.lower().replace('mask', 'image'):
                    pairs.append((img_path, mpath))
                    break
    
    pairs = list(set(pairs))
    
    print(f"Found {len(all_files)} total image/raster files.")
    print(f"Categorised {len(images)} images and {len(masks)} masks.")
    print(f"Successfully paired {len(pairs)} image/mask pairs.")
    
    if len(pairs) == 0:
        print("ERROR: Could not pair images and masks. Check the folder layout.")
        return []
        
    # Inspect the first pair to get stats
    with rasterio.open(pairs[0][0]) as src:
        dtype = src.dtypes[0]
        count = src.count
        crs = src.crs if src.crs else "None"
        transform = src.transform
        img_data = src.read()
        
    with rasterio.open(pairs[0][1]) as msrc:
        mask_data = msrc.read(1)
        
    water_frac = np.mean(mask_data > 0)
    
    print("\n--- Dataset Metadata (from sample) ---")
    print(f"Image Dtype: {dtype}, Channels: {count}")
    print(f"CRS: {crs}")
    print(f"Transform: {transform}")
    for c in range(count):
        print(f"Value range ch{c}: [{img_data[c].min()}, {img_data[c].max()}]")
    print(f"Mask unique values: {np.unique(mask_data)}")
    print(f"Sample water pixel fraction: {water_frac:.3f}")
    
    return pairs

def compute_ndwi(img):
    # ch0=NIR (B8), ch1=Red (B4), ch2=Green (B3) as per Kaggle dataset spec
    nir = img[0].astype(np.float32)
    green = img[2].astype(np.float32)
    
    denom = green + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(denom == 0, 0, (green - nir) / denom)
    return ndwi

def sweep_thresholds(pairs):
    thresholds = np.arange(-0.2, 0.65, 0.05)
    sample_size = min(50, len(pairs))
    subset = random.sample(pairs, sample_size)
    
    print(f"\nSweeping NDWI thresholds over {sample_size} images...")
    
    best_mean_iou = -1
    best_thresh = 0
    best_ious = []
    
    for t in thresholds:
        ious = []
        for img_path, mask_path in subset:
            with rasterio.open(img_path) as src:
                img = src.read()
            with rasterio.open(mask_path) as src:
                mask = src.read(1) > 0 # binarize mask
                
            ndwi = compute_ndwi(img)
            pred = ndwi > t
            
            intersection = np.logical_and(pred, mask).sum()
            union = np.logical_or(pred, mask).sum()
            iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
            ious.append(iou)
            
        mean_iou = np.mean(ious)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_thresh = t
            best_ious = ious
            
    print(f"Best NDWI threshold: {best_thresh:.2f}")
    print(f"Best Mean IoU: {best_mean_iou:.3f}")
    print(f"IoU Spread: Min {np.min(best_ious):.3f}, Max {np.max(best_ious):.3f}, Std {np.std(best_ious):.3f}")
    
    return best_thresh, best_mean_iou

def print_verdict(mean_iou):
    print("\n" + "="*70)
    print("VERDICT:")
    if mean_iou > 0.95:
        print(f"IoU = {mean_iou:.3f} > 0.95")
        print("Masks are highly likely NDWI-derived. Model cannot beat its teacher.")
        print("--> BRANCH B: Kaggle is unusable as ground truth. Will use GEE + inventory.")
    elif mean_iou >= 0.75:
        print(f"IoU = {mean_iou:.3f} (between 0.75 and 0.95)")
        print("Partially independent. Usable. The gap is the headroom.")
    else:
        print(f"IoU = {mean_iou:.3f} < 0.75")
        print("Independent labels. Best case.")
        print("--> BRANCH A: Kaggle is the training set. Skip Phase 1B.")
    print("="*70 + "\n")

def generate_grid(pairs, best_thresh, output_path="audit_grid.png"):
    print("Generating 12-sample PNG grid...")
    samples = random.sample(pairs, min(12, len(pairs)))
    
    fig, axes = plt.subplots(12, 3, figsize=(10, 36))
    if len(samples) == 1:
        axes = [axes]
        
    for i, (img_path, mask_path) in enumerate(samples):
        with rasterio.open(img_path) as src:
            img = src.read()
        with rasterio.open(mask_path) as src:
            mask = src.read(1) > 0
            
        ndwi = compute_ndwi(img)
        pred = ndwi > best_thresh
        
        # Normalize for display
        img_disp = np.transpose(img, (1, 2, 0)).astype(np.float32)
        for c in range(img_disp.shape[2]):
            cmin, cmax = img_disp[:,:,c].min(), img_disp[:,:,c].max()
            img_disp[:,:,c] = (img_disp[:,:,c] - cmin) / (cmax - cmin + 1e-8)
            
        axes[i, 0].imshow(img_disp)
        axes[i, 0].set_title(f"Sample {i+1} (NIR/R/G)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Provided Mask")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f"NDWI > {best_thresh:.2f}")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Kaggle Masks against NDWI")
    parser.add_argument("--data-dir", type=str, default="./data/kaggle_dataset", help="Path to Kaggle dataset")
    args = parser.parse_args()
    
    random.seed(42)
    pairs = discover_dataset(args.data_dir)
    if not pairs:
        sys.exit(1)
        
    best_thresh, best_iou = sweep_thresholds(pairs)
    print_verdict(best_iou)
    generate_grid(pairs, best_thresh)
