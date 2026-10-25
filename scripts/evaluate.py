import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cryovision.device import get_device
from src.cryovision.dataset import CryovisionDataset
from src.cryovision.model import UNet

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calc_confusion(pred, mask):
    # pred and mask are boolean numpy arrays
    tp = np.logical_and(pred, mask).sum()
    fp = np.logical_and(pred, ~mask).sum()
    fn = np.logical_and(~pred, mask).sum()
    return tp, fp, fn

def global_metrics(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return iou, dice, precision, recall

def find_best_ndwi_threshold(val_dl):
    print("Finding optimal NDWI threshold on VAL split...")
    thresholds = np.arange(-0.2, 0.65, 0.05)
    best_thresh = 0.0
    best_iou = -1.0
    
    all_ndwi = []
    all_mask = []
    for x, y in val_dl:
        all_ndwi.append(x[:, 3].numpy())
        all_mask.append(y[:, 0].numpy() > 0.5)
        
    all_ndwi = np.concatenate(all_ndwi, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    
    for t in thresholds:
        pred = all_ndwi > t
        tp = np.logical_and(pred, all_mask).sum()
        union = np.logical_or(pred, all_mask).sum()
        iou = tp / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_thresh = t
            
    print(f"Optimal NDWI Threshold (from VAL): {best_thresh:.2f} (IoU: {best_iou:.4f})")
    return best_thresh

def evaluate(config_path):
    config = load_config(config_path)
    device = get_device()
    
    patches_dir = config['data']['patches_dir']
    manifest_path = config['data']['manifest_path']
    
    val_ds = CryovisionDataset(patches_dir, manifest_path, split="val")
    test_ds = CryovisionDataset(patches_dir, manifest_path, split="test")
    
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    best_ndwi_thresh = find_best_ndwi_threshold(val_dl)
    
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    
    print("Evaluating U-Net and NDWI on TEST split...")
    
    unet_tp = unet_fp = unet_fn = 0
    ndwi_tp = ndwi_fp = ndwi_fn = 0
    
    patch_results = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_dl, file=sys.stdout)):
            x_dev = x.to(device)
            logits = model(x_dev)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            
            mask = y.numpy()[0, 0] > 0.5
            unet_pred = probs > 0.5
            ndwi = x.numpy()[0, 3]
            ndwi_pred = ndwi > best_ndwi_thresh
            
            u_tp, u_fp, u_fn = calc_confusion(unet_pred, mask)
            unet_tp += u_tp; unet_fp += u_fp; unet_fn += u_fn
            
            n_tp, n_fp, n_fn = calc_confusion(ndwi_pred, mask)
            ndwi_tp += n_tp; ndwi_fp += n_fp; ndwi_fn += n_fn
            
            p_iou = u_tp / (u_tp + u_fp + u_fn) if (u_tp + u_fp + u_fn) > 0 else (1.0 if (u_tp == 0 and not mask.any()) else 0.0)
            
            patch_results.append({
                'iou': p_iou,
                'img': x.numpy()[0, :3],
                'mask': mask,
                'unet_pred': unet_pred,
                'ndwi_pred': ndwi_pred
            })
            
    u_iou, u_dice, u_prec, u_rec = global_metrics(unet_tp, unet_fp, unet_fn)
    n_iou, n_dice, n_prec, n_rec = global_metrics(ndwi_tp, ndwi_fp, ndwi_fn)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS (TEST SPLIT)")
    print("="*80)
    print(f"{'Metric':<15} | {'U-Net':<15} | {'NDWI Baseline':<15} | {'Delta (U-Net - NDWI)':<20}")
    print("-" * 80)
    print(f"{'IoU':<15} | {u_iou:<15.4f} | {n_iou:<15.4f} | {u_iou - n_iou:<20.4f}")
    print(f"{'Dice':<15} | {u_dice:<15.4f} | {n_dice:<15.4f} | {u_dice - n_dice:<20.4f}")
    print(f"{'Precision':<15} | {u_prec:<15.4f} | {n_prec:<15.4f} | {u_prec - n_prec:<20.4f}")
    print(f"{'Recall':<15} | {u_rec:<15.4f} | {n_rec:<15.4f} | {u_rec - n_rec:<20.4f}")
    print("="*80)
    
    if abs(u_iou - n_iou) < 0.02:
        print("\nFINDING: The delta between U-Net and NDWI is near zero.")
        print("The U-Net has not significantly outperformed the classical baseline.")
    
    interesting_patches = [p for p in patch_results if p['mask'].any() or p['unet_pred'].any()]
    if len(interesting_patches) < 12:
        interesting_patches = patch_results
        
    interesting_patches.sort(key=lambda x: x['iou'])
    worst_12 = interesting_patches[:12]
    
    print("\nGenerating 12-worst predictions grid...")
    fig, axes = plt.subplots(12, 4, figsize=(12, 36))
    if len(worst_12) == 1: axes = [axes]
    
    for i, p in enumerate(worst_12):
        img = np.transpose(p['img'], (1, 2, 0))
        for c in range(3):
            cmin, cmax = img[:,:,c].min(), img[:,:,c].max()
            img[:,:,c] = (img[:,:,c] - cmin) / (cmax - cmin + 1e-8)
            
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"RGB (IoU: {p['iou']:.3f})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(p['mask'], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(p['unet_pred'], cmap='gray')
        axes[i, 2].set_title("U-Net")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(p['ndwi_pred'], cmap='gray')
        axes[i, 3].set_title(f"NDWI > {best_ndwi_thresh:.2f}")
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    out_path = "worst_predictions.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved 12-worst grid to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    evaluate(args.config)
