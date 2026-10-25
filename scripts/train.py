import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cryovision.device import get_device, make_scaler, autocast_context, seed_everything, memory_report, configure_dataloader_kwargs
from src.cryovision.dataset import CryovisionDataset
from src.cryovision.model import UNet
from src.cryovision.losses import BCEDiceLoss

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_dice(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()

def train(config_path, smoke=False):
    config = load_config(config_path)
    
    seed_everything(42)
    device = get_device()
    scaler = make_scaler()
    
    print(f"Starting Training on Device: {device.type.upper()}")
    
    patches_dir = config['data']['patches_dir']
    manifest_path = config['data']['manifest_path']
    
    train_ds = CryovisionDataset(patches_dir, manifest_path, split="train")
    val_ds = CryovisionDataset(patches_dir, manifest_path, split="val")
    
    if smoke:
        train_ds.patch_ids = train_ds.patch_ids[:min(50, len(train_ds.patch_ids))]
        val_ds.patch_ids = val_ds.patch_ids[:min(10, len(val_ds.patch_ids))]
        config['training']['epochs'] = 2
        print(f"SMOKE TEST MODE: Train size={len(train_ds)}, Val size={len(val_ds)}")
    
    if device.type == "cuda":
        batch_size = config['training']['batch_size_cuda']
        grad_accum = config['training']['grad_accum_steps_cuda']
    elif device.type == "mps":
        batch_size = config['training']['batch_size_mps']
        grad_accum = config['training']['grad_accum_steps_mps']
    else:
        batch_size = 2
        grad_accum = 1
        
    dl_kwargs = configure_dataloader_kwargs(config['training']['num_workers_windows'], config['training']['num_workers_mac'])
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        
    criterion = BCEDiceLoss(bce_weight=config['training']['bce_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter()
    
    best_val_dice = 0.0
    patience = config['training']['patience']
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            if device.type == "cuda":
                inputs = inputs.to(memory_format=torch.channels_last)
                
            try:
                with autocast_context():
                    logits = model(inputs)
                    loss = criterion(logits, targets) / grad_accum
                    
                scaler.scale(loss).backward()
                
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_dl):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                train_loss += loss.item() * grad_accum
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CRITICAL: CUDA Out of Memory!")
                    if inputs.shape[-1] > 256:
                        print("Falling back to 256x256 random crops...")
                        torch.cuda.empty_cache()
                        _, _, h, w = inputs.shape
                        th, tw = 256, 256
                        if h >= th and w >= tw:
                            top = random.randint(0, h - th)
                            left = random.randint(0, w - tw)
                            inputs = inputs[..., top:top+th, left:left+tw]
                            targets = targets[..., top:top+th, left:left+tw]
                            with autocast_context():
                                logits = model(inputs)
                                loss = criterion(logits, targets) / grad_accum
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            train_loss += loss.item() * grad_accum
                            continue
                raise e
            
        train_loss /= len(train_dl)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                if device.type == "cuda":
                    inputs = inputs.to(memory_format=torch.channels_last)
                
                with autocast_context():
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    
                val_loss += loss.item()
                val_dice += compute_dice(logits, targets)
                
        val_loss /= len(val_dl)
        val_dice /= len(val_dl)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f} - {memory_report()}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metric/Val_Dice', val_dice, epoch)
        
        scheduler.step()
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_val_dice
        }
        
        torch.save(ckpt, os.path.join(config['training']['checkpoint_dir'], 'last.pt'))
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(ckpt, os.path.join(config['training']['checkpoint_dir'], 'best.pt'))
            patience_counter = 0
            print("  -> Saved new best checkpoint!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break
            
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    if platform.system() == "Windows":
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
            
    train(args.config, args.smoke)
