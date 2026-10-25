import torch
import platform
import contextlib

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_scaler():
    device = get_device()
    if device.type == "cuda":
        return torch.amp.GradScaler('cuda')
    
    # MPS/CPU does not use GradScaler
    class NoOpScaler:
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
    return NoOpScaler()

def autocast_context():
    device = get_device()
    if device.type == "cuda":
        return torch.amp.autocast('cuda', dtype=torch.float16)
    else:
        # No autocast for MPS currently, fallback to no-op context
        return contextlib.nullcontext()

def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def memory_report():
    device = get_device()
    if device.type == "cuda":
        return f"CUDA Max Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    elif device.type == "mps":
        try:
            return f"MPS Mem: {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
        except AttributeError:
            return "MPS Mem tracking unavailable"
    return "CPU (No VRAM)"

def configure_dataloader_kwargs(config_workers_win, config_workers_mac):
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["num_workers"] = config_workers_win
        kwargs["multiprocessing_context"] = "spawn" if config_workers_win > 0 else None
    elif platform.system() == "Darwin":
        kwargs["num_workers"] = config_workers_mac
    else:
        kwargs["num_workers"] = config_workers_win
    
    return {k: v for k, v in kwargs.items() if v is not None}
