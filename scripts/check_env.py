import os
import sys
import platform
import subprocess

def run_check(name, check_fn):
    print(f"Checking {name}...", end=" ")
    try:
        result = check_fn()
        print(f"OK ({result})")
    except Exception as e:
        print(f"FAILED\n  -> {e}")
        sys.exit(1)

def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        raise RuntimeError(f"Python 3.11+ required, got {v.major}.{v.minor}")
    return f"Python {sys.version.split()[0]}"

def check_torch_device():
    import torch
    import torch.nn as nn
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    if platform.system() == "Windows" and device != "cuda":
        raise RuntimeError(
            "CRITICAL: Running on Windows but CUDA is NOT available. "
            "You likely have a CPU-only PyTorch wheel installed."
        )
        
    # Real conv pass
    try:
        x = torch.randn(1, 3, 256, 256, device=device)
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
        out = conv(x)
        assert out.shape == (1, 16, 256, 256)
    except Exception as e:
        raise RuntimeError(f"Failed to perform Conv2D on device {device}: {e}")
        
    return f"PyTorch {torch.__version__}, Device: {device}"

def check_cuda_memory():
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"CUDA device 0 VRAM: {vram:.2f} GB"
    elif device == "mps":
        return "MPS enabled (unified memory)"
    else:
        return "CPU mode"

def check_rasterio():
    import rasterio
    import numpy as np
    from rasterio.transform import from_origin
    
    # Create in-memory dummy GeoTIFF
    try:
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff', width=10, height=10, count=1,
                dtype=rasterio.uint8, crs='EPSG:4326',
                transform=from_origin(86.0, 27.0, 0.1, 0.1)
            ) as dataset:
                dataset.write(np.zeros((1, 10, 10), dtype=np.uint8))
                crs = dataset.crs
                return f"rasterio {rasterio.__version__}, CRS capability OK ({crs})"
    except Exception as e:
        raise RuntimeError(f"Rasterio spatial operations failed: {e}")

def check_geopandas():
    import geopandas as gpd
    from shapely.geometry import Point
    
    try:
        gdf = gpd.GeoDataFrame(
            {'id': [1]}, 
            geometry=[Point(86.0, 27.0)], 
            crs="EPSG:4326"
        )
        # Bounding box calculation
        bounds = gdf.total_bounds
        return f"geopandas {gpd.__version__}, shapely backend OK"
    except Exception as e:
        raise RuntimeError(f"GeoPandas failed to instantiate or calculate bounds: {e}")

def check_ee():
    import ee
    return f"earthengine-api {ee.__version__}"

def check_gdal_origin():
    try:
        # Check if gdal is in conda list
        result = subprocess.run(["conda", "list", "^gdal$"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            if "gdal" in result.stdout.lower() and "conda-forge" in result.stdout.lower():
                return "GDAL verified from conda-forge"
            elif "gdal" in result.stdout.lower() and "pypi" in result.stdout.lower():
                raise RuntimeError("CRITICAL: GDAL appears to be installed via pip! This breaks C++ ABIs. Use conda-forge.")
            
        # Fallback check for module path
        from osgeo import gdal
        path = gdal.__file__
        if "site-packages" in path and "miniforge" not in path.lower() and "conda" not in path.lower():
             print(f"Warning: GDAL path ({path}) doesn't explicitly mention conda. Verify manually.", end=" ")
        return f"GDAL {gdal.__version__} OK"
    except FileNotFoundError:
        # If conda command is not available, just check version
        from osgeo import gdal
        return f"GDAL {gdal.__version__} (unable to verify conda origin)"
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        from osgeo import gdal
        return f"GDAL {gdal.__version__} (unable to verify conda origin: {e})"

if __name__ == "__main__":
    print("=== Cryovision Environment Check ===")
    run_check("Python", check_python)
    run_check("PyTorch & Device", check_torch_device)
    run_check("GPU Memory", check_cuda_memory)
    run_check("Rasterio", check_rasterio)
    run_check("GeoPandas", check_geopandas)
    run_check("Earth Engine API", check_ee)
    run_check("GDAL Origin", check_gdal_origin)
    print("====================================")
    print("SUCCESS: Environment is correctly configured.")
