import os
import sys
import sqlite3
import torch
import json
import yaml
import tempfile
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cryovision.gee import init_ee, fetch_sentinel_median
from src.cryovision.inference import load_and_scale_geotiff, predict_image, extract_polygons
from src.cryovision.model import UNet
from src.cryovision.device import get_device

def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS lakes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  year INTEGER,
                  region TEXT,
                  area_km2 REAL,
                  geometry_geojson TEXT,
                  confidence REAL,
                  method TEXT)''')
    conn.commit()
    return conn

def predict_ndwi(img, threshold=0.15):
    nir = img[0].astype(np.float32)
    green = img[2].astype(np.float32)
    denom = green + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(denom == 0, 0, (green - nir) / denom)
    return ndwi > threshold

def generate_tiles(bbox, step=0.1):
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            next_lon = min(lon + step, max_lon)
            next_lat = min(lat + step, max_lat)
            tiles.append([lon, lat, next_lon, next_lat])
            lat += step
        lon += step
    return tiles

def precompute():
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    init_ee()
    device = get_device()
    
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    
    os.makedirs("data", exist_ok=True)
    db_path = "data/glacier_lakes.db"
    conn = setup_db(db_path)
    c = conn.cursor()
    
    # Himalayan glacial lake regions
    regions = {
        "Khumbu": [86.7, 27.8, 87.1, 28.1],
        "Sikkim": [88.0, 27.6, 88.3, 28.0],
        "Langtang": [85.3, 28.1, 85.7, 28.4],
        "Manaslu": [84.4, 28.3, 84.8, 28.6],
        "Bhutan": [89.2, 27.8, 89.6, 28.1],
    }
    
    years = list(range(2019, 2026))
    
    for region_name, bbox in regions.items():
        for year in years:
            print(f"Processing {region_name} for year {year}...")
            tiles = generate_tiles(bbox, step=0.1)
            total_unet = 0
            total_ndwi = 0
            
            for tile_idx, tile_bbox in enumerate(tiles):
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tif_path = tmp.name
                    
                try:
                    fetch_sentinel_median(tile_bbox, year, tif_path)
                    img, transform_mat, crs = load_and_scale_geotiff(tif_path)
                    
                    # U-Net
                    mask_unet = predict_image(model, img, device)
                    polys_unet = extract_polygons(mask_unet, transform_mat, crs)
                    
                    # NDWI
                    mask_ndwi = predict_ndwi(img, threshold=0.15)
                    polys_ndwi = extract_polygons(mask_ndwi, transform_mat, crs)
                    
                    for p in polys_unet:
                        c.execute("INSERT INTO lakes (year, region, area_km2, geometry_geojson, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
                                  (year, region_name, p["area_km2"], json.dumps(p["geometry"]), 0.9, "unet"))
                                  
                    for p in polys_ndwi:
                        c.execute("INSERT INTO lakes (year, region, area_km2, geometry_geojson, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
                                  (year, region_name, p["area_km2"], json.dumps(p["geometry"]), 0.9, "ndwi"))
                                  
                    conn.commit()
                    total_unet += len(polys_unet)
                    total_ndwi += len(polys_ndwi)
                    
                except Exception as e:
                    err_msg = str(e)
                    if "no bands" in err_msg or "Image.select" in err_msg:
                        # Cloud filter removed all images for this small tile
                        pass
                    else:
                        print(f"    Tile {tile_idx} warning: {e}")
                finally:
                    if os.path.exists(tif_path):
                        os.remove(tif_path)
                        
            print(f"  -> Found {total_unet} U-Net lakes and {total_ndwi} NDWI lakes.")
                    
    conn.close()
    print("Batch precompute complete!")

if __name__ == "__main__":
    precompute()
