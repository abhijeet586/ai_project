import rasterio
import rasterio.features
import numpy as np
import torch
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

def load_and_scale_geotiff(path):
    with rasterio.open(path) as src:
        img = src.read() # (3, H, W) -> B8, B4, B3
        transform_mat = src.transform
        crs = src.crs
        
    # GEE SR is 0-10000. Stretch 0-3000 to 0-255 uint8 to match Kaggle
    img = np.clip(img, 0, 3000)
    img = (img / 3000.0 * 255).astype(np.uint8)
    
    return img, transform_mat, crs

def predict_image(model, img, device, tile_size=400):
    _, h, w = img.shape
    mask = np.zeros((h, w), dtype=bool)
    
    nir = img[0].astype(np.float32)
    green = img[2].astype(np.float32)
    denom = green + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(denom == 0, 0, (green - nir) / denom)
        
    x_full = np.stack([img[0], img[1], img[2], ndwi], axis=0).astype(np.float32)
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            i_end = min(i + tile_size, h)
            j_end = min(j + tile_size, w)
            
            tile = x_full[:, i:i_end, j:j_end]
            
            pad_h = tile_size - (i_end - i)
            pad_w = tile_size - (j_end - j)
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
                
            x_t = torch.from_numpy(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    logits = model(x_t)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    
            pred = probs > 0.5
            mask[i:i_end, j:j_end] = pred[:(tile_size-pad_h), :(tile_size-pad_w)]
            
    return mask

def calculate_area(geom, src_crs):
    project = pyproj.Transformer.from_crs(src_crs, pyproj.CRS('ESRI:54009'), always_xy=True).transform
    projected_geom = transform(project, geom)
    return projected_geom.area / 1e6

def extract_polygons(mask, transform_mat, src_crs):
    polygons = []
    mask_u8 = mask.astype(np.uint8)
    
    for geom_dict, val in rasterio.features.shapes(mask_u8, transform=transform_mat):
        if val == 1:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = geom.buffer(0)
                
            area_km2 = calculate_area(geom, src_crs)
            
            if area_km2 > 0.01:
                polygons.append({
                    "geometry": geom_dict,
                    "area_km2": area_km2
                })
                
    return polygons
