import ee
import os
import requests
import tempfile

def init_ee():
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not initialized. Attempting authentication...")
        ee.Authenticate()
        ee.Initialize()

def fetch_sentinel_median(bbox, year, out_path):
    """
    bbox: [min_lon, min_lat, max_lon, max_lat]
    Downloads median B8, B4, B3 for Sep 1-30.
    """
    geom = ee.Geometry.BBox(*bbox)
    start_date = f"{year}-09-01"
    end_date = f"{year}-09-30"
    
    # Sentinel-2 SR
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(geom)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
                  
    image = collection.median().select(['B8', 'B4', 'B3'])
    image = image.clip(geom)
    
    # Download as GeoTIFF
    url = image.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': geom,
        'format': 'GEO_TIFF'
    })
    
    response = requests.get(url)
    with open(out_path, 'wb') as f:
        f.write(response.content)
        
    return out_path
