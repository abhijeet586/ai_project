from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import torch
import os
import sys
import yaml
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cryovision.gee import init_ee, fetch_sentinel_median
from src.cryovision.inference import load_and_scale_geotiff, predict_image, extract_polygons
from src.cryovision.model import UNet
from src.cryovision.device import get_device

app = FastAPI()

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/history")
def get_history():
    conn = sqlite3.connect("data/glacier_lakes.db")
    c = conn.cursor()
    c.execute("""
        SELECT year, region, method, SUM(area_km2) 
        FROM lakes 
        GROUP BY year, region, method
    """)
    rows = c.fetchall()
    conn.close()
    
    data = []
    for r in rows:
        data.append({"year": r[0], "region": r[1], "method": r[2], "area_km2": r[3]})
    return JSONResponse(content=data)

@app.get("/risk")
def get_risk(region: str = None, year: int = None):
    conn = sqlite3.connect("data/glacier_lakes.db")
    c = conn.cursor()
    
    query = """SELECT lake_id, region, latest_year, latest_area_km2, earliest_area_km2,
               growth_rate_pct, risk_score, risk_level, centroid_lon, centroid_lat,
               geometry_geojson FROM risk_scores"""
    params = []
    conditions = []
    if region and region != "All":
        conditions.append("region = ?")
        params.append(region)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY risk_score DESC"
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    features = []
    for r in rows:
        try:
            geom = __import__('json').loads(r[10])
        except:
            continue
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "lake_id": r[0],
                "region": r[1],
                "latest_year": r[2],
                "area_km2": r[3],
                "earliest_area_km2": r[4],
                "growth_pct": r[5],
                "risk_score": r[6],
                "risk_level": r[7],
                "centroid_lon": r[8],
                "centroid_lat": r[9]
            }
        })
    return JSONResponse(content={"type": "FeatureCollection", "features": features})

@app.get("/lakes")
def get_lakes(region: str = None, year: int = None, method: str = "unet"):
    """Get individual lake polygons filtered by region, year, and method."""
    conn = sqlite3.connect("data/glacier_lakes.db")
    c = conn.cursor()
    
    query = "SELECT year, region, area_km2, geometry_geojson, method FROM lakes"
    params = []
    conditions = ["method = ?"]
    params.append(method)
    if region and region != "All":
        conditions.append("region = ?")
        params.append(region)
    if year:
        conditions.append("year = ?")
        params.append(year)
    query += " WHERE " + " AND ".join(conditions)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    features = []
    for r in rows:
        try:
            geom = __import__('json').loads(r[3])
        except:
            continue
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "year": r[0],
                "region": r[1],
                "area_km2": r[2],
                "method": r[4]
            }
        })
    return JSONResponse(content={"type": "FeatureCollection", "features": features})

@app.get("/regions")
def get_regions():
    """Get available regions and years for filter dropdowns."""
    conn = sqlite3.connect("data/glacier_lakes.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT region FROM lakes ORDER BY region")
    regions = [r[0] for r in c.fetchall()]
    c.execute("SELECT DISTINCT year FROM lakes ORDER BY year")
    years = [r[0] for r in c.fetchall()]
    conn.close()
    return JSONResponse(content={"regions": regions, "years": years})

@app.get("/lake-timeline")
def lake_timeline(lat: float, lon: float, region: str):
    """Find all lake polygons across years that overlap a given point.
    Returns one polygon per year showing how the lake boundary changed."""
    import json as json_mod
    from shapely.geometry import shape, Point
    
    conn = sqlite3.connect("data/glacier_lakes.db")
    c = conn.cursor()
    c.execute("""SELECT year, area_km2, geometry_geojson 
                 FROM lakes 
                 WHERE method='unet' AND region=?
                 ORDER BY year""", (region,))
    rows = c.fetchall()
    conn.close()
    
    target = Point(lon, lat)
    
    # For each year, find the lake whose polygon contains or is closest to the clicked point
    years_seen = {}
    for year, area, geom_json in rows:
        try:
            geom = shape(json_mod.loads(geom_json))
            if geom.contains(target) or geom.distance(target) < 0.005:  # ~500m tolerance
                dist = geom.distance(target)
                if year not in years_seen or dist < years_seen[year]['dist']:
                    years_seen[year] = {
                        'year': year,
                        'area_km2': area,
                        'geometry': json_mod.loads(geom_json),
                        'dist': dist
                    }
        except:
            continue
    
    features = []
    for year in sorted(years_seen.keys()):
        entry = years_seen[year]
        features.append({
            "type": "Feature",
            "geometry": entry['geometry'],
            "properties": {
                "year": entry['year'],
                "area_km2": entry['area_km2']
            }
        })
    
    return JSONResponse(content={"type": "FeatureCollection", "features": features})

class PredictRequest(BaseModel):
    bbox: list[float]
    year: int

model = None
device = None

@app.on_event("startup")
def startup_event():
    global model, device
    init_ee()
    device = get_device()
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()

@app.post("/predict")
def predict(req: PredictRequest):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tif_path = tmp.name
        
    try:
        fetch_sentinel_median(req.bbox, req.year, tif_path)
        img, transform_mat, crs = load_and_scale_geotiff(tif_path)
        mask = predict_image(model, img, device)
        polygons = extract_polygons(mask, transform_mat, crs)
        
        features = []
        for p in polygons:
            features.append({
                "type": "Feature",
                "geometry": p["geometry"],
                "properties": {
                    "area_km2": p["area_km2"]
                }
            })
            
        return {
            "type": "FeatureCollection",
            "features": features
        }
    finally:
        if os.path.exists(tif_path):
            os.remove(tif_path)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
