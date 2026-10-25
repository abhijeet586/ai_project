"""
GLOF Risk Scoring Engine
Matches lakes across years by spatial overlap, computes area growth rates,
and assigns a composite GLOF risk score (0-100).
"""
import os
import sys
import sqlite3
import json
from shapely.geometry import shape

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def compute_risk_scores():
    db_path = "data/glacier_lakes.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create risk table
    c.execute('''CREATE TABLE IF NOT EXISTS risk_scores
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  lake_id TEXT,
                  region TEXT,
                  latest_year INTEGER,
                  latest_area_km2 REAL,
                  earliest_area_km2 REAL,
                  growth_rate_pct REAL,
                  risk_score REAL,
                  risk_level TEXT,
                  centroid_lon REAL,
                  centroid_lat REAL,
                  geometry_geojson TEXT)''')
    c.execute("DELETE FROM risk_scores")
    
    # Get all U-Net lakes grouped by region
    c.execute("SELECT DISTINCT region FROM lakes WHERE method='unet'")
    regions = [r[0] for r in c.fetchall()]
    
    for region in regions:
        # Get all lakes for this region, ordered by year
        c.execute("""SELECT year, area_km2, geometry_geojson 
                     FROM lakes 
                     WHERE method='unet' AND region=?
                     ORDER BY year""", (region,))
        rows = c.fetchall()
        
        # Group lakes by year
        lakes_by_year = {}
        for year, area, geom_json in rows:
            if year not in lakes_by_year:
                lakes_by_year[year] = []
            try:
                geom = shape(json.loads(geom_json))
                lakes_by_year[year].append({
                    'year': year,
                    'area_km2': area,
                    'geometry': geom,
                    'geom_json': geom_json
                })
            except:
                continue
        
        years = sorted(lakes_by_year.keys())
        if len(years) < 2:
            continue
            
        latest_year = max(years)
        earliest_year = min([y for y in years if len(lakes_by_year.get(y, [])) > 0])
        
        if latest_year not in lakes_by_year:
            continue
            
        # For each lake in the latest year, try to find its match in earlier years
        lake_counter = 0
        for lake in lakes_by_year[latest_year]:
            lake_counter += 1
            lake_id = f"{region}_{lake_counter}"
            
            # Find earliest matching lake by spatial overlap
            earliest_match = None
            for yr in sorted(years):
                if yr == latest_year:
                    continue
                for candidate in lakes_by_year.get(yr, []):
                    try:
                        if lake['geometry'].intersects(candidate['geometry']):
                            intersection = lake['geometry'].intersection(candidate['geometry']).area
                            union = lake['geometry'].union(candidate['geometry']).area
                            iou = intersection / union if union > 0 else 0
                            if iou > 0.1:  # At least 10% overlap
                                if earliest_match is None or candidate['year'] < earliest_match['year']:
                                    earliest_match = candidate
                    except:
                        continue
            
            # Compute growth rate
            if earliest_match and earliest_match['area_km2'] > 0.001:
                growth_pct = ((lake['area_km2'] - earliest_match['area_km2']) / earliest_match['area_km2']) * 100
                earliest_area = earliest_match['area_km2']
                time_span = latest_year - earliest_match['year']
            else:
                growth_pct = 0
                earliest_area = lake['area_km2']
                time_span = 1
            
            # Composite risk score (0-100)
            # Factor 1: Annual growth rate (0-50 points)
            annual_growth = growth_pct / max(time_span, 1)
            growth_score = min(50, max(0, annual_growth * 2.5))
            
            # Factor 2: Absolute size (0-30 points) — larger = more energy
            size_score = min(30, lake['area_km2'] * 60)
            
            # Factor 3: Rapid recent expansion bonus (0-20 points)
            expansion_bonus = 0
            if growth_pct > 20:
                expansion_bonus = min(20, (growth_pct - 20) * 0.5)
            
            risk_score = min(100, growth_score + size_score + expansion_bonus)
            
            # Risk level
            if risk_score >= 70:
                risk_level = "CRITICAL"
            elif risk_score >= 40:
                risk_level = "HIGH"
            elif risk_score >= 20:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            # Centroid
            try:
                centroid = lake['geometry'].centroid
                centroid_lon = centroid.x
                centroid_lat = centroid.y
            except:
                centroid_lon = 0
                centroid_lat = 0
            
            c.execute("""INSERT INTO risk_scores 
                         (lake_id, region, latest_year, latest_area_km2, earliest_area_km2,
                          growth_rate_pct, risk_score, risk_level, centroid_lon, centroid_lat,
                          geometry_geojson) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (lake_id, region, latest_year, lake['area_km2'], earliest_area,
                       round(growth_pct, 2), round(risk_score, 1), risk_level,
                       round(centroid_lon, 6), round(centroid_lat, 6), lake['geom_json']))
    
    conn.commit()
    
    # Print summary
    c.execute("SELECT risk_level, COUNT(*) FROM risk_scores GROUP BY risk_level ORDER BY risk_score DESC")
    print("\n" + "="*60)
    print("GLOF RISK ASSESSMENT SUMMARY")
    print("="*60)
    for level, count in c.fetchall():
        print(f"  {level:12s} : {count} lakes")
    
    c.execute("SELECT lake_id, region, latest_area_km2, growth_rate_pct, risk_score, risk_level FROM risk_scores WHERE risk_level IN ('CRITICAL', 'HIGH') ORDER BY risk_score DESC LIMIT 10")
    rows = c.fetchall()
    if rows:
        print(f"\nTop {len(rows)} Highest-Risk Lakes:")
        print(f"  {'Lake ID':<15s} {'Region':<10s} {'Area (km²)':<12s} {'Growth %':<10s} {'Score':<8s} {'Level'}")
        print("  " + "-"*65)
        for r in rows:
            print(f"  {r[0]:<15s} {r[1]:<10s} {r[2]:<12.4f} {r[3]:<10.1f} {r[4]:<8.1f} {r[5]}")
    
    print("="*60)
    conn.close()
    print("Risk scores saved to database.")

if __name__ == "__main__":
    compute_risk_scores()
