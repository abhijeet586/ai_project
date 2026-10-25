// ─── Risk Color Mapping ───
const RISK_COLORS = {
    CRITICAL: '#ef4444',
    HIGH: '#f97316',
    MODERATE: '#eab308',
    LOW: '#22c55e'
};

// ─── Year Colors for Timeline Overlay ───
const YEAR_COLORS = {
    2019: '#94a3b8',
    2020: '#818cf8',
    2021: '#38bdf8',
    2022: '#34d399',
    2023: '#eab308',
    2024: '#f97316',
    2025: '#ef4444'
};

// ─── Initialize Leaflet Map (empty) ───
const map = L.map('map', { center: [27.95, 86.9], zoom: 10, zoomControl: false });
L.control.zoom({ position: 'bottomright' }).addTo(map);

// ─── Basemaps ───
const CARTO_KEY = 'cb1_4qpd_1_84b70d51ac58a0d0b66a07ed';

// Original CartoDB Dark Matter (authenticated with API key)
const cartoDarkLayer = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png?key=${CARTO_KEY}`, {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
});

// High-res Esri Satellite
const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
    maxZoom: 19
});

// CartoDB Voyager
const cartoVoyagerLayer = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${CARTO_KEY}`, {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
});

// OpenStreetMap
const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
});

// Default to original CartoDB Dark Matter
cartoDarkLayer.addTo(map);

// ─── Basemap Dropdown Switcher (Top Controls) ───
const basemapSelect = document.getElementById('basemap-select');
if (basemapSelect) {
    basemapSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        [cartoDarkLayer, satelliteLayer, cartoVoyagerLayer, osmLayer].forEach(layer => {
            if (map.hasLayer(layer)) map.removeLayer(layer);
        });

        if (val === 'carto-dark') cartoDarkLayer.addTo(map);
        else if (val === 'satellite') satelliteLayer.addTo(map);
        else if (val === 'carto-voyager') cartoVoyagerLayer.addTo(map);
        else if (val === 'osm') osmLayer.addTo(map);
    });
}


// ─── Drawing Controls ───
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
map.addControl(new L.Control.Draw({
    draw: {
        polygon: false, polyline: false, circle: false, marker: false, circlemarker: false,
        rectangle: { shapeOptions: { color: '#38bdf8', weight: 2, fillOpacity: 0.1 } }
    },
    edit: { featureGroup: drawnItems }
}));

const statusEl = document.getElementById('status');
const yearSelect = document.getElementById('year-select');
const resultsContainer = document.getElementById('results-container');
const emptyState = document.getElementById('empty-state');

let detectedLayer = null;
let riskLayer = null;
let timelineLayer = null;
let historyChart = null;

// ─── Determine closest precomputed region ───
function getClosestRegion(lat, lon) {
    const regions = {
        Khumbu: { lat: 27.95, lon: 86.9 },
        Sikkim: { lat: 27.8, lon: 88.15 },
        Langtang: { lat: 28.25, lon: 85.5 },
        Manaslu: { lat: 28.45, lon: 84.6 },
        Bhutan: { lat: 27.95, lon: 89.4 }
    };
    let closest = null;
    let minDist = Infinity;
    for (const [name, center] of Object.entries(regions)) {
        const dist = Math.sqrt((lat - center.lat) ** 2 + (lon - center.lon) ** 2);
        if (dist < minDist) { minDist = dist; closest = name; }
    }
    return closest;
}

// ─── Quick Region Selector ───
const quickRegion = document.getElementById('quick-region');
quickRegion.addEventListener('change', async (e) => {
    const region = e.target.value;
    const centers = {
        Khumbu: [27.95, 86.9],
        Sikkim: [27.8, 88.15],
        Langtang: [28.25, 85.5],
        Manaslu: [28.45, 84.6],
        Bhutan: [27.95, 89.4]
    };
    if (centers[region]) {
        map.setView(centers[region], 11);
        
        // Show loading state while fetching region data
        emptyState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        document.getElementById('summary-module').classList.add('hidden'); // Hide scan summary since we didn't run a live scan
        
        await loadRiskForRegion(region);
        await loadHistoryForRegion(region);
    }
});

// ─── Clear Timeline Overlay ───
function clearTimeline() {
    if (timelineLayer) { map.removeLayer(timelineLayer); timelineLayer = null; }
    const panel = document.getElementById('timeline-panel');
    if (panel) panel.classList.add('hidden');
}

// ─── Show Lake Boundary Timeline ───
async function showLakeTimeline(lat, lon, region, lakeName) {
    clearTimeline();
    try {
        const res = await fetch(`/lake-timeline?lat=${lat}&lon=${lon}&region=${region}`);
        const data = await res.json();

        if (!data.features || data.features.length === 0) return;

        timelineLayer = L.geoJSON(data, {
            style: function (feature) {
                const yr = feature.properties.year;
                return {
                    color: YEAR_COLORS[yr] || '#fff',
                    weight: 2.5,
                    fillOpacity: 0.15,
                    fillColor: YEAR_COLORS[yr] || '#fff'
                };
            },
            onEachFeature: function (feature, lyr) {
                const p = feature.properties;
                lyr.bindPopup(
                    `<div style="font-family:Inter,sans-serif;color:#1e293b;">
                        <b>${p.year}</b><br>Area: ${p.area_km2.toFixed(4)} km²
                    </div>`
                );
            }
        }).addTo(map);

        // Populate timeline panel
        const panel = document.getElementById('timeline-panel');
        const list = document.getElementById('timeline-list');
        const title = document.getElementById('timeline-title');
        title.textContent = lakeName || 'Selected Lake';
        list.innerHTML = '';

        data.features.forEach(f => {
            const p = f.properties;
            const color = YEAR_COLORS[p.year] || '#fff';
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `
                <span class="timeline-dot" style="background:${color};"></span>
                <span class="timeline-year">${p.year}</span>
                <span class="timeline-area">${p.area_km2.toFixed(4)} km²</span>
            `;
            list.appendChild(item);
        });

        // Compute growth summary
        const first = data.features[0].properties;
        const last = data.features[data.features.length - 1].properties;
        const change = ((last.area_km2 - first.area_km2) / first.area_km2 * 100).toFixed(1);
        document.getElementById('timeline-summary').textContent =
            `${first.year}→${last.year}: ${change > 0 ? '+' : ''}${change}% change`;

        panel.classList.remove('hidden');
    } catch (err) {
        console.error('Timeline fetch failed', err);
    }
}

// ─── Main: Draw Rectangle → Full Scan ───
map.on(L.Draw.Event.CREATED, async function (e) {
    const layer = e.layer;
    drawnItems.clearLayers();
    drawnItems.addLayer(layer);

    if (detectedLayer) { map.removeLayer(detectedLayer); detectedLayer = null; }
    if (riskLayer) { map.removeLayer(riskLayer); riskLayer = null; }
    clearTimeline();

    const bounds = layer.getBounds();
    const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
    const year = parseInt(yearSelect.value);
    const centerLat = bounds.getCenter().lat;
    const centerLon = bounds.getCenter().lng;
    const closestRegion = getClosestRegion(centerLat, centerLon);

    statusEl.className = 'status-msg loading';
    statusEl.textContent = '🛰️ Fetching Sentinel-2 imagery & running U-Net inference...';
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bbox, year })
        });
        if (!res.ok) throw new Error('API Error');
        const predData = await res.json();

        detectedLayer = L.geoJSON(predData, {
            style: { color: '#38bdf8', weight: 2, fillOpacity: 0.4, fillColor: '#38bdf8' },
            onEachFeature: function (feature, lyr) {
                if (feature.properties && feature.properties.area_km2) {
                    const center = lyr.getBounds().getCenter();
                    lyr.on('click', function () {
                        showLakeTimeline(center.lat, center.lng, closestRegion, `Lake at ${center.lat.toFixed(3)}, ${center.lng.toFixed(3)}`);
                    });
                    lyr.bindPopup(
                        `<div style="font-family:Inter,sans-serif;color:#1e293b;">
                            <b>Detected Lake (${year})</b><br>
                            Area: ${feature.properties.area_km2.toFixed(4)} km²<br>
                            <em style="color:#818cf8;cursor:pointer;">Click to view boundary changes →</em>
                        </div>`
                    );
                }
            }
        }).addTo(map);

        const lakeCount = predData.features ? predData.features.length : 0;
        const totalArea = predData.features
            ? predData.features.reduce((sum, f) => sum + (f.properties.area_km2 || 0), 0) : 0;

        statusEl.className = 'status-msg success';
        statusEl.textContent = `✅ Scan complete — ${lakeCount} lake(s) detected. Click a lake to see its timeline.`;

        document.getElementById('lake-count').textContent = `${lakeCount} lakes`;
        document.getElementById('scan-summary').innerHTML = `
            <div class="stat-card">
                <div class="stat-value" style="color:#38bdf8;">${lakeCount}</div>
                <div class="stat-label">Lakes Found</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#38bdf8;">${totalArea.toFixed(3)}</div>
                <div class="stat-label">Total Area (km²)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#c084fc;">${year}</div>
                <div class="stat-label">Target Year</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#c084fc;">${closestRegion}</div>
                <div class="stat-label">Region</div>
            </div>
        `;

        await loadRiskForRegion(closestRegion);
        await loadHistoryForRegion(closestRegion);
        resultsContainer.classList.remove('hidden');
    } catch (err) {
        console.error(err);
        statusEl.className = 'status-msg error';
        statusEl.textContent = '❌ Error: Box too large or GEE quota exceeded. Try a smaller area.';
    }
});

// ─── Load Risk Alerts ───
async function loadRiskForRegion(region) {
    try {
        const res = await fetch(`/risk?region=${region}`);
        const data = await res.json();

        if (riskLayer) { map.removeLayer(riskLayer); riskLayer = null; }

        const riskList = document.getElementById('risk-list');
        const alertCount = document.getElementById('alert-count');

        if (!data.features || data.features.length === 0) {
            riskList.innerHTML = '<div class="no-risk-msg">No historical risk data for this region.</div>';
            alertCount.textContent = '0';
            return;
        }

        riskLayer = L.geoJSON(data, {
            style: function (feature) {
                const color = RISK_COLORS[feature.properties.risk_level] || '#22c55e';
                return { color, weight: 1.5, fillOpacity: 0.2, fillColor: color, dashArray: '4 3' };
            },
            onEachFeature: function (feature, lyr) {
                const p = feature.properties;
                const growthStr = p.growth_pct > 0 ? `+${p.growth_pct.toFixed(1)}%` : `${p.growth_pct.toFixed(1)}%`;
                // Click to show timeline
                lyr.on('click', function () {
                    showLakeTimeline(p.centroid_lat, p.centroid_lon, p.region, p.lake_id);
                });
                lyr.bindPopup(
                    `<div style="font-family:Inter,sans-serif;color:#1e293b;min-width:180px;">
                        <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${p.lake_id}</div>
                        <div style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;color:white;background:${RISK_COLORS[p.risk_level]};margin-bottom:8px;">
                            ${p.risk_level} — Score ${p.risk_score}
                        </div>
                        <table style="font-size:12px;width:100%;border-collapse:collapse;">
                            <tr><td style="padding:2px 0;color:#64748b;">Area</td><td style="text-align:right;">${p.area_km2.toFixed(4)} km²</td></tr>
                            <tr><td style="padding:2px 0;color:#64748b;">Growth</td><td style="text-align:right;font-weight:600;color:${p.growth_pct > 0 ? '#ef4444' : '#22c55e'}">${growthStr}</td></tr>
                        </table>
                        <div style="margin-top:8px;color:#818cf8;font-size:11px;font-style:italic;">Click to view boundary changes →</div>
                    </div>`
                );
            }
        }).addTo(map);

        const critHigh = data.features.filter(f => f.properties.risk_level === 'CRITICAL' || f.properties.risk_level === 'HIGH');
        alertCount.textContent = `${critHigh.length} alerts`;
        riskList.innerHTML = '';

        data.features.slice(0, 12).forEach(f => {
            const p = f.properties;
            const card = document.createElement('div');
            card.className = 'risk-card';
            card.innerHTML = `
                <div class="lake-info">
                    <div class="lake-name">${p.lake_id}</div>
                    <div class="lake-details">${p.region} · ${p.area_km2.toFixed(3)} km² · ${p.growth_pct > 0 ? '+' : ''}${p.growth_pct.toFixed(1)}%</div>
                </div>
                <div class="risk-badge ${p.risk_level}">${p.risk_level}</div>
            `;
            card.addEventListener('click', () => {
                map.setView([p.centroid_lat, p.centroid_lon], 15);
                showLakeTimeline(p.centroid_lat, p.centroid_lon, p.region, p.lake_id);
            });
            riskList.appendChild(card);
        });
    } catch (err) {
        console.error('Failed to load risk data', err);
    }
}

// ─── Load History Chart ───
async function loadHistoryForRegion(region) {
    try {
        const res = await fetch('/history');
        const rawData = await res.json();
        const data = rawData.filter(r => r.region === region);

        const years = [...new Set(rawData.map(r => r.year))].sort();
        const unetData = {}, ndwiData = {};
        years.forEach(y => { unetData[y] = 0; ndwiData[y] = 0; });
        data.forEach(row => {
            if (row.method === 'unet') unetData[row.year] += row.area_km2;
            else if (row.method === 'ndwi') ndwiData[row.year] += row.area_km2;
        });

        if (historyChart) historyChart.destroy();

        const ctx = document.getElementById('historyChart').getContext('2d');
        historyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'U-Net Area (km²)',
                        data: years.map(y => unetData[y]),
                        borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.12)',
                        borderWidth: 2.5, tension: 0.4, fill: true,
                        pointBackgroundColor: '#f43f5e', pointRadius: 4, pointHoverRadius: 6
                    },
                    {
                        label: 'NDWI Baseline (km²)',
                        data: years.map(y => ndwiData[y]),
                        borderColor: '#94a3b8', borderWidth: 1.5,
                        borderDash: [6, 4], tension: 0.4, fill: false,
                        pointRadius: 3, pointBackgroundColor: '#94a3b8'
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index', intersect: false,
                        backgroundColor: 'rgba(10, 15, 30, 0.95)',
                        titleColor: '#fff', bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
                        padding: 12, cornerRadius: 8
                    }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 11 } } },
                    x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 11 } } }
                },
                interaction: { mode: 'nearest', axis: 'x', intersect: false }
            }
        });
    } catch (err) {
        console.error('Failed to load history', err);
    }
}

// ─── Auto-load default region (Khumbu) on start ───
window.addEventListener('DOMContentLoaded', () => {
    if (quickRegion) {
        quickRegion.value = 'Khumbu';
        quickRegion.dispatchEvent(new Event('change'));
    }
});

