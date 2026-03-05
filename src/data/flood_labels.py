"""
flood_labels.py — Download FEMA National Flood Hazard Layer (NFHL) for Harris County.

The NFHL is distributed as a file geodatabase (.gdb) zip per county.
We extract the S_FLD_HAZ_AR (flood hazard area) layer and export it
as GeoJSON for rasterization in processing/labels.py.

Fallback: if the bulk download link is unavailable, query the FEMA NFHL
REST MapServer for Harris County features.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

from src.config import (
    BBOX_GEO,
    FEMA_FIPS,
    FEMA_NFHL_DIRECT,
    FEMA_NFHL_URL,
    LABELS_DIR,
    RAW_LABELS_DIR,
)

log = logging.getLogger(__name__)

FEMA_ZIP_PATH   = RAW_LABELS_DIR / f"NFHL_{FEMA_FIPS}.zip"
FEMA_GJ_PATH    = LABELS_DIR / "fema_flood_zones_harris.geojson"
FLOOD_LAYER     = "S_FLD_HAZ_AR"


def download_nfhl_zip(out_dir: Path = RAW_LABELS_DIR) -> Path | None:
    """Attempt bulk GDB zip download from FEMA MSC. Returns None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if FEMA_ZIP_PATH.exists():
        log.info("NFHL zip already cached: %s", FEMA_ZIP_PATH)
        return FEMA_ZIP_PATH

    log.info("Attempting FEMA NFHL bulk download for FIPS %s ...", FEMA_FIPS)
    try:
        with requests.get(FEMA_NFHL_URL, stream=True, timeout=300,
                          allow_redirects=True) as r:
            r.raise_for_status()
            if "html" in r.headers.get("Content-Type", ""):
                log.warning("FEMA MSC returned HTML page — bulk download requires manual portal access.")
                return None
            with open(FEMA_ZIP_PATH, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        log.info("NFHL zip saved -> %s", FEMA_ZIP_PATH)
        return FEMA_ZIP_PATH
    except Exception as exc:
        log.warning("Bulk NFHL download failed: %s", exc)
        return None


def extract_flood_layer_from_gdb(zip_path: Path, out_dir: Path = RAW_LABELS_DIR) -> gpd.GeoDataFrame:
    """Extract S_FLD_HAZ_AR from the NFHL .gdb inside the zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    gdb_dirs = list(out_dir.glob("*.gdb"))
    if not gdb_dirs:
        raise FileNotFoundError("No .gdb found after extracting NFHL zip.")
    log.info("Reading %s from %s ...", FLOOD_LAYER, gdb_dirs[0].name)
    return gpd.read_file(gdb_dirs[0], layer=FLOOD_LAYER)


def fetch_flood_zones_rest(bbox: tuple = BBOX_GEO) -> gpd.GeoDataFrame:
    """Query FEMA NFHL REST MapServer (layer 28) with pagination."""
    minx, miny, maxx, maxy = bbox
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 2000,
    }
    all_features: list = []
    offset = 0
    while True:
        params["resultOffset"] = offset
        log.info("Fetching FEMA REST page at offset %d ...", offset)
        r = requests.get(FEMA_NFHL_DIRECT, params=params, timeout=120)
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            break
        all_features.extend(features)
        if len(features) < 2000:
            break
        offset += 2000

    if not all_features:
        raise RuntimeError("FEMA REST API returned no features.")
    fc = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.read_file(json.dumps(fc))
    log.info("FEMA REST: retrieved %d flood zone features.", len(gdf))
    return gdf


def get_flood_zones(out_path: Path = FEMA_GJ_PATH) -> gpd.GeoDataFrame:
    """Get FEMA flood zones — tries bulk GDB, falls back to REST API."""
    if out_path.exists():
        log.info("Flood zones already saved: %s", out_path)
        return gpd.read_file(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = download_nfhl_zip()
    if zip_path is not None:
        try:
            gdf = extract_flood_layer_from_gdb(zip_path)
        except Exception as exc:
            log.warning("GDB extraction failed (%s) — using REST fallback.", exc)
            gdf = fetch_flood_zones_rest()
    else:
        gdf = fetch_flood_zones_rest()

    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Flood zones saved -> %s (%d polygons)", out_path, len(gdf))
    return gdf


def run(out_dir: Path = RAW_LABELS_DIR) -> Path:
    get_flood_zones()
    return FEMA_GJ_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(f"Flood labels ready: {result}")
