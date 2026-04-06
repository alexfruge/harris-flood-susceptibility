"""
flood_labels.py — Download FEMA National Flood Hazard Layer (NFHL) for Harris County.

The NFHL is distributed as a file geodatabase (.gdb) zip per county.
We extract the S_FLD_HAZ_AR (flood hazard area) layer and export it
as GeoJSON for rasterization in processing/labels.py.

Fallback: if the bulk download link is unavailable, query the FEMA NFHL
REST MapServer for Harris County features.

Zone scoring
------------
After download, a `flood_score` integer column (0–5) is added using
ZONE_SUBTY to disambiguate the two flavours of zone X:

    5  VE, V1–V30          Coastal high hazard (wave action)
    4  AE, A1–A30, AO, AH  High hazard — studied, with BFE
    3  A                   High hazard — unstudied, no BFE
    2  X (shaded), B, AR   Moderate / 0.2 % annual-chance hazard
    1  X (unshaded), C     Minimal hazard
    0  everything else     Outside designated zones
"""

from __future__ import annotations

import json
import logging
import zipfile
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# ---------------------------------------------------------------------------
# Zone → susceptibility score mapping
# ---------------------------------------------------------------------------
# Score 4 covers the A1–A30 numbered zones (pre-FIRM legacy codes).
_A_NUMBERED = {f"A{i}" for i in range(1, 31)}
_V_NUMBERED = {f"V{i}" for i in range(1, 31)}

# ZONE_SUBTY values that identify shaded X (moderate, 0.2%-annual-chance).
# All other X / B zones are treated as unshaded (score 1).
_X_MODERATE_SUBTYPES = {
    "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
    "0.2 PCT ANNUAL CHANCE FLOOD HAZARD CONTAINED IN CHANNEL",
    "AREA OF MINIMAL FLOOD HAZARD",   # occasionally mis-coded; keep moderate
    "FLOODWAY",                        # floodways inside X-shaded envelope
}


def assign_flood_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a ``flood_score`` column (int, 0-5) to *gdf* in-place and return it.

    Requires FLD_ZONE; uses ZONE_SUBTY when present to split zone X into
    shaded (score 2) vs unshaded (score 1).
    """
    fz = gdf["FLD_ZONE"].str.strip().str.upper().fillna("")

    # ZONE_SUBTY is present in the GDB layer and in the REST outFields;
    # fall back to empty string if the column is missing.
    if "ZONE_SUBTY" in gdf.columns:
        subty = gdf["ZONE_SUBTY"].str.strip().str.upper().fillna("")
    else:
        log.warning("ZONE_SUBTY column not found — zone X shaded/unshaded distinction unavailable; defaulting to score 1.")
        subty = pd.Series("", index=gdf.index)

    def _score(zone: str, sub: str) -> int:
        if zone in _V_NUMBERED or zone == "VE":
            return 5
        if zone in _A_NUMBERED or zone in ("AE", "AO", "AH"):
            return 4
        if zone == "A":
            return 3
        if zone in ("AR", "B"):
            return 2
        if zone == "X":
            return 2 if sub in _X_MODERATE_SUBTYPES else 1
        if zone == "C":
            return 1
        return 0

    gdf["flood_score"] = [
        _score(z, s) for z, s in zip(fz, subty)
    ]

    score_counts = gdf["flood_score"].value_counts().sort_index()
    log.info("flood_score distribution:\n%s", score_counts.to_string())

    return gdf

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
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

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
        "resultRecordCount": 100,
    }

    chunks: list[gpd.GeoDataFrame] = []
    offset = 0
    while True:
        params["resultOffset"] = offset
        log.info("Fetching FEMA REST page at offset %d ...", offset)
        r = session.get(FEMA_NFHL_DIRECT, params=params, timeout=120)
        r.raise_for_status()

        data = r.json()
        features = data.get("features") or []

        if not features:
            log.info("No features returned at offset %d — pagination complete.", offset)
            break

        # Parse each page individually — avoids one giant string at the end
        fc_page = {"type": "FeatureCollection", "features": features}
        chunks.append(gpd.read_file(json.dumps(fc_page)))

        if not data.get("exceededTransferLimit", False):
            break
        if len(features) < 100:
            break

        offset += 100
        time.sleep(0.5)

    if not chunks:
        raise RuntimeError("FEMA REST API returned no features.")

    gdf = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), crs=chunks[0].crs)
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

    gdf = assign_flood_scores(gdf)

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