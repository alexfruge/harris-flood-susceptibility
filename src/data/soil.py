"""
soil.py — Download USDA SSURGO soil data for Harris County (TX201).

The Soil Data Access WFS returns a GeoJSON/GML feature collection
with mukey, drclassdcd (drainage class), ksat_l/r/h (saturated hydraulic
conductivity) and other attributes for every map unit polygon.

We save the raw WFS response, then convert it to a GeoPackage for later
rasterisation by src/processing/features.py.
"""

import logging
import zipfile
from pathlib import Path

import requests
import geopandas as gpd

from src.config import SSURGO_WFS_URL, SOIL_RAW_DIR, SOIL_SHP

log = logging.getLogger(__name__)

# Soil Data Access tabular query for ksat representative value (μm/s)
SSURGO_TAB_URL = (
    "https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService.asmx"
    "/RunQuery?query=SELECT+mu.mukey,+c.ksat_r+FROM+mapunit+mu"
    "+INNER+JOIN+component+c+ON+mu.mukey=c.mukey"
    "+INNER+JOIN+legend+l+ON+mu.lkey=l.lkey"
    "+WHERE+l.areasymbol='TX201'+AND+c.majcompflag='Yes'"
)


def download_spatial() -> Path:
    SOIL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    gpkg = SOIL_RAW_DIR / "harris_soil.gpkg"

    if gpkg.exists():
        log.info("Soil spatial file already exists: %s", gpkg)
        return gpkg

    from src.config import BBOX_WGS84
    west, south, east, north = BBOX_WGS84
    url = SSURGO_WFS_URL.format(west=west, south=south, east=east, north=north)

    log.info("Downloading SSURGO spatial data …")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    gdf.to_file(gpkg, driver="GPKG")
    log.info("Soil polygons saved → %s (%d features)", gpkg, len(gdf))
    return gpkg


def download_tabular() -> Path:
    """Download ksat tabular data via SSURGO REST POST endpoint."""
    gpkg = SOIL_RAW_DIR / "harris_soil.gpkg"
    if not gpkg.exists():
        download_spatial()

    csv_path = SOIL_RAW_DIR / "harris_ksat.csv"
    if csv_path.exists():
        log.info("ksat CSV already exists: %s", csv_path)
        return csv_path

    import pandas as pd

    query = (
        "SELECT mu.mukey, AVG(CAST(ch.ksat_r AS FLOAT)) AS ksat_r "
        "FROM chorizon ch "
        "INNER JOIN component co ON ch.cokey=co.cokey "
        "INNER JOIN mapunit mu ON co.mukey=mu.mukey "
        "INNER JOIN legend l ON mu.lkey=l.lkey "
        "WHERE l.areasymbol='TX201' AND co.majcompflag='Yes' "
        "GROUP BY mu.mukey"
    )

    log.info("Downloading ksat tabular data …")
    resp = requests.post(
        "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
        json={"query": query, "format": "JSON"},
        timeout=120,
    )
    resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data["Table"], columns=["mukey", "ksat_r"])
    df.to_csv(csv_path, index=False)
    log.info("ksat CSV saved → %s (%d rows)", csv_path, len(df))
    return csv_path


def run() -> Path:
    download_spatial()
    return download_tabular()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
