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
    """Download SSURGO spatial polygons via WFS → GeoPackage."""
    SOIL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    gpkg = SOIL_RAW_DIR / "harris_soil.gpkg"

    if gpkg.exists():
        log.info("Soil spatial file already exists: %s", gpkg)
        return gpkg

    log.info("Downloading SSURGO spatial data from Soil Data Access WFS …")
    resp = requests.get(SSURGO_WFS_URL, timeout=300)
    resp.raise_for_status()

    # The WFS returns GML; geopandas can read it directly from bytes via fiona
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
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
    """Download ksat tabular data and join to spatial polygons."""
    gpkg = SOIL_RAW_DIR / "harris_soil.gpkg"
    if not gpkg.exists():
        download_spatial()

    csv_path = SOIL_RAW_DIR / "harris_ksat.csv"
    if csv_path.exists():
        log.info("ksat CSV already exists: %s", csv_path)
        return csv_path

    log.info("Downloading ksat tabular data …")
    resp = requests.get(SSURGO_TAB_URL, timeout=120)
    resp.raise_for_status()

    # Response is XML; parse with pandas
    import io
    import pandas as pd
    from xml.etree import ElementTree as ET

    root = ET.fromstring(resp.content)
    ns   = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    rows = []
    for row in root.iter("{urn:schemas-microsoft-com:office:spreadsheet}Row"):
        cells = [c.text for c in row.iter("{urn:schemas-microsoft-com:office:spreadsheet}Data")]
        rows.append(cells)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.to_csv(csv_path, index=False)
    log.info("ksat CSV saved → %s", csv_path)
    return csv_path


def run() -> Path:
    download_spatial()
    return download_tabular()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
