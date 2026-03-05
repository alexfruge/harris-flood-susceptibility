"""
streams.py — Download NHDPlus HR flowlines for HUC4 1207 (San Jacinto/Galveston).

The zip (~600 MB) contains a file geodatabase.  We extract the NHDFlowline
layer, filter to Harris County's bbox, and save as a GeoPackage.
"""

import logging
import zipfile
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import box

from src.config import NHD_DOWNLOAD_URL, NHD_RAW_DIR, STREAMS_SHP, BBOX_WGS84

log = logging.getLogger(__name__)
NHD_ZIP = NHD_RAW_DIR / "NHDPLUS_H_1207_HU4_GDB.zip"
NHD_GDB = NHD_RAW_DIR / "NHDPLUS_H_1207_HU4_GDB.gdb"


def download_zip() -> Path:
    if NHD_ZIP.exists():
        log.info("NHD zip already downloaded: %s", NHD_ZIP)
        return NHD_ZIP

    NHD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading NHDPlus HR HUC4-1207 (~600 MB) …")
    resp = requests.get(NHD_DOWNLOAD_URL, stream=True, timeout=600)
    resp.raise_for_status()

    with open(NHD_ZIP, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=4 << 20):
            fh.write(chunk)

    log.info("NHD zip saved → %s (%.0f MB)", NHD_ZIP, NHD_ZIP.stat().st_size / 1e6)
    return NHD_ZIP


def extract_gdb() -> Path:
    if NHD_GDB.exists():
        log.info("NHD GDB already extracted: %s", NHD_GDB)
        return NHD_GDB

    log.info("Extracting NHD zip …")
    with zipfile.ZipFile(NHD_ZIP, "r") as zf:
        zf.extractall(NHD_RAW_DIR)
    log.info("Extracted to %s", NHD_RAW_DIR)
    return NHD_GDB


def clip_to_county() -> Path:
    out = Path(str(STREAMS_SHP).replace(".shp", ".gpkg"))
    if out.exists():
        log.info("Clipped streams already exist: %s", out)
        return out

    bbox = box(*BBOX_WGS84)
    log.info("Reading NHDFlowline layer …")
    gdf = gpd.read_file(NHD_GDB, layer="NHDFlowline", bbox=BBOX_WGS84)
    gdf = gdf[gdf.geometry.intersects(bbox)].copy()
    gdf.to_file(out, driver="GPKG")
    log.info("Clipped flowlines saved → %s (%d features)", out, len(gdf))
    return out


def run() -> Path:
    download_zip()
    extract_gdb()
    return clip_to_county()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
