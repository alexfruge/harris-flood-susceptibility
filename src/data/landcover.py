"""
landcover.py — Download NLCD 2021 land-cover raster for Harris County
via the MRLC Web Coverage Service (WCS).
"""

import logging
from pathlib import Path

import requests

from src.config import NLCD_WCS_URL, NLCD_RAW, BBOX_WGS84

log = logging.getLogger(__name__)


def download() -> Path:
    """Download NLCD 2021 GeoTIFF clipped to Harris County bbox."""
    if NLCD_RAW.exists():
        log.info("NLCD already downloaded: %s", NLCD_RAW)
        return NLCD_RAW

    west, south, east, north = BBOX_WGS84
    url = NLCD_WCS_URL.format(west=west, east=east, south=south, north=north)
    log.info("Requesting NLCD 2021 from MRLC WCS …")

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    NLCD_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(NLCD_RAW, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)

    log.info("NLCD saved → %s (%.1f MB)", NLCD_RAW, NLCD_RAW.stat().st_size / 1e6)
    return NLCD_RAW


def run() -> Path:
    return download()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
