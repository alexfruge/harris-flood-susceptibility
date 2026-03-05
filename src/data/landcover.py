"""
landcover.py — Download NLCD 2021 land-cover raster for Harris County
via the MRLC Web Coverage Service (WCS).
"""

import logging
from pathlib import Path

from src.config import NLCD_RAW, BBOX_WGS84

log = logging.getLogger(__name__)


MANUAL_INSTRUCTIONS = """
NLCD 2024 must be downloaded manually:
  1. Go to https://www.mrlc.gov/viewer/
  2. Draw a box roughly around Harris County
  3. Once the box is drawn, you can manually enter the exact coordinates:
        Latitude: {south} to {north}
        Longitude: {west} to {east}
        (Use the BBOX_WGS84 coordinates from src/config.py for precision.)
  4. Under "Download Contents", select "GeoTIFF"
  5. Select the "Land Cover" layer within "Annual NLCD"
  6. Set the year range to 2024 only
  7. Download and place the GeoTIFF at:
     {path}
"""

def download() -> Path:
    if NLCD_RAW.exists():
        log.info("NLCD already downloaded: %s", NLCD_RAW)
        return NLCD_RAW

    raise FileNotFoundError(MANUAL_INSTRUCTIONS.format(path=NLCD_RAW))

def run() -> Path:
    return download()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
