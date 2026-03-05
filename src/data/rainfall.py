"""
rainfall.py — Download CHIRPS v3 annual rainfall GeoTIFFs (1981–2023)
and compute a long-term mean clipped to Harris County.

Files are ~10 MB each; total ~420 MB for the full time series.
"""

import logging
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from src.config import (
    CHIRPS_BASE_URL,
    CHIRPS_YEARS,
    CHIRPS_RAW_DIR,
    CHIRPS_MEAN,
    BBOX_WGS84,
)

log = logging.getLogger(__name__)


def _tile_url(year: int) -> str:
    """CHIRPS v3 global annual GeoTIFF URL for a given year."""
    return f"{CHIRPS_BASE_URL}/chirps-v3.0.{year}.tif.gz"


def download_year(year: int) -> Path:
    """Download (and gunzip) one annual CHIRPS tile."""
    import gzip, shutil

    CHIRPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHIRPS_RAW_DIR / f"chirps_{year}.tif"
    gz   = dest.with_suffix(".tif.gz")

    if dest.exists():
        return dest

    url  = _tile_url(year)
    log.info("Downloading CHIRPS %d …", year)
    resp = requests.get(url, stream=True, timeout=300)
    if resp.status_code == 404:
        # Some years may use a slightly different naming pattern
        log.warning("CHIRPS %d not found at %s", year, url)
        return None
    resp.raise_for_status()

    with open(gz, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)

    with gzip.open(gz, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz.unlink()
    return dest


def compute_mean(year_paths: list[Path]) -> Path:
    """Average all annual rasters and clip to Harris County bbox."""
    valid = [p for p in year_paths if p is not None and p.exists()]
    if not valid:
        raise RuntimeError("No CHIRPS annual files found.")

    clip_geom = [box(*BBOX_WGS84).__geo_interface__]

    # Read first file to get metadata and clip transform
    with rasterio.open(valid[0]) as src:
        out_img, out_transform = mask(src, clip_geom, crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": out_img.shape[1],
            "width":  out_img.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "dtype": "float32",
        })
        nodata = src.nodata if src.nodata is not None else -9999.0
        meta["nodata"] = nodata

    accumulator = np.zeros((meta["height"], meta["width"]), dtype=np.float64)
    count       = np.zeros_like(accumulator, dtype=np.int32)

    for p in valid:
        with rasterio.open(p) as src:
            arr, _ = mask(src, clip_geom, crop=True)
            data = arr[0].astype(np.float64)
            valid_mask = data != nodata
            accumulator[valid_mask] += data[valid_mask]
            count[valid_mask] += 1

    mean = np.where(count > 0, accumulator / count, nodata).astype(np.float32)

    CHIRPS_MEAN.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(CHIRPS_MEAN, "w", **meta) as dst:
        dst.write(mean, 1)

    log.info("CHIRPS mean saved → %s (years used: %d)", CHIRPS_MEAN, len(valid))
    return CHIRPS_MEAN


def run() -> Path:
    if CHIRPS_MEAN.exists():
        log.info("CHIRPS mean already exists: %s", CHIRPS_MEAN)
        return CHIRPS_MEAN

    paths = [download_year(y) for y in CHIRPS_YEARS]
    return compute_mean(paths)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
