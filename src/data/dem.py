"""
dem.py — Download Copernicus GLO-30 DEM tiles covering Harris County.

Tiles are served as Cloud-Optimised GeoTIFFs from a public AWS S3 bucket
(no credentials needed).  After download, all tiles are merged into a
single VRT / GeoTIFF clipped to the study-area bounding box.
"""

import logging
from pathlib import Path

import requests
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np

from src.config import (
    COP_DEM_S3_BASE,
    COP_DEM_TILES,
    DEM_RAW_DIR,
    DEM_MERGED,
    BBOX_WGS84,
)

log = logging.getLogger(__name__)


def download_tiles() -> list[Path]:
    """Download all GLO-30 tiles and return list of local paths."""
    DEM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for tile in COP_DEM_TILES:
        url  = COP_DEM_S3_BASE.format(**tile)
        dest = DEM_RAW_DIR / f"dem_{tile['lat']}_{tile['lon']}.tif"

        if dest.exists():
            log.info("DEM tile already exists, skipping: %s", dest.name)
            paths.append(dest)
            continue

        log.info("Downloading DEM tile: %s", url)
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            log.warning("Tile not found (404), skipping: %s", url)
            continue
        resp.raise_for_status()

        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        paths.append(dest)

    return paths


def merge_tiles(tile_paths: list[Path]) -> Path:
    """Merge tiles and clip to Harris County bbox → DEM_MERGED."""
    if not tile_paths:
        raise RuntimeError("No DEM tiles to merge.")

    west, south, east, north = BBOX_WGS84
    clip_geom = [box(west, south, east, north).__geo_interface__]

    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                  "width": mosaic.shape[2], "transform": transform,
                  "compress": "lzw", "tiled": True})

    # Write temporary mosaic then clip
    tmp = DEM_MERGED.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    # Clip to bbox
    with rasterio.open(tmp) as src:
        out_img, out_transform = mask(src, clip_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width":  out_img.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "tiled": True,
        })

    DEM_MERGED.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DEM_MERGED, "w", **out_meta) as dst:
        dst.write(out_img)

    tmp.unlink(missing_ok=True)
    log.info("DEM merged → %s", DEM_MERGED)
    return DEM_MERGED


def run() -> Path:
    """Entry-point: download + merge DEM tiles."""
    if DEM_MERGED.exists():
        log.info("Merged DEM already exists, skipping: %s", DEM_MERGED)
        return DEM_MERGED
    tiles = download_tiles()
    return merge_tiles(tiles)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
