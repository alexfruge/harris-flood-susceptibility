"""
align.py — Reproject, resample, and clip all rasters to a common 30 m reference
grid (EPSG:32615) derived from the merged DEM.

Outputs (all written to DATA_PROCESSED):
    dem_harris_aligned.tif      — elevation, bilinear
    nlcd_harris_aligned.tif     — land cover, nearest-neighbour (categorical)
    rainfall_mean_aligned.tif   — CHIRPS annual mean, bilinear

The reference transform/shape is derived once from the DEM and reused for every
layer so all rasters share an identical grid.

Usage
-----
    from src.processing.align import run
    run()
    # or from the shell:
    python -m src.processing.align
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

from src.config import (
    BBOX_WGS84,
    CRS as TARGET_CRS_STR,
    DATA_PROCESSED,
    DEM_MERGED,
    NLCD_RAW,
    CHIRPS_MEAN,
    RESOLUTION_M,
)

log = logging.getLogger(__name__)

TARGET_CRS = CRS.from_string(TARGET_CRS_STR)
NODATA_FLOAT = -9999.0

# ── Output paths ──────────────────────────────────────────────────────────────
DEM_ALIGNED      = DATA_PROCESSED / "dem_harris_aligned.tif"
NLCD_ALIGNED     = DATA_PROCESSED / "nlcd_harris_aligned.tif"
RAINFALL_ALIGNED = DATA_PROCESSED / "rainfall_mean_aligned.tif"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reference_grid() -> tuple[rasterio.transform.Affine, int, int]:
    """
    Derive the canonical (transform, width, height) for the 30 m EPSG:32615 grid
    from the merged DEM bounding box.

    Returns
    -------
    transform : Affine
    width     : int
    height    : int
    """
    west, south, east, north = BBOX_WGS84
    transform, width, height = calculate_default_transform(
        src_crs=CRS.from_epsg(4326),
        dst_crs=TARGET_CRS,
        width=1,            # dummy — we override resolution below
        height=1,
        left=west,
        bottom=south,
        right=east,
        top=north,
        resolution=RESOLUTION_M,
    )
    log.info(
        "Reference grid: %d × %d pixels @ %.0f m  CRS=%s",
        width, height, RESOLUTION_M, TARGET_CRS_STR,
    )
    return transform, width, height


def _align_raster(
    src_path: Path,
    dst_path: Path,
    ref_transform: rasterio.transform.Affine,
    ref_width: int,
    ref_height: int,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = NODATA_FLOAT,
    dtype: str = "float32",
) -> Path:
    """
    Reproject *src_path* onto the reference grid and write to *dst_path*.

    Parameters
    ----------
    src_path    : source raster (any CRS / resolution)
    dst_path    : output path
    ref_*       : reference grid parameters from _reference_grid()
    resampling  : rasterio.enums.Resampling enum value
    nodata      : output nodata value
    dtype       : output numpy dtype string
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        n_bands = src.count
        src_nodata = src.nodata

        dst_profile = {
            "driver": "GTiff",
            "crs": TARGET_CRS,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height,
            "count": n_bands,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 2,
        }

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for band_idx in range(1, n_bands + 1):
                dest_array = np.full(
                    (ref_height, ref_width), nodata, dtype=dtype
                )
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=dest_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=ref_transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=nodata,
                    resampling=resampling,
                )
                dst.write(dest_array.astype(dtype), band_idx)

    log.info("Aligned → %s", dst_path)
    return dst_path


# ── Public API ────────────────────────────────────────────────────────────────

def align_dem(
    ref_transform: rasterio.transform.Affine,
    ref_width: int,
    ref_height: int,
) -> Path:
    """Align the merged DEM (elevation) to the reference grid."""
    if DEM_ALIGNED.exists():
        log.info("DEM already aligned: %s", DEM_ALIGNED)
        return DEM_ALIGNED
    return _align_raster(
        DEM_MERGED, DEM_ALIGNED,
        ref_transform, ref_width, ref_height,
        resampling=Resampling.bilinear,
        dtype="float32",
    )


def align_nlcd(
    ref_transform: rasterio.transform.Affine,
    ref_width: int,
    ref_height: int,
) -> Path:
    """Align NLCD land-cover (categorical → nearest-neighbour)."""
    if NLCD_ALIGNED.exists():
        log.info("NLCD already aligned: %s", NLCD_ALIGNED)
        return NLCD_ALIGNED
    return _align_raster(
        NLCD_RAW, NLCD_ALIGNED,
        ref_transform, ref_width, ref_height,
        resampling=Resampling.nearest,
        nodata=0,           # NLCD uses 0 as background
        dtype="uint8",
    )


def align_rainfall(
    ref_transform: rasterio.transform.Affine,
    ref_width: int,
    ref_height: int,
) -> Path:
    """Align CHIRPS annual-mean rainfall to the reference grid."""
    if RAINFALL_ALIGNED.exists():
        log.info("Rainfall already aligned: %s", RAINFALL_ALIGNED)
        return RAINFALL_ALIGNED
    return _align_raster(
        CHIRPS_MEAN, RAINFALL_ALIGNED,
        ref_transform, ref_width, ref_height,
        resampling=Resampling.bilinear,
        dtype="float32",
    )


def run() -> dict[str, Path]:
    """Run the full alignment pipeline. Returns a dict of output paths."""
    log.info("=== align.py: aligning all rasters to 30 m EPSG:32615 grid ===")
    ref_transform, ref_width, ref_height = _reference_grid()

    results = {
        "dem":      align_dem(ref_transform, ref_width, ref_height),
        "nlcd":     align_nlcd(ref_transform, ref_width, ref_height),
        "rainfall": align_rainfall(ref_transform, ref_width, ref_height),
    }

    log.info("=== align.py: done — %d rasters aligned ===", len(results))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
