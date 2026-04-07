"""
features.py — Derive engineered features on the aligned 30 m EPSG:32615 grid.

Features produced
-----------------
slope           degrees, from aligned DEM via GDAL DEMProcessing (fallback: numpy)
twi             ln(flow_acc / tan(slope_rad)); richdem D8 preferred, else unit-area
dist_stream     Euclidean distance (m) from rasterised NHD flowlines
soil_ksat       SSURGO saturated hydraulic conductivity (μm/s) burned onto grid

All outputs share the reference grid defined in align.py (DEM_ALIGNED sets shape /
transform).  Missing values are written as NODATA_FLOAT = -9999.0.

Usage
-----
    from src.processing.features import run
    run()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

from src.config import (
    DATA_PROCESSED,
    NHD_RAW_DIR,
    SOIL_RAW_DIR,
    STREAMS_SHP,
    SLOPE_RASTER,
    TWI_RASTER,
    DIST_STREAM_RASTER,
    SOIL_RASTER,
)

log = logging.getLogger(__name__)

NODATA_FLOAT = -9999.0

# ── Derived from align.py outputs ────────────────────────────────────────────
DEM_ALIGNED      = DATA_PROCESSED / "dem_harris_aligned.tif"
SOIL_GPKG        = SOIL_RAW_DIR / "harris_soil.gpkg"
KSAT_CSV         = SOIL_RAW_DIR / "harris_ksat.csv"

# ── Helper: read reference grid meta ─────────────────────────────────────────

def _ref_meta() -> dict:
    """Return rasterio profile from the aligned DEM (reference grid)."""
    with rasterio.open(DEM_ALIGNED) as src:
        meta = src.meta.copy()
    meta.update(
        count=1,
        dtype="float32",
        nodata=NODATA_FLOAT,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
    )
    return meta


def _read_dem() -> tuple[np.ndarray, Affine]:
    with rasterio.open(DEM_ALIGNED) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata or NODATA_FLOAT
    dem[dem == nodata] = np.nan
    return dem, transform


# ── Slope ─────────────────────────────────────────────────────────────────────

def _slope_gdal(dem_path: Path, out_path: Path) -> bool:
    """Try GDAL DEMProcessing for slope. Returns True on success."""
    try:
        from osgeo import gdal  # type: ignore
        gdal.UseExceptions()
        opts = gdal.DEMProcessingOptions(computeEdges=True, alg="ZevenbergenThorne")
        gdal.DEMProcessing(str(out_path), str(dem_path), "slope", options=opts)
        log.info("Slope computed via GDAL DEMProcessing → %s", out_path)
        return True
    except Exception as exc:
        log.warning("GDAL slope failed (%s) — using numpy fallback.", exc)
        return False


def _slope_numpy(dem: np.ndarray, res: float, out_path: Path, meta: dict) -> None:
    """Finite-difference slope (degrees) when GDAL is unavailable."""
    # Pad with edge values so gradient is defined everywhere
    padded = np.pad(dem, 1, mode="edge")
    dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * res)
    dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * res)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")
    slope_deg[np.isnan(dem)] = NODATA_FLOAT

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(slope_deg, 1)
    log.info("Slope computed via numpy → %s", out_path)


def derive_slope() -> Path:
    if SLOPE_RASTER.exists():
        log.info("Slope already exists: %s", SLOPE_RASTER)
        return SLOPE_RASTER

    meta = _ref_meta()
    res = abs(meta["transform"].a)   # pixel width in metres

    # Try GDAL first; fall back to numpy
    if not _slope_gdal(DEM_ALIGNED, SLOPE_RASTER):
        dem, _ = _read_dem()
        _slope_numpy(dem, res, SLOPE_RASTER, meta)

    return SLOPE_RASTER


# ── TWI ───────────────────────────────────────────────────────────────────────

def _flow_acc_richdem(dem: np.ndarray) -> np.ndarray:
    """D8 flow accumulation using richdem (cells upslope of each cell)."""
    import richdem as rd  # type: ignore
    rdem = rd.rdarray(dem, no_data=np.nan)
    rd.FillDepressions(rdem, epsilon=True, in_place=True)
    acc = rd.FlowAccumulation(rdem, method="D8")
    return np.array(acc).astype("float64")


def _flow_acc_unit(dem: np.ndarray) -> np.ndarray:
    """Simple unit contributing area (each cell = 1) — no richdem needed."""
    log.warning(
        "richdem not available — using unit contributing area for TWI. "
        "Install richdem for more accurate flow accumulation."
    )
    return np.ones_like(dem, dtype="float64")


def derive_twi() -> Path:
    if TWI_RASTER.exists():
        log.info("TWI already exists: %s", TWI_RASTER)
        return TWI_RASTER

    dem, transform = _read_dem()
    res = abs(transform.a)   # cell size in metres

    # Flow accumulation (area in m²)
    try:
        acc = _flow_acc_richdem(dem)
        area_m2 = (acc + 1) * (res ** 2)   # +1 avoids log(0)
    except ImportError:
        acc = _flow_acc_unit(dem)
        area_m2 = (acc + 1) * (res ** 2)

    # Slope in radians from the aligned slope raster (or recompute)
    if SLOPE_RASTER.exists():
        with rasterio.open(SLOPE_RASTER) as s:
            slope_deg = s.read(1).astype("float64")
            slope_deg[slope_deg == NODATA_FLOAT] = np.nan
    else:
        # Inline numpy fallback
        padded = np.pad(dem, 1, mode="edge")
        dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * res)
        dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * res)
        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    slope_rad = np.radians(slope_deg)
    tan_slope = np.tan(slope_rad)
    # Clamp tiny slopes to avoid division by zero / very large TWI values
    tan_slope = np.where(tan_slope < 1e-6, 1e-6, tan_slope)

    twi = np.log(area_m2 / tan_slope).astype("float32")
    nodata_mask = np.isnan(dem) | np.isnan(slope_deg)
    twi[nodata_mask] = NODATA_FLOAT

    meta = _ref_meta()
    with rasterio.open(TWI_RASTER, "w", **meta) as dst:
        dst.write(twi, 1)

    log.info("TWI → %s", TWI_RASTER)
    return TWI_RASTER


# ── Distance to stream ────────────────────────────────────────────────────────

def _find_streams_path() -> Path:
    """Return path to NHD flowlines, checking multiple candidate locations."""
    candidates = [
        Path(STREAMS_SHP),
        NHD_RAW_DIR / "nhd_flowlines_harris.gpkg",
        NHD_RAW_DIR / "nhd_flowlines_harris.shp",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"NHD flowlines not found. Tried: {[str(c) for c in candidates]}"
    )


def derive_dist_stream() -> Path:
    if DIST_STREAM_RASTER.exists():
        log.info("dist_stream already exists: %s", DIST_STREAM_RASTER)
        return DIST_STREAM_RASTER

    meta = _ref_meta()
    transform = meta["transform"]
    height = meta["height"]
    width = meta["width"]
    res = abs(transform.a)

    streams_path = _find_streams_path()
    log.info("Reading NHD flowlines from %s ...", streams_path)
    streams = gpd.read_file(streams_path)

    # Reproject to target CRS if needed
    from rasterio.crs import CRS
    target_crs = CRS.from_string(meta["crs"].to_string())
    if streams.crs is None or streams.crs != target_crs:
        streams = streams.to_crs(target_crs)

    # Rasterise flowlines as a binary mask (1 = stream pixel)
    stream_mask = rasterize(
        ((geom, 1) for geom in streams.geometry if geom is not None),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # Euclidean distance transform: distance from nearest stream pixel (in pixels)
    # multiply by res to convert to metres
    dist_pixels = distance_transform_edt(stream_mask == 0)
    dist_m = (dist_pixels * res).astype("float32")

    with rasterio.open(DIST_STREAM_RASTER, "w", **meta) as dst:
        dst.write(dist_m, 1)

    log.info(
        "dist_stream → %s  (max=%.0f m, mean=%.0f m)",
        DIST_STREAM_RASTER, float(dist_m.max()), float(dist_m.mean()),
    )
    return DIST_STREAM_RASTER


# ── Soil ksat ─────────────────────────────────────────────────────────────────

def derive_soil_ksat() -> Path:
    if SOIL_RASTER.exists():
        log.info("soil_ksat already exists: %s", SOIL_RASTER)
        return SOIL_RASTER

    log.info("Reading SSURGO polygons from %s ...", SOIL_GPKG)
    soil = gpd.read_file(SOIL_GPKG)

    log.info("Reading ksat table from %s ...", KSAT_CSV)
    ksat = pd.read_csv(KSAT_CSV)

    # Normalise column names (lowercase, strip whitespace)
    soil.columns = [c.strip().lower() for c in soil.columns]
    ksat.columns = [c.strip().lower() for c in ksat.columns]

    # Identify join key — SSURGO uses mukey
    if "mukey" not in soil.columns:
        # Try common alternatives
        for alt in ("map_unit_key", "mapunitkey"):
            if alt in soil.columns:
                soil = soil.rename(columns={alt: "mukey"})
                break
        else:
            raise KeyError("Cannot find mukey column in SSURGO polygons.")

    if "mukey" not in ksat.columns:
        raise KeyError("Cannot find mukey column in ksat CSV.")

    # Ensure ksat_r column exists
    ksat_col = "ksat_r" if "ksat_r" in ksat.columns else ksat.columns[-1]
    ksat = ksat[["mukey", ksat_col]].rename(columns={ksat_col: "ksat_r"})
    ksat["mukey"] = ksat["mukey"].astype(str).str.strip()
    soil["mukey"] = soil["mukey"].astype(str).str.strip()

    # Aggregate: if multiple components per mukey, take mean
    ksat_agg = ksat.groupby("mukey")["ksat_r"].mean().reset_index()

    soil_merged = soil.merge(ksat_agg, on="mukey", how="left")
    n_missing = soil_merged["ksat_r"].isna().sum()
    if n_missing:
        log.warning("%d soil polygons have no ksat_r — will be NODATA.", n_missing)
    soil_merged["ksat_r"] = soil_merged["ksat_r"].fillna(NODATA_FLOAT)

    # Reproject to target CRS
    meta = _ref_meta()
    from rasterio.crs import CRS
    target_crs = CRS.from_string(meta["crs"].to_string())
    soil_merged = soil_merged.to_crs(target_crs)

    # Rasterise: burn ksat_r values onto 30m grid
    shapes = (
        (row.geometry, float(row.ksat_r))
        for row in soil_merged.itertuples()
        if row.geometry is not None
    )
    ksat_arr = rasterize(
        shapes,
        out_shape=(meta["height"], meta["width"]),
        transform=meta["transform"],
        fill=NODATA_FLOAT,
        dtype="float32",
        merge_alg=rasterio.enums.MergeAlg.replace,
    )

    with rasterio.open(SOIL_RASTER, "w", **meta) as dst:
        dst.write(ksat_arr, 1)

    valid = ksat_arr[ksat_arr != NODATA_FLOAT]
    log.info(
        "soil_ksat → %s  (valid pixels=%d, mean=%.4f μm/s)",
        SOIL_RASTER, len(valid), float(valid.mean()) if len(valid) else 0,
    )
    return SOIL_RASTER


# ── Public API ────────────────────────────────────────────────────────────────

def run() -> dict[str, Path]:
    """Derive all features. Skips steps where output already exists."""
    log.info("=== features.py: deriving engineered features ===")
    results = {
        "slope":       derive_slope(),
        "twi":         derive_twi(),
        "dist_stream": derive_dist_stream(),
        "soil_ksat":   derive_soil_ksat(),
    }
    log.info("=== features.py: done ===")
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
