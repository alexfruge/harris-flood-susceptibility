"""
labels.py — Rasterise FEMA NFHL flood zones to integer class labels.

Output: data/labels/flood_labels_harris.tif
    3  High      zones A, AE, AH, AO, AR, A99, VE, V, V1-V30, A1-A30
    2  Moderate  zones B, X (shaded / 0.2 % annual-chance), X500
    1  Low       everything else inside Harris County boundary
    0  NoData    outside county or no information

The county boundary is derived from the convex hull of the flood-zone layer
itself (avoids downloading a separate county shapefile).  If a county polygon
is available at DATA_RAW / "county" / "harris_county.gpkg" it is used instead.

Usage
-----
    from src.processing.labels import run
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
from rasterio.crs import CRS

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FLOOD_LABELS_RASTER,
    FEMA_SHP,
    FEMA_HIGH_ZONES,
    FEMA_MODERATE_ZONES,
    CRS as TARGET_CRS_STR,
)

log = logging.getLogger(__name__)

# Canonical path — may be .geojson or .gpkg from flood_labels.py
FEMA_GJ_CANDIDATES = [
    Path(FEMA_SHP),
    DATA_RAW / "fema" / "harris_fema_flood.gpkg",
    # processed labels from flood_labels.py
    Path("data") / "labels" / "fema_flood_zones_harris.geojson",
]

COUNTY_SHP_CANDIDATES = [
    DATA_RAW / "county" / "harris_county.gpkg",
    DATA_RAW / "county" / "harris_county.shp",
]

NODATA_LABEL = 0
CLASS_LOW      = 1
CLASS_MODERATE = 2
CLASS_HIGH     = 3

# ── Expanded high / moderate zone sets (mirrors config + flood_score logic) ──
_A_NUMBERED  = {f"A{i}" for i in range(1, 31)}
_V_NUMBERED  = {f"V{i}" for i in range(1, 31)}
HIGH_ZONES   = FEMA_HIGH_ZONES | _A_NUMBERED | _V_NUMBERED
MODERATE_SUBTYPES = {
    "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
    "0.2 PCT ANNUAL CHANCE FLOOD HAZARD CONTAINED IN CHANNEL",
}


# ── Helper: read reference grid ───────────────────────────────────────────────

def _ref_meta() -> dict:
    dem_aligned = DATA_PROCESSED / "dem_harris_aligned.tif"
    with rasterio.open(dem_aligned) as src:
        meta = src.meta.copy()
    meta.update(
        count=1,
        dtype="uint8",
        nodata=NODATA_LABEL,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    return meta


def _find_fema_path() -> Path:
    for p in FEMA_GJ_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "FEMA flood zones file not found. "
        f"Tried: {[str(c) for c in FEMA_GJ_CANDIDATES]}"
    )


def _load_flood_zones() -> gpd.GeoDataFrame:
    path = _find_fema_path()
    log.info("Reading FEMA flood zones from %s ...", path)
    gdf = gpd.read_file(path)
    # Normalise column names
    gdf.columns = [c.strip() for c in gdf.columns]
    return gdf


def _assign_class(fld_zone: str, zone_subty: str = "") -> int:
    zone = str(fld_zone).strip().upper() if fld_zone == fld_zone else ""
    subty = str(zone_subty).strip().upper() if zone_subty == zone_subty else ""

    if zone in HIGH_ZONES:
        return CLASS_HIGH
    if zone in FEMA_MODERATE_ZONES:
        return CLASS_MODERATE
    if zone == "X":
        # Shaded X (0.2 % annual-chance) → moderate; unshaded → low
        return CLASS_MODERATE if subty in MODERATE_SUBTYPES else CLASS_LOW
    if zone in ("B",):
        return CLASS_MODERATE
    # C, D, open water, etc.
    return CLASS_LOW


def rasterise_labels(
    gdf: gpd.GeoDataFrame,
    meta: dict,
    out_path: Path,
) -> Path:
    """
    Burn FEMA flood zones onto the reference grid as class integers.

    Strategy
    --------
    1. Fill the county footprint with CLASS_LOW (background = low susceptibility).
    2. Burn moderate zones on top (overwrite low pixels).
    3. Burn high zones on top (highest priority).

    This ensures every pixel inside the county boundary has a label.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = CRS.from_string(TARGET_CRS_STR)

    if gdf.crs is None or gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    h, w = meta["height"], meta["width"]
    transform = meta["transform"]

    # ── Step 1: county background → all 1 (Low) ──────────────────────────────
    # Try explicit county polygon; fall back to convex hull of flood zones
    county_path = next((p for p in COUNTY_SHP_CANDIDATES if p.exists()), None)
    if county_path:
        log.info("Using county boundary: %s", county_path)
        county = gpd.read_file(county_path).to_crs(target_crs)
        county_geoms = list(county.geometry)
    else:
        log.info(
            "No county shapefile found — using convex hull of flood zones as county boundary."
        )
        hull = gdf.union_all().convex_hull
        county_geoms = [hull]

    label_arr = rasterize(
        ((geom, CLASS_LOW) for geom in county_geoms if geom is not None),
        out_shape=(h, w),
        transform=transform,
        fill=NODATA_LABEL,
        dtype="uint8",
    )

    # ── Step 2: Compute per-polygon class ────────────────────────────────────
    fld_zone_col = next(
        (c for c in gdf.columns if c.upper() in ("FLD_ZONE", "FLDZONE")), None
    )
    subty_col = next(
        (c for c in gdf.columns if c.upper() == "ZONE_SUBTY"), None
    )

    if fld_zone_col is None:
        # If flood_score column exists (from flood_labels.py), map it directly
        if "flood_score" in gdf.columns:
            log.info("Using pre-computed flood_score column.")
            gdf = gdf.copy()
            gdf["_class"] = gdf["flood_score"].map(
                {5: CLASS_HIGH, 4: CLASS_HIGH, 3: CLASS_HIGH,
                 2: CLASS_MODERATE, 1: CLASS_LOW, 0: NODATA_LABEL}
            ).fillna(CLASS_LOW).astype("uint8")
        else:
            raise KeyError(
                "Cannot find FLD_ZONE or flood_score column in flood zones GeoDataFrame."
            )
    else:
        gdf = gdf.copy()
        gdf["_class"] = [
            _assign_class(
                row[fld_zone_col],
                row[subty_col] if subty_col else "",
            )
            for _, row in gdf.iterrows()
        ]

    # ── Step 3: Burn moderate then high (higher priority wins) ───────────────
    for cls in (CLASS_MODERATE, CLASS_HIGH):
        subset = gdf[gdf["_class"] == cls]
        if subset.empty:
            log.info("No polygons for class %d — skipping.", cls)
            continue
        burned = rasterize(
            ((geom, cls) for geom in subset.geometry if geom is not None),
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        mask = burned > 0
        label_arr[mask] = burned[mask]
        log.info(
            "Burned class %d (%s): %d pixels",
            cls,
            {CLASS_MODERATE: "Moderate", CLASS_HIGH: "High"}[cls],
            int(mask.sum()),
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    unique, counts = np.unique(label_arr, return_counts=True)
    class_names = {0: "NoData", 1: "Low", 2: "Moderate", 3: "High"}
    for u, c in zip(unique, counts):
        pct = 100 * c / label_arr.size
        log.info("  Class %d %-10s : %8d pixels  (%.1f %%)", u, class_names.get(u, "?"), c, pct)

    # ── Write ─────────────────────────────────────────────────────────────────
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(label_arr, 1)
        dst.update_tags(
            band_1="flood_label",
            class_0="NoData",
            class_1="Low",
            class_2="Moderate",
            class_3="High",
        )

    log.info("Flood labels → %s", out_path)
    return out_path


def run(out_path: Path = Path(FLOOD_LABELS_RASTER)) -> Path:
    """Full labels pipeline: load → classify → rasterise → write."""
    if out_path.exists():
        log.info("Flood labels already exist: %s", out_path)
        return out_path

    log.info("=== labels.py: rasterising FEMA flood zones ===")
    meta = _ref_meta()
    gdf  = _load_flood_zones()
    rasterise_labels(gdf, meta, out_path)
    log.info("=== labels.py: done ===")
    return out_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()