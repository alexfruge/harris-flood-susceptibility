"""
run_pipeline.py — Processing stage entry point (Week 2–3).

Runs the full pipeline in order:
    1. align.py    — reproject/resample all rasters to 30 m EPSG:32615 grid
    2. features.py — derive slope, TWI, dist_stream, soil_ksat
    3. labels.py   — rasterise FEMA flood zones to class labels
    4. stack       — assemble 7-band feature_stack.tif

Final output: data/processed/feature_stack.tif
    Band 1  elevation       (m)
    Band 2  slope           (degrees)
    Band 3  twi             (dimensionless)
    Band 4  land_cover      (NLCD integer code)
    Band 5  soil_ksat       (μm/s)
    Band 6  dist_stream     (m)
    Band 7  rainfall_mean   (mm/yr)

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --skip rainfall      # skip slow step(s)
    python run_pipeline.py --only align stack   # run subset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

from src.config import (
    DATA_PROCESSED,
    FEATURE_STACK,
    FEATURE_NAMES,
    FLOOD_LABELS_RASTER,
    SLOPE_RASTER,
    TWI_RASTER,
    DIST_STREAM_RASTER,
    SOIL_RASTER,
)

log = logging.getLogger(__name__)

NODATA_FLOAT = -9999.0

# ── Band map: feature name → aligned raster path ─────────────────────────────
def _band_paths() -> dict[str, Path]:
    return {
        "elevation":     DATA_PROCESSED / "dem_harris_aligned.tif",
        "slope":         Path(SLOPE_RASTER),
        "twi":           Path(TWI_RASTER),
        "land_cover":    DATA_PROCESSED / "nlcd_harris_aligned.tif",
        "soil_ksat":     Path(SOIL_RASTER),
        "dist_stream":   Path(DIST_STREAM_RASTER),
        "rainfall_mean": DATA_PROCESSED / "rainfall_mean_aligned.tif",
    }


# ── Step runners ──────────────────────────────────────────────────────────────

def step_align() -> None:
    log.info("━━━  STEP 1 / 4: ALIGN  ━━━")
    from src.processing.align import run
    results = run()
    for name, path in results.items():
        log.info("  %-15s → %s", name, path)


def step_features() -> None:
    log.info("━━━  STEP 2 / 4: FEATURES  ━━━")
    from src.processing.features import run
    results = run()
    for name, path in results.items():
        log.info("  %-15s → %s", name, path)


def step_labels() -> None:
    log.info("━━━  STEP 3 / 4: LABELS  ━━━")
    from src.processing.labels import run
    result = run()
    log.info("  flood_labels → %s", result)


def step_stack(out_path: Path = Path(FEATURE_STACK)) -> None:
    """Assemble the 7-band feature stack from individual aligned rasters."""
    log.info("━━━  STEP 4 / 4: STACK  ━━━")

    if out_path.exists():
        log.info("Feature stack already exists: %s", out_path)
        return

    band_paths = _band_paths()

    # Validate all source rasters exist
    missing = [name for name, p in band_paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot build stack — missing rasters: {missing}. "
            "Run align and features steps first."
        )

    # Use elevation as reference for dimensions/transform
    ref_path = band_paths["elevation"]
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        height, width = ref.height, ref.width
        transform = ref.transform
        crs = ref.crs

    profile.update(
        count=len(FEATURE_NAMES),
        dtype="float32",
        nodata=NODATA_FLOAT,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        for band_idx, feature_name in enumerate(FEATURE_NAMES, start=1):
            src_path = band_paths[feature_name]
            log.info("  Band %d  %-15s ← %s", band_idx, feature_name, src_path.name)

            with rasterio.open(src_path) as src:
                arr = src.read(1)

                # Validate shape
                if arr.shape != (height, width):
                    raise ValueError(
                        f"Shape mismatch for {feature_name}: "
                        f"expected ({height}, {width}), got {arr.shape}. "
                        "Re-run align step."
                    )

                # Convert to float32; preserve nodata
                src_nodata = src.nodata
                arr_f = arr.astype("float32")
                if src_nodata is not None:
                    arr_f[arr == src_nodata] = NODATA_FLOAT

            dst.write(arr_f, band_idx)
            dst.update_tags(band_idx, name=feature_name)

            valid = arr_f[arr_f != NODATA_FLOAT]
            if len(valid):
                log.info(
                    "           min=%.3f  max=%.3f  mean=%.3f  nodata_pct=%.1f%%",
                    float(valid.min()),
                    float(valid.max()),
                    float(valid.mean()),
                    100 * (arr_f == NODATA_FLOAT).mean(),
                )

    log.info("Feature stack written → %s  (%d bands)", out_path, len(FEATURE_NAMES))
    _print_stack_summary(out_path)


def _print_stack_summary(stack_path: Path) -> None:
    """Print a concise summary table for the finished stack."""
    sep = "─" * 65
    log.info(sep)
    log.info("%-4s  %-18s  %8s  %8s  %8s  %8s",
             "Band", "Feature", "Min", "Max", "Mean", "NoData%")
    log.info(sep)
    with rasterio.open(stack_path) as src:
        for i, name in enumerate(FEATURE_NAMES, start=1):
            arr = src.read(i).astype("float64")
            valid = arr[arr != NODATA_FLOAT]
            nodata_pct = 100 * (arr == NODATA_FLOAT).mean()
            if len(valid):
                log.info(
                    "%-4d  %-18s  %8.2f  %8.2f  %8.2f  %7.1f%%",
                    i, name,
                    float(valid.min()), float(valid.max()),
                    float(valid.mean()), nodata_pct,
                )
            else:
                log.info("%-4d  %-18s  %8s  %8s  %8s  %7.1f%%",
                         i, name, "N/A", "N/A", "N/A", nodata_pct)
    log.info(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_STEPS = ["align", "features", "labels", "stack"]

STEP_FNS = {
    "align":    step_align,
    "features": step_features,
    "labels":   step_labels,
    "stack":    step_stack,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Harris County flood susceptibility — processing pipeline."
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=ALL_STEPS,
        metavar="STEP",
        help="Steps to skip (space-separated).",
    )
    p.add_argument(
        "--only", nargs="*", default=None,
        choices=ALL_STEPS,
        metavar="STEP",
        help="Run only these steps (space-separated, in order).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    steps_to_run = args.only if args.only is not None else ALL_STEPS
    steps_to_run = [s for s in ALL_STEPS if s in steps_to_run]  # preserve order
    steps_to_run = [s for s in steps_to_run if s not in (args.skip or [])]

    if not steps_to_run:
        log.warning("No steps selected — nothing to do.")
        return

    log.info("Pipeline steps to run: %s", steps_to_run)

    for step in steps_to_run:
        STEP_FNS[step]()

    log.info("═" * 55)
    log.info("PROCESSING PIPELINE COMPLETE")
    log.info("Feature stack: %s", FEATURE_STACK)
    log.info("Flood labels : %s", FLOOD_LABELS_RASTER)
    log.info("═" * 55)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
