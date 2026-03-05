#!/usr/bin/env python3
"""
run_pipeline.py — Entry point 2: Processing, feature engineering, and label generation.

Steps
-----
1. Align all raw rasters to the common 30 m EPSG:32615 grid (processing/align.py)
2. Derive slope, TWI, and distance-to-stream (processing/features.py)
3. Rasterize FEMA flood zones to integer class labels (processing/labels.py)
4. Assemble the feature matrix and save as data/processed/feature_matrix.npz

Usage
-----
    uv run python run_pipeline.py
    uv run python run_pipeline.py --skip-align   # if alignment already done
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Run the Harris County processing pipeline.")
    p.add_argument("--skip-align",    action="store_true", help="Skip raster alignment step.")
    p.add_argument("--skip-features", action="store_true", help="Skip feature derivation step.")
    p.add_argument("--skip-labels",   action="store_true", help="Skip label rasterization step.")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("run_pipeline")
    errors = []

    # Step 1: Align
    if not args.skip_align:
        log.info("=" * 60)
        log.info("STEP 1: Aligning rasters to 30 m EPSG:32615 grid")
        log.info("=" * 60)
        try:
            from src.processing.align import run_all_alignments
            run_all_alignments()
        except Exception as exc:
            log.error("Alignment failed: %s", exc, exc_info=True)
            errors.append("align")

    # Step 2: Feature derivation
    if not args.skip_features:
        log.info("=" * 60)
        log.info("STEP 2: Deriving slope, TWI, distance-to-stream")
        log.info("=" * 60)
        try:
            from src.processing.features import run as run_features
            run_features()
        except Exception as exc:
            log.error("Feature derivation failed: %s", exc, exc_info=True)
            errors.append("features")

    # Step 3: Rasterize labels
    if not args.skip_labels:
        log.info("=" * 60)
        log.info("STEP 3: Rasterizing FEMA flood zones to class labels")
        log.info("=" * 60)
        try:
            from src.processing.labels import run as run_labels
            run_labels()
        except Exception as exc:
            log.error("Label rasterization failed: %s", exc, exc_info=True)
            errors.append("labels")

    # Step 4: Assemble feature matrix
    log.info("=" * 60)
    log.info("STEP 4: Assembling feature matrix")
    log.info("=" * 60)
    try:
        from src.processing.features import assemble_feature_matrix
        assemble_feature_matrix()
    except Exception as exc:
        log.error("Feature matrix assembly failed: %s", exc, exc_info=True)
        errors.append("feature_matrix")

    if errors:
        log.error("Pipeline completed with errors in: %s", errors)
        return 1

    log.info("Pipeline complete — feature matrix ready for modeling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())