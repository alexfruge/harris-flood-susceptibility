"""
sample.py — Extract valid pixel samples from feature_stack.tif and flood_labels_harris.tif.

Outputs
-------
data/processed/X_samples.npy   : float32 array, shape (n_valid, 7)
data/processed/y_samples.npy   : uint8  array, shape (n_valid,)
data/processed/coords.npy      : int32  array, shape (n_valid, 2)  — (row, col) pixel indices
"""

import numpy as np
import rasterio
import logging
from pathlib import Path

# Allow running as a script from the repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    FEATURE_STACK, FLOOD_LABELS_RASTER,
    DATA_PROCESSED, FEATURE_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NODATA_FEATURES = -9999.0
NODATA_LABELS   = 0

X_OUT     = DATA_PROCESSED / "X_samples.npy"
Y_OUT     = DATA_PROCESSED / "y_samples.npy"
COORD_OUT = DATA_PROCESSED / "coords.npy"


def extract_samples(
    feature_path: Path = FEATURE_STACK,
    label_path:   Path = FLOOD_LABELS_RASTER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read feature stack and labels in lockstep, return valid-pixel arrays.

    Returns
    -------
    X      : (n, 7)  float32
    y      : (n,)    uint8
    coords : (n, 2)  int32  — (row, col) in raster grid
    """
    log.info("Opening feature stack: %s", feature_path)
    with rasterio.open(feature_path) as feat_src:
        n_bands = feat_src.count
        if n_bands != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature stack has {n_bands} bands, expected {len(FEATURE_NAMES)}."
            )
        features = feat_src.read().astype(np.float32)   # (7, H, W)
        feat_nodata = feat_src.nodata if feat_src.nodata is not None else NODATA_FEATURES
        height, width = feat_src.height, feat_src.width
        log.info("  Grid: %d rows × %d cols, %d bands", height, width, n_bands)

    log.info("Opening label raster: %s", label_path)
    with rasterio.open(label_path) as lbl_src:
        if (lbl_src.height, lbl_src.width) != (height, width):
            raise ValueError(
                "Feature stack and label raster grids do not match: "
                f"features=({height},{width}), labels=({lbl_src.height},{lbl_src.width})"
            )
        labels = lbl_src.read(1).astype(np.int32)       # (H, W)

    # ── Build valid-pixel mask ────────────────────────────────────────────────
    # A pixel is valid when:
    #   • no feature band equals the nodata sentinel
    #   • label is not 0 (NoData)
    feat_valid = np.all(features != float(feat_nodata), axis=0)   # (H, W)
    lbl_valid  = labels != NODATA_LABELS                           # (H, W)
    valid_mask = feat_valid & lbl_valid                            # (H, W)

    n_valid = int(valid_mask.sum())
    log.info(
        "Valid pixels: %d / %d  (%.1f %%)",
        n_valid, height * width, 100.0 * n_valid / (height * width),
    )

    # ── Flatten ───────────────────────────────────────────────────────────────
    rows, cols = np.where(valid_mask)
    coords = np.stack([rows, cols], axis=1).astype(np.int32)      # (n, 2)

    # features shape: (7, H, W) → index with valid mask → (n, 7)
    X = features[:, valid_mask].T                                  # (n, 7)
    y = labels[valid_mask].astype(np.uint8)                        # (n,)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    unique_classes = np.unique(y)
    log.info("Class distribution:")
    for cls in unique_classes:
        count = int((y == cls).sum())
        log.info("  Class %d: %d pixels (%.1f %%)", cls, count, 100.0 * count / n_valid)

    return X, y, coords


def save_samples(X: np.ndarray, y: np.ndarray, coords: np.ndarray) -> None:
    np.save(X_OUT,     X)
    np.save(Y_OUT,     y)
    np.save(COORD_OUT, coords)
    log.info("Saved X → %s", X_OUT)
    log.info("Saved y → %s", Y_OUT)
    log.info("Saved coords → %s", COORD_OUT)


def load_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load previously saved sample arrays."""
    X      = np.load(X_OUT)
    y      = np.load(Y_OUT)
    coords = np.load(COORD_OUT)
    log.info(
        "Loaded samples: X=%s  y=%s  coords=%s",
        X.shape, y.shape, coords.shape,
    )
    return X, y, coords


def run() -> None:
    X, y, coords = extract_samples()
    save_samples(X, y, coords)
    log.info("Sampling complete.")


if __name__ == "__main__":
    run()
